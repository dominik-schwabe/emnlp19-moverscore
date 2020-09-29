import inspect
import logging
import os
import string
from pathlib import Path

from collections import defaultdict
import numpy as np
import torch
from pyemd import emd
from torch import nn

from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from ._utils import DEVICE, Download, get_idf_dict, safe_divide, collate_idf, load_ngram, pairwise_distances, get_idf_dict

MOVERSCORE_DIR = Path(os.environ.get("MOVERSCORE", Path("~/.cache/moverscore").expanduser()))

MNLI_BERT_URL = (
    "https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip"
)


plus_mask = lambda x, m: x + (1.0 - m).unsqueeze(-1) * 1e30
minus_mask = lambda x, m: x - (1.0 - m).unsqueeze(-1) * 1e30
mul_mask = lambda x, m: x * m.unsqueeze(-1)
masked_reduce_min = lambda x, m: torch.min(plus_mask(x, m), dim=1, out=None)
masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1, out=None)
masked_reduce_mean = lambda x, m: mul_mask(x, m).sum(1) / (
    m.sum(1, keepdim=True) + 1e-10
)
masked_reduce_geomean = lambda x, m: np.exp(
    mul_mask(np.log(x), m).sum(1) / (m.sum(1, keepdim=True) + 1e-10)
)
idf_reduce_mean = lambda x, m: mul_mask(x, m).sum(1)
idf_reduce_max = lambda x, m, idf: torch.max(
    mul_mask(minus_mask(x, m), idf), dim=1, out=None
)
idf_reduce_min = lambda x, m, idf: torch.min(
    mul_mask(plus_mask(x, m), idf), dim=1, out=None
)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None
    ):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True
        )
        return encoded_layers, pooled_output


class MoverScore:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            MOVERSCORE_DIR, do_lower_case=True
        )
        self.model = BertForSequenceClassification.from_pretrained(MOVERSCORE_DIR, 3)
        self.model.eval()
        self.model.to(DEVICE)

    @staticmethod
    def model_setup():
        MOVERSCORE_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(inspect.currentframe().f_code.co_name)
        Download(MNLI_BERT_URL, MOVERSCORE_DIR).download_zip(logger)

    @staticmethod
    def bert_encode(model, x, attention_mask):
        model.eval()
        x_seg = torch.zeros_like(x, dtype=torch.long)
        with torch.no_grad():
            x_encoded_layers, pooled_output = model(
                x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True
            )
        return x_encoded_layers

    def get_bert_embedding(self, all_sens, idf_dict, batch_size=-1, device=DEVICE):
        tokenizer = self.tokenizer
        model = self.model
        padded_sens, padded_idf, lens, mask, tokens = collate_idf(
            all_sens,
            tokenizer,
            tokenizer.convert_tokens_to_ids,
            idf_dict,
            device=device,
        )

        if batch_size == -1:
            batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(
                    model,
                    padded_sens[i : i + batch_size],
                    attention_mask=mask[i : i + batch_size],
                )
                batch_embedding = torch.stack(batch_embedding)
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def get_idf_dict(self, arr):
        if len(arr) == 1:
            return defaultdict(lambda: 1.)
        return get_idf_dict(arr, self.tokenizer)

    def score(
        self,
        refs,
        hyps,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=256,
        device=DEVICE,
    ):
        idf_dict_ref = self.get_idf_dict(refs)
        idf_dict_hyp = self.get_idf_dict(hyps)
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]

            (
                ref_embedding,
                ref_lens,
                ref_masks,
                ref_idf,
                ref_tokens,
            ) = self.get_bert_embedding(batch_refs, idf_dict_ref, device=device)
            (
                hyp_embedding,
                hyp_lens,
                hyp_masks,
                hyp_idf,
                hyp_tokens,
            ) = self.get_bert_embedding(batch_hyps, idf_dict_hyp, device=device)

            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)

            ref_embedding = torch.cat(
                [ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1
            )
            hyp_embedding = torch.cat(
                [hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1
            )

            for i in range(len(ref_tokens)):
                if remove_subwords:
                    ref_ids = [
                        k
                        for k, w in enumerate(ref_tokens[i])
                        if w not in set(string.punctuation)
                        and "##" not in w
                        and w not in stop_words
                    ]
                    hyp_ids = [
                        k
                        for k, w in enumerate(hyp_tokens[i])
                        if w not in set(string.punctuation)
                        and "##" not in w
                        and w not in stop_words
                    ]
                else:
                    ref_ids = [
                        k
                        for k, w in enumerate(ref_tokens[i])
                        if w not in set(string.punctuation) and w not in stop_words
                    ]
                    hyp_ids = [
                        k
                        for k, w in enumerate(hyp_tokens[i])
                        if w not in set(string.punctuation) and w not in stop_words
                    ]

                ref_embedding_i, ref_idf_i = load_ngram(
                    ref_ids, ref_embedding[i], ref_idf[i], n_gram, 1
                )
                hyp_embedding_i, hyp_idf_i = load_ngram(
                    hyp_ids, hyp_embedding[i], hyp_idf[i], n_gram, 1
                )

                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

                distance_matrix = pairwise_distances(raw, raw)

                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

                c1[: len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i) :] = hyp_idf_i

                c1 = safe_divide(c1, np.sum(c1))
                c2 = safe_divide(c2, np.sum(c2))
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
        return preds
