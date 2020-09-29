import inspect
import logging
import string
from collections import defaultdict

import numpy as np
import torch
from pyemd import emd_with_flow
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

from ._utils import (DEVICE, batched_cdist_l2, collate_idf, get_idf_dict,
                    safe_divide, truncate)


class MoverScoreV2:
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self):
        model_name = self.MODEL_NAME
        config = DistilBertConfig.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_name, do_lower_case=True
        )
        self.model = DistilBertModel.from_pretrained(model_name, config=config)
        self.model.eval()
        self.model.to(DEVICE)

    @classmethod
    def model_setup(cls):
        model_name = cls.MODEL_NAME
        logger = logging.getLogger(inspect.currentframe().f_code.co_name)
        logger.setLevel(logging.INFO)

        logger.info("begin setup")
        DistilBertConfig.from_pretrained(model_name)
        DistilBertTokenizer.from_pretrained(model_name)
        DistilBertModel.from_pretrained(model_name)
        logger.info("setup done")

    @staticmethod
    def bert_encode(model, x, attention_mask):
        model.eval()
        with torch.no_grad():
            output, x_encoded_layers, _ = model(
                input_ids=x, attention_mask=attention_mask
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
            return defaultdict(lambda: 1.0)
        return get_idf_dict(arr, self.tokenizer)

    def score(
        self, refs, hyps, stop_words=[], batch_size=256, device=DEVICE,
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

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]

            batch_size = len(ref_tokens)
            for i in range(batch_size):
                ref_ids = [
                    k
                    for k, w in enumerate(ref_tokens[i])
                    if w in stop_words or "##" in w or w in set(string.punctuation)
                ]
                hyp_ids = [
                    k
                    for k, w in enumerate(hyp_tokens[i])
                    if w in stop_words or "##" in w or w in set(string.punctuation)
                ]

                ref_embedding[i, ref_ids, :] = 0
                hyp_embedding[i, hyp_ids, :] = 0

                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0

            raw = torch.cat([ref_embedding, hyp_embedding], 1)

            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

            distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()

            for i in range(batch_size):
                c1 = np.zeros(raw.shape[1], dtype=np.float)
                c2 = np.zeros(raw.shape[1], dtype=np.float)
                c1[: len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]) :] = hyp_idf[i]

                c1 = safe_divide(c1, np.sum(c1))
                c2 = safe_divide(c2, np.sum(c2))

                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1 - np.sum(flow * dst)
                preds.append(score)

        return preds
