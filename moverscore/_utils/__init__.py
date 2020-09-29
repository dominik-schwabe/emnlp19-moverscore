import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool
from pathlib import Path
from transformers import DistilBertTokenizer

import numpy as np
import requests
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

STOPWORDS = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "d", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "more", "most", "mustn", "my", "myself", "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "wouldn", "y", "you", "your", "yours", "yourself", "yourselves"]


class Download:
    def __init__(self, url, extract_path):
        self.url = url
        self.extract_path = Path(extract_path)
        self.hash_path = self.extract_path.parent / (
            self.extract_path.name + "_hash.json"
        )

    def download_zip(self, logger=None):
        self.extract_path.mkdir(parents=True, exist_ok=True)
        if not self._hash_is_valid():
            if logger is not None:
                logger.info(f"Downloading {self.url} to {self.extract_path}")
            with requests.get(self.url, stream=True) as req:
                req.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(req.content)) as file:
                    file.extractall(self.extract_path)
            self._write_hash()

    def _hash_is_valid(self):
        try:
            with open(self.hash_path, "r") as file:
                validate_hash = json.load(file)["hash"]
                return self._gen_hash() == validate_hash
        except FileNotFoundError:
            return False

    def _write_hash(self):
        with open(self.hash_path, "w") as file:
            json.dump({"hash": self._gen_hash()}, file)

    def _gen_hash(self):
        files = [p for p in Path(self.extract_path).rglob("*") if p.is_file()]
        md5 = hashlib.md5()
        md5.update(bytes(self.url, "utf-8"))
        for file in sorted(files):
            with open(file, "rb") as binary:
                md5.update(binary.read())
        return md5.digest().hex()


def truncate(tokenizer, tokens):
    if isinstance(tokenizer, DistilBertTokenizer):
        max_len = tokenizer.model_max_length
    else:
        max_len = tokenizer.max_len
    upper = max_len - 2
    if len(tokens) > upper:
        tokens = tokens[0:upper]
    return tokens


def process(tokenizer, arr):
    arr = ["[CLS]"] + truncate(tokenizer, tokenizer.tokenize(arr)) + ["[SEP]"]
    arr = tokenizer.convert_tokens_to_ids(arr)
    return set(arr)


def get_idf_dict(arr, tokenizer, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_with_tokenizer = partial(process, tokenizer)

    with Pool(nthreads) as pool:
        idf_count.update(chain.from_iterable(pool.map(process_with_tokenizer, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict


def collate_idf(arr, tokenizer, numericalize, idf_dict, pad="[PAD]", device=DEVICE):
    tokens = [
        ["[CLS]"] + truncate(tokenizer, tokenizer.tokenize(a)) + ["[SEP]"] for a in arr
    ]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def load_ngram(ids, embedding, idf, n, o):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = (
            safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(DEVICE)
        )
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(DEVICE)
    return new_a, new_idf


def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = (
        torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2)
        .add_(x1_norm)
        .clamp_min_(1e-30)
        .sqrt_()
    )
    return res
