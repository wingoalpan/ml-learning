from collections import Counter

import numpy as np
import torch.utils.data as Data
import json as js


def ngrams(sentences, n):
    assert (n in [1, 2, 3, 4]), 'invalid parameter n, n should be a integer between 1 and 4'
    if not isinstance(sentences, list):
        sentences = [sentences]

    grams = Counter()
    for sentence in sentences:
        cur_grams = Counter()
        parts = sentence.lower().strip().split()
        for i in range(1, n + 1):
            count = len(parts) - i + 1
            gram_list = [tuple(parts[t: t + i]) for t in range(count)]
            for gram in gram_list:
                cur_grams[gram] += 1
        grams |= cur_grams
    return grams


def _bleu(candidate, refs, n):
    c_len = len(candidate.strip().split())
    # 取和 candidate长度最接近的 reference 长度
    ref_len_diff = [abs(c_len - len(ref.strip().split())) for ref in refs]
    min_diff_index = ref_len_diff.index(min(ref_len_diff))
    r_len = len(refs[min_diff_index].strip().split())

    candidate_counter = ngrams(candidate, n)
    reference_counter = ngrams(refs, n)
    clip_counter = candidate_counter & reference_counter

    numerator = np.zeros(n)
    denominator = np.zeros(n)
    for counter in clip_counter:
        numerator[len(counter)-1] += clip_counter[counter]
    for counter in candidate_counter:
        denominator[len(counter)-1] += candidate_counter[counter]

    return numerator, denominator, c_len, r_len


def bleu(candidate_list, refs_list, n):
    c_len, r_len = 0, 0
    numerator = np.zeros(n)
    denominator = np.zeros(n)

    for candidate, refs in zip(candidate_list, refs_list):
        _numerator, _denominator, _c_len, _r_len = _bleu(candidate, refs, n)
        numerator += _numerator
        denominator += _denominator
        c_len += _c_len
        r_len += _r_len

    probs = numerator / denominator
    mean_log = np.array([1.0 / n] * n) * np.log(probs)
    base_score = np.exp(np.sum(mean_log))
    brevity_penalty = np.array(1.0) if c_len >= r_len else np.exp(1 - (np.array(r_len) / np.array(c_len)))
    return brevity_penalty * base_score



