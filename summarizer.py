"""
Deterministic extractive summarizer to avoid hallucinations.

Functions:
- summarize_memory(text, min_len=30, max_len=140): returns an extractive summary
  composed of the highest-scoring sentences from the input `text`.
"""

import re
from typing import List

_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def _tokens(s: str):
    return [t for t in re.findall(r"\w+", s.lower()) if t]

def summarize_memory(text: str, min_len: int = 30, max_len: int = 140) -> str:
    """
    Extractive summarizer:
    - Splits `text` into sentences
    - Scores sentences by overlap with overall token frequency (simple TF-like)
    - Returns a concatenation of top sentences whose combined char length is between
      min_len and max_len (tries to respect both).
    This is deterministic and only uses the input `text` (no external API or generative model).
    """
    if not text:
        return ""

    text = text.strip()
    if len(text) <= max_len:
        return text

    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return text[:max_len].rstrip()

    tokens = _tokens(text)
    if not tokens:
        return sents[0][:max_len].rstrip()

    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    sent_scores = []
    for i, s in enumerate(sents):
        toks = _tokens(s)
        if not toks:
            score = 0.0
        else:
            score = sum(freq.get(t, 0) for t in toks) / len(toks)
        sent_scores.append((i, s, score, len(s)))

    sent_scores_sorted = sorted(sent_scores, key=lambda x: x[2], reverse=True)

    selected = []
    total_len = 0
    for idx, s, score, slen in sent_scores_sorted:
        if total_len >= min_len:
            break
        selected.append((idx, s))
        total_len += len(s)

    if not selected:
        idx, s, score, slen = sent_scores_sorted[0]
        return s[:max_len].rstrip()

    selected = sorted(selected, key=lambda x: x[0])
    summary = " ".join(s for _, s in selected).strip()

    if len(summary) > max_len:
        if len(selected) > 1:
            selected = selected[:-1]
            summary = " ".join(s for _, s in selected).strip()
            if len(summary) > max_len:
                return summary[:max_len].rstrip()
            return summary
        else:
            return summary[:max_len].rstrip()

    if len(summary) < min_len:
        sel_idxs = {i for i, _ in selected}
        remaining = [ (i,s) for i, s, _, _ in sent_scores_sorted if i not in sel_idxs ]
        remaining = sorted(remaining, key=lambda x: x[0])
        for i, s in remaining:
            if len(summary) >= min_len:
                break
            summary = (summary + " " + s).strip()
            if len(summary) >= max_len:
                break
        if len(summary) > max_len:
            summary = summary[:max_len].rstrip()

    return summary
