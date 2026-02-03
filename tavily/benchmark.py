import json
import os
import time
import random
import re
import argparse
from typing import Dict, Any, List, Optional

from tqdm import tqdm
from openai import OpenAI


SEED = 42
random.seed(SEED)

SUMMARY_CHAR_LIMIT = 1500  # <-- chars (per your correction)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _truncate_to_chars(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _detect_lang_code(text: str) -> Optional[str]:
    """
    Best-effort language detection.
    Returns ISO-like code, e.g. 'en', 'es', 'fr', 'de', ...
    """
    try:
        from langdetect import detect
        t = (text or "").strip()
        if len(t) < 80:
            return None
        return detect(t)
    except Exception:
        return None


def _same_language(src_text: str, out_text: str) -> bool:
    ls = _detect_lang_code(src_text)
    lo = _detect_lang_code(out_text)
    if not ls or not lo:
        return True
    return ls == lo


def clean_web_boilerplate(text: str) -> str:
    """
    Aggressive cleaner for noisy web data.
    Removes navigation links, short fragments, and common footer terms.
    """
    if not text: return ""

    lines = text.split('\n')
    cleaned = []

    junk_triggers = {
        "sign in", "log in", "subscribe", "search", "menu",
        "copyright", "all rights reserved", "privacy policy",
        "terms of use", "skip to content", "loading..."
    }

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        if not line_stripped:
            continue

        if len(line_stripped) < 20 and not line_stripped[-1] in ".!?":
            continue

        if any(trigger in line_lower for trigger in junk_triggers):
            continue

        cleaned.append(line_stripped)

    return "\n".join(cleaned)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if len(s.strip()) > 20]


def lead_tfidf_extractive_summary(
    text: str,
    *,
    lead_sentences: int = 3,
    tfidf_extra_sentences: int = 5,
    max_chars: int = SUMMARY_CHAR_LIMIT,
) -> str:
    if not text:
        return ""

    sents = split_sentences(text)
    if not sents:
        return _truncate_to_chars(text, max_chars)

    lead = sents[:lead_sentences]
    remaining = sents[lead_sentences:]
    selected = list(lead)

    if remaining and tfidf_extra_sentences > 0:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        np.random.seed(SEED)

        vec = TfidfVectorizer()
        X = vec.fit_transform(remaining)
        scores = np.asarray(X.sum(axis=1)).ravel()
        top_idx = scores.argsort()[::-1][:tfidf_extra_sentences]
        for i in sorted(top_idx):
            selected.append(remaining[i])

    out = " ".join(selected)
    return _truncate_to_chars(out, max_chars)


def summarize_with_llm(
    client: OpenAI,
    model: str,
    source_text: str,
    *,
    max_chars: int = SUMMARY_CHAR_LIMIT,
) -> str:
    lang = _detect_lang_code(source_text)
    lang_line = (
        f"Output language must be: {lang}.\n" if lang else "Output language must match the input language.\n"
    )

    prompt = (
        "Summarize the following text.\n"
        "Requirements:\n"
        "- Faithful: do not add facts not supported by the text.\n"
        "- Keep the most important points.\n"
        f"- HARD LIMIT: at most {max_chars} characters (including spaces).\n"
        f"{lang_line}\n"
        "TEXT:\n"
        f"{source_text}"
    )

    resp = client.responses.create(
        model=model,
        temperature=0,
        input=[{"role": "user", "content": prompt}],
    )
    out = (resp.output_text or "").strip()

    return _truncate_to_chars(out, max_chars)


# Hybrid rewrite (nano)
def rewrite_with_nano(
        client: OpenAI,
        extractive_draft: str,
        *,
        max_chars: int = 1500,
) -> str:

    safe_input = _truncate_to_chars(extractive_draft, 2500)

    prompt = (
        "Identify the core topic and 3 key details from the text below.\n"
        "Ignore all navigation menus, ads, headers, and cookie warnings.\n"
        "do not add facts not supported by the text.\n"
        "Keep only the most important points.\n"
        "Output format:\n"
        "- Topic sentence.\n"
        "- Detail 1\n"
        "- Detail 2\n"
        "Stop immediately after.\n\n"
        f"TEXT:\n{safe_input}"
    )

    resp = client.responses.create(
        model="gpt-4.1-nano",
        temperature=0,
        max_output_tokens=75,
        input=[{"role": "user", "content": prompt}],
    )

    out = (resp.output_text or "").strip()
    return _truncate_to_chars(out, max_chars)


def judge_summary(
    client: OpenAI,
    model: str,
    source_text: str,
    baseline: str,
    candidate: str,
) -> Dict[str, Any]:
    prompt = (
        "Score the candidate summary from 1–5 on:\n"
        "faithfulness, coverage, conciseness, clarity.\n"
        "Return JSON only with keys: faithfulness, coverage, conciseness, clarity.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"BASELINE:\n{baseline}\n\n"
        f"CANDIDATE:\n{candidate}"
    )
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    raw = (resp.output_text or "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except Exception:
                parsed = None
        else:
            parsed = None

    if not isinstance(parsed, dict):
        return {
            "faithfulness": None,
            "coverage": None,
            "conciseness": None,
            "clarity": None,
            "_raw": raw[:500],
        }

    out = {}
    for k in ["faithfulness", "coverage", "conciseness", "clarity"]:
        out[k] = parsed.get(k)
    out["_raw"] = raw[:500]
    return out


def rouge_scores(candidate: str, reference: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def run_benchmark(
    json_path: str,
    num_samples: int = 50,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    sample = random.sample(data, min(num_samples, len(data)))

    metrics = {
        "mini": {"latency": [], "judge": [], "rouge": [], "len_chars": [], "lang_ok": []},
        "hybrid": {"latency": [], "judge": [], "rouge": [], "len_chars": [], "lang_ok": []},
    }

    for item in tqdm(sample, desc="Benchmarking"):
        text = clean_web_boilerplate(item["markdown_content"])
        baseline = item["summary"]

        # ---- ADVANCED ----
        t0 = time.perf_counter()
        advanced_input = lead_tfidf_extractive_summary(text, max_chars=20000)
        mini_summary = summarize_with_llm(client, "gpt-4.1-mini", advanced_input, max_chars=SUMMARY_CHAR_LIMIT)
        t1 = time.perf_counter()

        metrics["mini"]["latency"].append(t1 - t0)
        metrics["mini"]["judge"].append(judge_summary(client, "gpt-5-nano", advanced_input, baseline, mini_summary))
        metrics["mini"]["rouge"].append(rouge_scores(mini_summary, baseline))
        metrics["mini"]["len_chars"].append(len(mini_summary))
        metrics["mini"]["lang_ok"].append(_same_language(text, mini_summary))

        # ---- FAST (HYBRID) ----
        t0 = time.perf_counter()
        fast_draft = lead_tfidf_extractive_summary(text, max_chars=4000)
        hybrid_summary = rewrite_with_nano(client, fast_draft, max_chars=SUMMARY_CHAR_LIMIT)
        t1 = time.perf_counter()

        metrics["hybrid"]["latency"].append(t1 - t0)
        metrics["hybrid"]["judge"].append(judge_summary(client, "gpt-5-nano", fast_draft, baseline, hybrid_summary))
        metrics["hybrid"]["rouge"].append(rouge_scores(hybrid_summary, baseline))
        metrics["hybrid"]["len_chars"].append(len(hybrid_summary))
        metrics["hybrid"]["lang_ok"].append(_same_language(text, hybrid_summary))

    def avg(xs):
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else None

    def _to_number(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x = x.strip()
            try:
                return float(x)
            except Exception:
                return None
        return None

    def avg_judge(judges, key):
        vals = []
        for j in judges:
            v = _to_number(j.get(key))
            if v is not None:
                vals.append(v)
        return avg(vals)

    summary = {
        "mini": {
            "avg_latency": avg(metrics["mini"]["latency"]),
            "judge": {k: avg_judge(metrics["mini"]["judge"], k)
                      for k in ["faithfulness", "coverage", "conciseness", "clarity"]},
            "rougeL": avg([r["rougeL_f"] for r in metrics["mini"]["rouge"]]),
            "avg_len_chars": avg(metrics["mini"]["len_chars"]),
            "max_len_chars_observed": max(metrics["mini"]["len_chars"]) if metrics["mini"]["len_chars"] else None,
            "lang_match_rate": avg([1.0 if ok else 0.0 for ok in metrics["mini"]["lang_ok"]]),
        },
        "hybrid": {
            "avg_latency": avg(metrics["hybrid"]["latency"]),
            "judge": {k: avg_judge(metrics["hybrid"]["judge"], k)
                      for k in ["faithfulness", "coverage", "conciseness", "clarity"]},
            "rougeL": avg([r["rougeL_f"] for r in metrics["hybrid"]["rouge"]]),
            "avg_len_chars": avg(metrics["hybrid"]["len_chars"]),
            "max_len_chars_observed": max(metrics["hybrid"]["len_chars"]) if metrics["hybrid"]["len_chars"] else None,
            "lang_match_rate": avg([1.0 if ok else 0.0 for ok in metrics["hybrid"]["lang_ok"]]),
        },
        "constraints": {
            "summary_char_limit": SUMMARY_CHAR_LIMIT,
            "language_detection": "langdetect (best-effort; missing/short text treated as OK)",
        }
    }

    print("\n=== BENCHMARK RESULTS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run summarization benchmark")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to summaries_1k.json"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of samples to evaluate"
    )

    args = parser.parse_args()
    run_benchmark(json_path=args.data, num_samples=args.n)
