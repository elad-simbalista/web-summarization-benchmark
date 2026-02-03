# Web Summarization Benchmark

This project explores summarization strategies for noisy, multilingual web content under three competing constraints: **latency**, **quality**, and **cost**. The goal is to design two practical summarization pipelines:

- **Fast**: very low latency, reasonable quality
- **Advanced**: higher quality under moderate latency

All summaries are constrained to a maximum of **1500 characters**.

---

## Strategies

**Fast (Hybrid)**  
A two-stage pipeline where the input document is first compressed using a Lead-TFIDF extractive step, and the reduced content is then rewritten using `gpt-4.1-nano`. This improves quality significantly while keeping latency low.

**Advanced**  
A direct abstractive summarization approach using `gpt-4.1-mini`, which provides the highest overall quality and is robust to noisy inputs.



![Plot](models.png)


## Table 1 — Benchmark Results

API cost is reported as pricing units ($/1M input tokens / $/1M output tokens).  
Actual cost per summary depends on token usage.

| Approach          | Model / Method | Avg Latency (s) | Cost (USD per 1M tokens) | Faithfulness ↑ | Coverage ↑ | Conciseness ↑ | Clarity ↑ | ROUGE-L ↑ |
|-------------------|---|---:|--------------------------|---:|---:|---:|---:|---:|
| Fast              | Lead-TFIDF (extractive) | 0.02 | $0 / $0                  | 3.90 | 2.86 | 2.98 | 3.26 | 0.266 |
| Fast              | TextRank (extractive) | 5.76 | $0 / $0                  | 3.76 | 2.63 | 3.08 | 3.06 | 0.268 |
| **Fast (Hybrid)** | **Lead-TFIDF → GPT-4.1-nano** | **1.10** | **$0.1 / $0.4**          | **4.94** | 4.33 | **4.36** | **4.97** | 0.086 |
| Fast              | GPT-4.1-nano (standalone) | 2.34 | $0.1 / $0.4              | 4.24 | 3.88 | 3.10 | 3.94 | 0.151 |
| **Advanced**      | **GPT-4.1-mini** | 5.04 | **$0.4 / $1.6**          | **5.00** | **4.84** | 4.16 | **4.96** | 0.174 |
| Ceiling (slow)    | GPT-5-nano | 18.76 | $0.05 / $0.4             | 4.71 | 4.53 | 4.24 | 4.61 | 0.196 |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="enter openai key here"

# run
python benchmark.py --data <path to summaries_1k.json> --n <Number of samples to evaluate>


