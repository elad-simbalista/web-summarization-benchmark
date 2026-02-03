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


---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."


