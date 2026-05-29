"""
Industry Keyword Generator
--------------------------
Reads a CSV of industries, calls NVIDIA via LangChain,
and outputs CSVs in batches of 75 rows: 1-75.csv, 76-150.csv, etc.

Input CSV  : one column named "industry"
Output CSV : industry | work_key | invest_key

Rate limit : max 38 requests/minute  →  5 industries per batch, ~1.7 s gap between batches
"""

import os
import re
import csv
import time
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

# ── env ────────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise EnvironmentError("NVIDIA_API_KEY not found in .env file.")

# ── config  ────────────────────────────────────────────────────────────────────
INPUT_CSV      = "/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/industries .csv"        # ← CHANGE to your actual CSV path
OUTPUT_DIR     = Path("output_csvs")
MODEL          = 'meta/llama-3.1-8b-instruct'
BATCH_SIZE     = 5                   # industries per API call
OUTPUT_CHUNK   = 75                  # rows per output file
SLEEP_BETWEEN  = 1.7                 # seconds between batches (rate-limit guard)
RETRY_ATTEMPTS = 2
RETRY_DELAY    = 10.0                # seconds before retrying on error

# model params — strict, low temp, no reasoning overhead
TEMPERATURE  = 0.2
MAX_TOKENS   = 1024
TOP_P        = 0.7

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── LLM ────────────────────────────────────────────────────────────────────────
def make_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=MODEL,
        nvidia_api_key=api_key,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )

llm = make_llm()

# ── prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise business-intelligence keyword extractor.
Your ONLY job is to output valid JSON — nothing else. No explanation, no markdown.

Rules:
- Every keyword must be a short noun phrase (2–4 words max).
- NO action phrases. BAD: "manufacturing steel products". GOOD: "steel manufacturing", "metal fabrication".
- NO hallucinated info. NO vague abstractions. NO filler.
- work_key  → what this industry PRODUCES, operates on, deals in, or is known for.
              Keywords describe the domain, output, core activity, and space of the industry.
              Examples: "software products", "cloud services", "financial instruments", "consumer electronics".
- invest_key → CATEGORY-level spend only: what this industry majorly BUYS, needs, or depends on to operate.
              Raw materials, infrastructure, tools, platforms, services, equipment.
              BAD: "hiring more staff", "paying salaries".
              GOOD: "cloud infrastructure", "raw materials", "manufacturing equipment", "logistics software".
- Exactly 10 keywords per list. No duplicates.
- Output ONLY raw JSON starting with { and ending with }. Nothing before or after.
"""

def build_prompt(industries: list) -> str:
    industries_str = "\n".join(f'  - "{i}"' for i in industries)
    return f"""For each industry below, produce exactly 10 work_key and 10 invest_key keywords.

Industries:
{industries_str}

Return this exact JSON structure:
{{
  "results": [
    {{
      "industry": "<industry>",
      "work_key": ["kw1", "kw2", "kw3", "...10 total"],
      "invest_key": ["kw1", "kw2", "kw3", "...10 total"]
    }}
  ]
}}"""

# ── JSON extractor ─────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end])

# ── API call with retry ────────────────────────────────────────────────────────
def call_llm(industries: list) -> list:
    prompt   = build_prompt(industries)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = llm.invoke(messages)
            data     = extract_json(response.content)
            results  = data.get("results", [])
            for r in results:
                for key in ("work_key", "invest_key"):
                    cnt = len(r.get(key, []))
                    if cnt < 10:
                        log.warning("  ⚠  '%s' → %s returned only %d keywords", r["industry"], key, cnt)
            return results
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                log.info("Sleeping %.0fs before retry…", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    log.error("All retries exhausted for: %s", industries)
    return [{"industry": i, "work_key": [], "invest_key": []} for i in industries]

# ── CSV helpers ────────────────────────────────────────────────────────────────
def load_industries(path: str) -> list:
    industries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ind = (
                row.get("industry")
                or row.get("Industry")
                or row.get("INDUSTRY")
                or list(row.values())[0]
            )
            if ind and ind.strip():
                industries.append(ind.strip())
    return industries

def write_chunk(rows: list, chunk_index: int):
    start = (chunk_index - 1) * OUTPUT_CHUNK + 1
    end   = start + len(rows) - 1
    fname = OUTPUT_DIR / f"{start}-{end}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["industry", "work_key", "invest_key"])
        for row in rows:
            work   = " | ".join(row.get("work_key",   []))
            invest = " | ".join(row.get("invest_key", []))
            writer.writerow([row["industry"], work, invest])
    log.info("💾  Saved %s  (%d rows)", fname.name, len(rows))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading industries from: %s", INPUT_CSV)
    all_industries = load_industries(INPUT_CSV)
    total = len(all_industries)
    log.info("Total industries: %d", total)

    batches       = [all_industries[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)
    buffer        = []
    chunk_index   = 1

    for b_idx, batch in enumerate(batches, 1):
        log.info("Batch %d/%d  industries=%s", b_idx, total_batches, batch)
        results = call_llm(batch)
        buffer.extend(results)

        # flush every OUTPUT_CHUNK rows
        while len(buffer) >= OUTPUT_CHUNK:
            write_chunk(buffer[:OUTPUT_CHUNK], chunk_index)
            buffer      = buffer[OUTPUT_CHUNK:]
            chunk_index += 1

        if b_idx < total_batches:
            log.info("  ⏱  sleeping %.1fs…", SLEEP_BETWEEN)
            time.sleep(SLEEP_BETWEEN)

    # flush remainder
    if buffer:
        write_chunk(buffer, chunk_index)

    log.info("✅  All done. Files saved in: %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()