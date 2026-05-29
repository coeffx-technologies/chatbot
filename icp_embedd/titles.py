"""
Title Keyword Generator
-----------------------
Reads a CSV of job titles, calls NVIDIA Nemotron-3-Super-120B via LangChain,
and outputs CSVs in batches of 50 rows: 1-50.csv, 51-100.csv, etc.

Input CSV  : one column named "title"
Output CSV : title | work_key | invest_key

Rate limit : max 38 requests/minute  →  3 titles per batch, ~5 s gap between batches
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
INPUT_CSV     = "/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/sample2.csv"        # ← CHANGE to your actual CSV path
OUTPUT_DIR    = Path("output_csvs")
MODEL         = 'meta/llama-3.1-8b-instruct'
BATCH_SIZE    = 5                   # titles per API call
OUTPUT_CHUNK  = 75                  # rows per output file
SLEEP_BETWEEN = 1.7                 # seconds between batches (rate-limit guard)
RETRY_ATTEMPTS = 2
RETRY_DELAY   = 10.0                # seconds before retrying on error

# model params — strict, low temp, no reasoning overhead
TEMPERATURE      = 0.2
MAX_TOKENS       = 1024
TOP_P            = 0.7


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
- NO action phrases. BAD: "defining company vision". GOOD: "company vision", "financial oversight".
- NO hallucinated duties. NO vague abstractions. NO filler.
- work_key  → what this role DOES, builds, manages, oversees, delivers day-to-day.
              Keywords describe the domain, function, output of the role.
- invest_key → CATEGORY-level spend only: software tools, platforms, services, equipment.
              BAD: "seed round legal documentation", "founder salary advance".
              GOOD: "CRM software", "legal counsel", "cloud infrastructure", "recruitment tools".
- Exactly 10 keywords per list. No duplicates.
- Output ONLY raw JSON starting with { and ending with }. Nothing before or after.
"""

def build_prompt(titles: list) -> str:
    titles_str = "\n".join(f'  - "{t}"' for t in titles)
    return f"""For each job title below, produce exactly 10 work_key and 10 invest_key keywords.

Titles:
{titles_str}

Return this exact JSON structure:
{{
  "results": [
    {{
      "title": "<title>",
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
def call_llm(titles: list) -> list:
    prompt   = build_prompt(titles)
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
                        log.warning("  ⚠  '%s' → %s returned only %d keywords", r["title"], key, cnt)
            return results
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                log.info("Sleeping %.0fs before retry…", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    log.error("All retries exhausted for: %s", titles)
    return [{"title": t, "work_key": [], "invest_key": []} for t in titles]

# ── CSV helpers ────────────────────────────────────────────────────────────────
def load_titles(path: str) -> list:
    titles = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (
                row.get("title")
                or row.get("Title")
                or row.get("TITLE")
                or list(row.values())[0]
            )
            if t and t.strip():
                titles.append(t.strip())
    return titles

def write_chunk(rows: list, chunk_index: int):
    start = (chunk_index - 1) * OUTPUT_CHUNK + 1
    end   = start + len(rows) - 1
    fname = OUTPUT_DIR / f"{start}-{end}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "work_key", "invest_key"])
        for row in rows:
            work   = " | ".join(row.get("work_key",   []))
            invest = " | ".join(row.get("invest_key", []))
            writer.writerow([row["title"], work, invest])
    log.info("💾  Saved %s  (%d rows)", fname.name, len(rows))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading titles from: %s", INPUT_CSV)
    all_titles = load_titles(INPUT_CSV)
    total = len(all_titles)
    log.info("Total titles: %d", total)

    batches       = [all_titles[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)
    buffer        = []
    chunk_index   = 1

    for b_idx, batch in enumerate(batches, 1):
        log.info("Batch %d/%d  titles=%s", b_idx, total_batches, batch)
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