"""
Function Keyword Generator
--------------------------
Reads a CSV of business functions/departments, calls NVIDIA via LangChain,
and outputs CSVs in batches of 75 rows: 1-75.csv, 76-150.csv, etc.

Input CSV  : one column named "function"
Output CSV : function | keywords

Rate limit : max 38 requests/minute  →  5 functions per batch, ~1.7 s gap between batches
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
INPUT_CSV      = "/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/functions.csv"         # ← CHANGE to your actual CSV path
OUTPUT_DIR     = Path("output_csvs")
MODEL          = 'meta/llama-3.1-8b-instruct'
BATCH_SIZE     = 5                   # functions per API call
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
- NO action phrases. BAD: "managing software deployments". GOOD: "software deployment", "release management".
- NO hallucinated info. NO vague abstractions. NO filler.
- keywords → what this business department/function IS, what space it occupies, and how a
             product or vendor would describe serving this department.
             Include: department identity, common tools/platforms used, what products are built for this function.
             BAD: "handles many tasks", "important department".
             GOOD: "CI/CD pipelines", "infrastructure automation", "security compliance", "CRM software".
- Exactly 10 keywords. No duplicates.
- Output ONLY raw JSON starting with { and ending with }. Nothing before or after.
"""

def build_prompt(functions: list) -> str:
    functions_str = "\n".join(f'  - "{f}"' for f in functions)
    return f"""For each business function/department below, produce exactly 10 keywords.

Functions:
{functions_str}

Return this exact JSON structure:
{{
  "results": [
    {{
      "function": "<function>",
      "keywords": ["kw1", "kw2", "kw3", "...10 total"]
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
def call_llm(functions: list) -> list:
    prompt   = build_prompt(functions)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = llm.invoke(messages)
            data     = extract_json(response.content)
            results  = data.get("results", [])
            for r in results:
                cnt = len(r.get("keywords", []))
                if cnt < 10:
                    log.warning("  ⚠  '%s' → keywords returned only %d", r["function"], cnt)
            return results
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                log.info("Sleeping %.0fs before retry…", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    log.error("All retries exhausted for: %s", functions)
    return [{"function": f, "keywords": []} for f in functions]

# ── CSV helpers ────────────────────────────────────────────────────────────────
def load_functions(path: str) -> list:
    functions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            func = (
                row.get("function")
                or row.get("Function")
                or row.get("FUNCTION")
                or list(row.values())[0]
            )
            if func and func.strip():
                functions.append(func.strip())
    return functions

def write_chunk(rows: list, chunk_index: int):
    start = (chunk_index - 1) * OUTPUT_CHUNK + 1
    end   = start + len(rows) - 1
    fname = OUTPUT_DIR / f"{start}-{end}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "keywords"])
        for row in rows:
            keywords = " | ".join(row.get("keywords", []))
            writer.writerow([row["function"], keywords])
    log.info("💾  Saved %s  (%d rows)", fname.name, len(rows))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading functions from: %s", INPUT_CSV)
    all_functions = load_functions(INPUT_CSV)
    total = len(all_functions)
    log.info("Total functions: %d", total)

    batches       = [all_functions[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)
    buffer        = []
    chunk_index   = 1

    for b_idx, batch in enumerate(batches, 1):
        log.info("Batch %d/%d  functions=%s", b_idx, total_batches, batch)
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