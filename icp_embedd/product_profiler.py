"""
Product Profiler
----------------
Reads a scraped website text file, extracts structured keyword profile
using NVIDIA LLM, and saves to CSV.

Input  : scraped .txt file
Output : product_profile.csv with columns:
         url | category | what_it_does | works_on | services | department
"""

import os
import re
import csv
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL         = 'meta/llama-3.1-8b-instruct'
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024
TOP_P         = 0.7
MAX_TXT_CHARS = 30000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise product keyword extractor.
Your ONLY job is to output valid JSON — nothing else. No explanation, no markdown, no preamble.

Rules:
- Every keyword must be a short noun phrase (2-4 words max).
- NO action phrases. BAD: "helps teams deploy faster". GOOD: "deployment automation", "CI/CD pipeline".
- Extract ONLY from the website text provided. Do NOT invent or assume anything.
- Exactly 6 keywords per field. No duplicates.
- Output ONLY raw JSON starting with { and ending with }. Nothing before or after.

Fields:
- category     → what type of product/tool/platform this is
- what_it_does → core capabilities and features of the product
- works_on     → what it integrates with, operates on, or is built for
- services     → services, offerings, or solutions it provides
- department   → which teams or departments inside a company use this product
"""

def build_prompt(scraped_text: str) -> str:
    return f"""Website text:
---
{scraped_text}
---

Extract keywords from the above text only. Return this exact JSON:
{{
  "category":     ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
  "what_it_does": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
  "works_on":     ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
  "services":     ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
  "department":   ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"]
}}"""

def extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if model adds them
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # strip think blocks if reasoning model adds them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response:\n{text[:300]}")
    return json.loads(text[start:end])

def profile_product(txt_path: str, out_path: str = "product_profile.csv") -> dict:
    scraped_text = Path(txt_path).read_text(encoding="utf-8")

    if len(scraped_text) > MAX_TXT_CHARS:
        scraped_text = scraped_text[:MAX_TXT_CHARS]
        log.info("Text capped at %d chars", MAX_TXT_CHARS)

    # extract url from header if present
    url = ""
    for line in scraped_text.splitlines()[:10]:
        if line.startswith("Website :"):
            url = line.replace("Website :", "").strip()
            break

    # call LLM
    log.info("Extracting keywords from: %s", txt_path)
    llm = ChatNVIDIA(
        model=MODEL,
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )

    prompt   = build_prompt(scraped_text)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    log.info("Raw response:\n%s", response.content[:400])

    # parse JSON
    data = extract_json(response.content)

    # validate — warn if any field has fewer than 6 keywords
    for field in ("category", "what_it_does", "works_on", "services", "department"):
        cnt = len(data.get(field, []))
        if cnt < 6:
            log.warning("⚠  '%s' returned only %d keywords", field, cnt)

    # build profile row
    profile = {
        "url":          url,
        "category":     " | ".join(data.get("category",     [])),
        "what_it_does": " | ".join(data.get("what_it_does", [])),
        "works_on":     " | ".join(data.get("works_on",     [])),
        "services":     " | ".join(data.get("services",     [])),
        "department":   " | ".join(data.get("department",   [])),
    }

    log.info("Profile:\n%s", json.dumps(profile, indent=2))

    # save CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=profile.keys())
        writer.writeheader()
        writer.writerow(profile)

    log.info("✅  Saved to: %s", out_path)
    return profile


if __name__ == "__main__":
    profile_product(
        txt_path="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/kissflow_com_scraped.txt",
        out_path="product_profile.csv"
    )