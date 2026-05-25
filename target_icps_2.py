"""
gtm_profiler.py

Analyzes a product info file (.md) and a leads CSV to produce a structured
GTM targeting profile: ranked target functions, ranked target industries,
ranked target titles, and functions to avoid.

Output: pure JSON, nothing else.
"""

import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_product_info(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def load_leads_summary(csv_path: str) -> str:
    """
    Pulls only the columns relevant for GTM profiling from the CSV
    and returns a compact text summary to stay within token limits.
    """
    df = pd.read_csv(csv_path)

    relevant_cols = [
        "Title", "Departments", "Sub Departments",
        "Seniority", "Industry", "# Employees", "Annual Revenue",
        "Company Name", "Country"
    ]
    # keep only columns that actually exist in this CSV
    cols = [c for c in relevant_cols if c in df.columns]
    df = df[cols].fillna("Unknown")

    lines = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in cols]
        lines.append(" | ".join(parts))

    return "\n".join(lines)



# helper
def load_allowed_options(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── main function ─────────────────────────────────────────────────────────────

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from response, even if surrounded by other text."""
    # Find the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        # Try to find JSON using regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        raise ValueError("No JSON object found")
    
    json_str = text[start:end+1]
    
    # Try to parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try cleaning up common issues
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        return json.loads(json_str)


def generate_gtm_profile(md_path: str, allowed_options_path: str) -> dict:
    """
    Parameters
    ----------
    md_path  : path to the product info markdown file
    allowed_options_path : path to JSON file with allowed values (target_functions, target_industries, target_titles)

    Returns
    -------
    dict with keys:
        target_functions  – array of exactly 25 values from allowed list
        target_industries – array of exactly 25 values from allowed list
        target_titles     – array of exactly 25 values from allowed list
        avoid_functions   – array of exactly 25 values inferred by LLM based on GTM analysis
    """

    product_info = load_product_info(md_path)
    allowed_options = load_allowed_options(allowed_options_path)
    


    SYSTEM_PROMPT = """You are a B2B go-to-market analyst. Your ONLY job is to return a valid JSON object.

    ABSOLUTE RULES — violating any of these means failure:
    1. Return ONLY the JSON object. No explanation, no markdown, no fences.
    2. Output starts with { and ends with }.
    3. For target_functions, target_industries, and target_titles: every value MUST be copied verbatim from the ALLOWED OPTIONS provided. Zero exceptions.
    4. For avoid_functions: generate exactly 25 values based on GTM principles and the product analysis. These should be deterministic roles/functions to avoid targeting based on the product.
    5. target_functions: array of exactly 25 strings, each from allowed target_functions list only.
    6. target_industries: array of exactly 25 strings, each from allowed target_industries list only.
    7. target_titles: array of exactly 25 strings, each from allowed target_titles list only.
    8. avoid_functions: array of exactly 25 strings, generated based on GTM analysis (NOT from an allowed list). These are roles/functions that would not be good targets for this product based on its characteristics.

    JSON schema:
    {
      "target_functions":  ["<allowed_function>", ...],
      "target_industries": ["<allowed_industry>", ...],
      "target_titles":     ["<allowed_title>", ...],
      "avoid_functions":   ["<inferred_function>", ...]
    }"""

    human_prompt = f"""PRODUCT INFORMATION:
    {product_info}

    ALLOWED OPTIONS — you MUST only pick from these exact strings for target_functions, target_industries, and target_titles:

    target_functions allowed values:
    {json.dumps(allowed_options["target_functions"], indent=2)}

    target_industries allowed values:
    {json.dumps(allowed_options["target_industries"], indent=2)}

    target_titles allowed values:
    {json.dumps(allowed_options["target_titles"], indent=2)}

    TASK:
    Read the product info above. Select the best-fit values.
    - target_functions: pick exactly 25 from the allowed target_functions list (copied verbatim)
    - target_industries: pick exactly 25 from the allowed target_industries list (copied verbatim)
    - target_titles: pick exactly 25 from the allowed target_titles list (copied verbatim)
    - avoid_functions: infer exactly 25 GTM-appropriate function/role names that would NOT be good targets for this product based on its characteristics and use case. These should be deterministic roles like Sales, Marketing, Security, HR, etc., that the product is not designed for.

    For the first three fields, only use exact values from the allowed lists above.
    For avoid_functions, generate new values based on GTM analysis - do NOT use the allowed lists.
    Return only the JSON object."""

    llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.0,
    max_completion_tokens=16000,
    model_kwargs={
        "response_format": {"type": "json_object"},
        "reasoning_budget": 0 ,
        "reasoning_effort": 'none'

    }
)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

# remove markdown fences if model adds them
    raw = raw.replace("```json", "").replace("```", "").strip()
    # Try direct JSON parsing first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # If that fails, extract JSON from response
    return extract_json_from_response(raw)


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = generate_gtm_profile(
        md_path="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/Tools/user_website/kissflow_com_scraped.txt",
        allowed_options_path="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/test/titles.json",
    )

    print(json.dumps(result, indent=2))