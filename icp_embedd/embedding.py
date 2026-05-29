import os
import csv
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _embed(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "END"}
    )
    return np.array([r.embedding for r in response.data])


def _embed_and_save(texts: list[str], labels: list[str], out_path: str):
    log.info("Embedding %d rows → %s", len(texts), out_path)
    # NIM API supports up to 50 texts per call
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        vectors.append(_embed(batch))
        log.info("  %d/%d done", min(i+50, len(texts)), len(texts))
    vectors = np.vstack(vectors)
    np.save(out_path, {"vectors": vectors, "labels": labels})
    log.info("✅  Saved: %s  shape=%s", out_path, vectors.shape)


def embed_industries(csv_path: str, out_path: str = "industries.npy"):
    labels, texts = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["industry"])
            texts.append(f"{row['industry']} | {row['work_key']} | {row['invest_key']}")
    _embed_and_save(texts, labels, out_path)


def embed_titles(csv_path: str, out_path: str = "titles.npy"):
    labels, texts = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["title"])
            texts.append(f"{row['title']} | {row['work_key']} | {row['invest_key']}")
    _embed_and_save(texts, labels, out_path)


def embed_functions(csv_path: str, out_path: str = "functions.npy"):
    labels, texts = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["function"])
            texts.append(f"{row['function']} | {row['keywords']}")
    _embed_and_save(texts, labels, out_path)


def embed_product(csv_path: str, out_path: str = "product.npy"):
    """
    Embeds each product field separately and saves a dict:
    {
        "what_it_does": (vectors, labels),
        "department":   (vectors, labels),
        "works_on":     (vectors, labels),
        "category":     (vectors, labels),
        "services":     (vectors, labels)
    }
    """
    field_data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("url", "product")
            for field in ["what_it_does", "department", "works_on", "category", "services"]:
                text = row.get(field, "")
                if not text:
                    continue
                keywords = [kw.strip() for kw in text.split("|") if kw.strip()]
                if not keywords:
                    continue
                # For each keyword, we need to embed individually.
                # We'll collect all keywords for this field across all rows (only one row usually)
                # But to be generic, we accumulate per field.
                if field not in field_data:
                    field_data[field] = {"keywords": [], "labels": []}
                for kw in keywords:
                    field_data[field]["keywords"].append(kw)
                    field_data[field]["labels"].append(label)
    
    # Now embed each field's keywords separately and save as dict
    result = {}
    for field, data in field_data.items():
        keywords = data["keywords"]
        labels = data["labels"]
        log.info("Embedding field '%s' with %d keywords", field, len(keywords))
        vectors = []
        for i in range(0, len(keywords), 50):
            batch = keywords[i:i+50]
            vectors.append(_embed(batch))
        vectors = np.vstack(vectors)
        result[field] = (vectors, labels)
    
    np.save(out_path, result)
    log.info("✅ Saved per‑field product embeddings to %s", out_path)

# embed_industries("/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/embedd_industries.csv")
# embed_titles("/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/embedd_titles.csv")
# embed_functions("/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/embedd_funcs.csv")
embed_product("/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/product_profile.csv")