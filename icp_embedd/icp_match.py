import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _load_npy(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True).item()
    return data["vectors"], data["labels"]


def _rank(product_vector: np.ndarray, vectors: np.ndarray, labels: list[str], top_n: int) -> list[dict]:
    scores = cosine_similarity(product_vector, vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [{"label": labels[i], "score": round(float(scores[i]), 4)} for i in top_indices]


def match(product_npy: str, titles_npy: str, industries_npy: str, functions_npy: str, out_path: str = "matches.csv"):
    product_vectors, product_labels = _load_npy(product_npy)
    product_vector = product_vectors[0:1]  # single product, shape (1, dims)

    titles_vectors,     titles_labels     = _load_npy(titles_npy)
    industries_vectors, industries_labels = _load_npy(industries_npy)
    functions_vectors,  functions_labels  = _load_npy(functions_npy)

    titles     = _rank(product_vector, titles_vectors,     titles_labels,     top_n=100)
    industries = _rank(product_vector, industries_vectors, industries_labels, top_n=35)
    functions  = _rank(product_vector, functions_vectors,  functions_labels,  top_n=35)

    # pad shorter lists so all columns have same length
    max_len = max(len(titles), len(industries), len(functions))
    def pad(lst, n): return lst + [{"label": "", "score": ""}] * (n - len(lst))

    titles     = pad(titles,     max_len)
    industries = pad(industries, max_len)
    functions  = pad(functions,  max_len)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "title_score", "industry", "industry_score", "function", "function_score"])
        for t, ind, func in zip(titles, industries, functions):
            writer.writerow([t["label"], t["score"], ind["label"], ind["score"], func["label"], func["score"]])

    print(f"✅  Saved: {out_path}  ({max_len} rows)")


match(
    product_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/product.npy",
    titles_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/titles.npy",
    industries_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/industries.npy",
    functions_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/functions.npy"
)