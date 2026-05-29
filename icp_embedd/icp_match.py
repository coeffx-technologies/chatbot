import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_npy(path: str):
    """Loads .npy file – can be either old format (dict with 'vectors','labels') or new per‑field dict."""
    data = np.load(path, allow_pickle=True).item()
    return data

def rank_with_max(product_vectors: np.ndarray, input_vectors: np.ndarray, input_labels: list[str], top_n: int):
    sim_matrix = cosine_similarity(input_vectors, product_vectors)
    best_scores = np.max(sim_matrix, axis=1)
    top_indices = np.argsort(best_scores)[::-1][:top_n]
    return [{"label": input_labels[i], "score": round(float(best_scores[i]), 4)} for i in top_indices]

def match(product_npy: str,
          titles_npy: str,
          industries_npy: str,
          functions_npy: str,
          out_path: str = "matches.csv"):

    # Load product per‑field data
    prod_data = load_npy(product_npy)   # dict: field -> (vectors, labels)
    
    # Extract vectors for each input type according to mapping
    # Title → what_it_does + department
    title_vectors = []
    title_labels = []
    for field in ["what_it_does", "department"]:
        if field in prod_data:
            v, lbl = prod_data[field]
            title_vectors.append(v)
            title_labels.extend(lbl)  # labels not really needed for similarity, but we keep for completeness
    title_vectors = np.vstack(title_vectors) if title_vectors else np.empty((0,0))
    
    # Industry → works_on + category
    ind_vectors = []
    for field in ["works_on", "category"]:
        if field in prod_data:
            v, _ = prod_data[field]
            ind_vectors.append(v)
    ind_vectors = np.vstack(ind_vectors) if ind_vectors else np.empty((0,0))
    
    # Function → services + department
    func_vectors = []
    for field in ["services", "department"]:
        if field in prod_data:
            v, _ = prod_data[field]
            func_vectors.append(v)
    func_vectors = np.vstack(func_vectors) if func_vectors else np.empty((0,0))
    
    # Load input vectors (old format)
    titles_data = np.load(titles_npy, allow_pickle=True).item()
    titles_vec, titles_lbl = titles_data["vectors"], titles_data["labels"]
    
    ind_data = np.load(industries_npy, allow_pickle=True).item()
    ind_vec, ind_lbl = ind_data["vectors"], ind_data["labels"]
    
    func_data = np.load(functions_npy, allow_pickle=True).item()
    func_vec, func_lbl = func_data["vectors"], func_data["labels"]
    
    # Rank
    titles_rank = rank_with_max(title_vectors, titles_vec, titles_lbl, top_n=100) if title_vectors.size > 0 else []
    ind_rank    = rank_with_max(ind_vectors,    ind_vec,    ind_lbl,    top_n=35)  if ind_vectors.size > 0 else []
    func_rank   = rank_with_max(func_vectors,   func_vec,   func_lbl,   top_n=35)  if func_vectors.size > 0 else []
    
    # Pad to same length
    max_len = max(len(titles_rank), len(ind_rank), len(func_rank))
    def pad(lst, n): return lst + [{"label": "", "score": ""}] * (n - len(lst))
    titles_rank = pad(titles_rank, max_len)
    ind_rank    = pad(ind_rank,    max_len)
    func_rank   = pad(func_rank,   max_len)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "title_score", "industry", "industry_score", "function", "function_score"])
        for t, ind, func in zip(titles_rank, ind_rank, func_rank):
            w.writerow([t["label"], t["score"], ind["label"], ind["score"], func["label"], func["score"]])
    
    print(f"✅ Saved: {out_path}  ({max_len} rows)")

if __name__ == "__main__":
    match(
        product_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/npys/product.npy",
        titles_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/npys/titles.npy",
        industries_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/npys/industries.npy",
        functions_npy="/media/prince/5A4E832F4E83034D/Rocketsteer/REST_API/icp_embedd/npys/functions.npy"
    )