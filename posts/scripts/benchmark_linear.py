import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

# === Configuration ===
image_dir = "/Users/gau147/Desktop/Master/Semester5/BigData/Project/socialnet/media/img"
embedding_dim = 512
num_db = 1000
num_q = 100
k = 10

embeddings_path = "image_embeddings.npy"
image_paths_path = "image_paths.npy"

# === Load CLIP Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        return features[0].cpu().numpy()

# === Load and embed images ===
image_paths = [
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_' not in f
]
image_paths = sorted(image_paths)

# all_embeddings = []
# for path in tqdm(image_paths, desc="Embedding images"):
#     try:
#         emb = embed_image(path)
#         all_embeddings.append(emb)
#     except Exception as e:
#         print(f"Skipping {path}: {e}")

# all_embeddings = np.stack(all_embeddings).astype('float32')
all_embeddings = np.load(embeddings_path)
image_paths = np.load(image_paths_path, allow_pickle=True)

# Save to cache
# np.save(embeddings_path, all_embeddings)
# np.save(image_paths_path, image_paths)


# === Split into database and query sets ===
database_vectors = all_embeddings[:-num_q]
query_vectors = all_embeddings[-num_q:]

# === Ground truth using FAISS (IndexFlatL2) ===
# print("Creating FAISS index for ground truth...")
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# print(f"Indexing {len(database_vectors)} database vectors...")
# faiss_index.add(database_vectors)
# print("Indexing complete.")
# print("DB shape:", database_vectors.shape)
# print("Query shape:", query_vectors.shape)
# print("Index dim:", faiss_index.d)
# print(np.isnan(query_vectors).any(), np.isinf(query_vectors).any())
# print(np.isnan(database_vectors).any(), np.isinf(database_vectors).any())
# query_vectors = query_vectors[:10] # Limit to 10 queries for testing

# gt_D, gt_I = faiss_index.search(query_vectors, k)
print("Ground truth search complete.")

# === Brute-force NumPy linear search ===
def l2_distance_search(query_vectors, database_vectors, k=10):
    print("Running NumPy L2 distance search...")
    import time
    start = time.time()
    dists = np.linalg.norm(query_vectors[:, None, :] - database_vectors[None, :, :], axis=2)
    I = np.argsort(dists, axis=1)[:, :k]
    end = time.time()
    return I, (end - start) * 1000 / len(query_vectors), end - start  # ms/query, total time

I_np, avg_time_np, total_time_np = l2_distance_search(query_vectors, database_vectors, k)

# === Evaluate Recall@K ===
# recall_np = np.mean([
#     len(set(I_np[i]) & set(gt_I[i])) / k
#     for i in range(num_q)
# ])

# === Display Results ===
df = pd.DataFrame([{
    "Method": "NumPy L2 (Brute-force)",
    # "Recall@10": round(recall_np, 4),
    "Query Time (ms)": round(avg_time_np, 2),
    "Total Time (s)": round(total_time_np, 2)
}])

print(df)
