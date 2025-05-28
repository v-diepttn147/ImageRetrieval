import faiss
import numpy as np
import time
import pandas as pd

# Parameters
d = 512        # Embedding dimension (CLIP ViT-B/32)
nb = 10000     # Number of database vectors
nq = 100       # Number of query vectors
k = 10         # Top-k neighbors to search

# np.random.seed(42)
# database_vectors = np.random.random((nb, d)).astype('float32')
# query_vectors = np.random.random((nq, d)).astype('float32')

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Configuration ===
image_dir = "/Users/gau147/Desktop/Master/Semester5/BigData/Project/socialnet/media/img"  # Set your image folder here
num_db = 1000                      # Number of images to use as database
num_q = 100                        # Number of query images (subset of db or separate)
image_size = 224                  # Input size for CLIP
embedding_dim = 512              # CLIP ViT-B/32 output size

# === Load CLIP Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", size=image_size).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        return features[0].cpu().numpy()

# === Load and embed images ===
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_' not in f]
# image_paths = sorted(image_paths)[:num_db + num_q]

all_embeddings = []
for path in tqdm(image_paths, desc="Embedding images"):
    embedding = embed_image(path)
    all_embeddings.append(embedding)

all_embeddings = np.stack(all_embeddings).astype('float32')

# === Split into database and query sets ===
database_vectors = all_embeddings[:-num_q]
query_vectors = all_embeddings[-num_q:]


# Ground truth using exact search
index_flat = faiss.IndexFlatL2(d)
index_flat.add(database_vectors)
gt_D, gt_I = index_flat.search(query_vectors, k)

# Helper function
def benchmark(index, name):
    start = time.time()
    D, I = index.search(query_vectors, k)
    end = time.time()

    recall = np.mean([
        len(set(I[i]) & set(gt_I[i])) / k
        for i in range(nq)
    ])

    return {
        "Index": name,
        "Recall@10": round(recall, 4),
        "Query Time (ms)": round((end - start) * 1000 / nq, 2),
        "Total Time (s)": round(end - start, 2)
    }

# Results
results = []

# 1. IndexFlatL2 (exact)
results.append(benchmark(index_flat, "IndexFlatL2 (Exact)"))

# 2. IndexIVFFlat
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(database_vectors)
index_ivf.add(database_vectors)
index_ivf.nprobe = 10
results.append(benchmark(index_ivf, "IndexIVFFlat (IVF)"))

# 3. IndexHNSWFlat
index_hnsw = faiss.IndexHNSWFlat(d, 32)
index_hnsw.add(database_vectors)
results.append(benchmark(index_hnsw, "IndexHNSWFlat (HNSW)"))

# 4. IndexIVFPQ
m = 16  # subquantizers
nbits = 8
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index_ivfpq.train(database_vectors)
index_ivfpq.add(database_vectors)
index_ivfpq.nprobe = 10
results.append(benchmark(index_ivfpq, "IndexIVFPQ (PQ)"))

# 5. GPU IndexFlatL2 (if available)
try:
    res = faiss.StandardGpuResources()
    index_flat_gpu = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    index_flat_gpu.add(database_vectors)
    results.append(benchmark(index_flat_gpu, "GPU IndexFlatL2"))
except Exception as e:
    print("Skipping GPU test:", e)

# Display results
df = pd.DataFrame(results)
print(df)

faiss.write_index(index_flat, "index_flatl2.faiss")
faiss.write_index(index_ivf, "index_ivfflat.faiss")
faiss.write_index(index_hnsw, "index_hnswflat.faiss")
faiss.write_index(index_ivfpq, "index_ivfpq.faiss")