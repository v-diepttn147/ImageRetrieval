# posts/search/vector_store.py

import faiss
import numpy as np
import os

INDEX_PATH = "index/index.faiss"
ID_MAP_PATH = "index/id_map.npy"

# Load once and cache
faiss_index = faiss.read_index(INDEX_PATH)
id_map = np.load(ID_MAP_PATH)  # Maps FAISS index to Post.id

def search(embedding, k=10):
    D, I = faiss_index.search(embedding.reshape(1, -1), k)
    matched_ids = id_map[I[0]]  # Post IDs
    return matched_ids

def add_to_index(embedding, post_id):
    global faiss_index, id_map

    embedding = embedding.astype('float32').reshape(1, -1)
    faiss_index.add(embedding)

    id_map = np.append(id_map, post_id)
    np.save(ID_MAP_PATH, id_map)
    faiss.write_index(faiss_index, INDEX_PATH)

def add_embedding(embedding, post_id):
    faiss_index.add(embedding.reshape(1, -1).astype('float32'))
    id_map.append(post_id)
    faiss.write_index(faiss_index, INDEX_PATH)
    np.save(ID_MAP_PATH, np.array(id_map))