# posts/scripts/build_index.py

import os
import sys
import numpy as np
import faiss
import django
from PIL import Image
from django.conf import settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "socialnet.settings")
django.setup()
from posts.models import Post
from posts.search.embedding import extract_embedding

INDEX_PATH = os.path.join(settings.BASE_DIR, 'index/index.faiss')
ID_MAP_PATH = os.path.join(settings.BASE_DIR, 'index/id_map.npy')
VECTOR_DIM = 2048  # ResNet-50

def build_faiss_index():
    posts = Post.objects.exclude(image='')

    embeddings = []
    post_ids = []

    for post in posts:
        img_path = os.path.join(settings.MEDIA_ROOT, post.image.name)
        try:
            image = Image.open(img_path).convert('RGB')
            emb = extract_embedding(image).astype(np.float32)
            print(f"✓ {post.image.name} → shape: {emb.shape}, dtype: {emb.dtype}, type: {type(emb)}")
            embeddings.append(emb)
            post_ids.append(post.id)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    if not embeddings:
        print("❌ No valid images found.")
        return

    # Create FAISS index
    index = faiss.IndexFlatL2(VECTOR_DIM)
    embeddings_np = np.vstack(embeddings).astype('float32')
    print("→ Final batch dtype:", embeddings_np.dtype)  # must be float32

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(embeddings_np)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(ID_MAP_PATH, np.array(post_ids))

    print(f"✅ Built index with {len(post_ids)} images.")

if __name__ == "__main__":
    # import django
    # os.environ.setdefault("DJANGO_SETTINGS_MODULE", "socialnet.settings")
    # django.setup()
    build_faiss_index()
