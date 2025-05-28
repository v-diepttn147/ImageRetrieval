import os
import numpy as np
import faiss
from PIL import Image
from django.core.management.base import BaseCommand
from django.conf import settings
from posts.models import Post
from posts.search.embedding import extract_embedding

class Command(BaseCommand):
    help = "Build FAISS index from all posts with images."

    def handle(self, *args, **options):
        index_path = os.path.join(settings.BASE_DIR, 'index/index.faiss')
        id_map_path = os.path.join(settings.BASE_DIR, 'index/id_map.npy')
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        embeddings = []
        post_ids = []

        posts = Post.objects.exclude(image='')
        for post in posts:
            img_path = os.path.join(settings.MEDIA_ROOT, post.image.name)
            try:
                image = Image.open(img_path).convert('RGB')
                emb = extract_embedding(image).reshape(-1).astype('float32')
                embeddings.append(emb)
                post_ids.append(post.id)
                self.stdout.write(self.style.SUCCESS(f"✓ {post.image.name}"))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"⚠️ Skipping {post.image.name}: {e}"))

        if not embeddings:
            self.stdout.write(self.style.ERROR("No valid embeddings found."))
            return

        X = np.vstack(embeddings).astype('float32')
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)

        faiss.write_index(index, index_path)
        np.save(id_map_path, np.array(post_ids))

        self.stdout.write(self.style.SUCCESS(f"✅ Index built with {len(post_ids)} images."))
