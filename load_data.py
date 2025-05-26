import os
import django
import csv
import random
from django.core.files import File

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'socialnet.settings')
django.setup()

from posts.models import Post
from django.contrib.auth.models import User

# Constants
CSV_PATH = 'media/captions.csv'
IMG_FOLDER = 'media/img/'

# Step 1: Create some random users
user_list = []
for i in range(5):
    username = f'user{i+1}'
    user, _ = User.objects.get_or_create(username=username)
    user_list.append(user)

# Step 2: Read the CSV and add posts
with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_file = row['Image File'] + '.jpg'
        caption = row['Caption'] or ''

        img_path = os.path.join(IMG_FOLDER, os.path.basename(img_file))
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        with open(img_path, 'rb') as f:
            post = Post(user=random.choice(user_list), caption=caption)
            post.image.save(os.path.basename(img_path), File(f), save=True)
            print(f"Saved: {img_path}")
