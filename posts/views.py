from django.shortcuts import render, redirect
from .models import Post
from .forms import PostForm
from .image_search import search_similar_images
from posts.search.embedding import extract_embedding
from posts.search.vector_store import search, add_to_index
from posts.models import Post
from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.shortcuts import redirect
from PIL import Image

def home(request):
    # Only include posts with non-empty image field
    posts = Post.objects.exclude(image='').order_by('-uploaded_at')
    paginator = Paginator(posts, 18)  # 3 per row × 6 rows

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number or 1)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('partials/post_cards.html', {'page_obj': page_obj})
        return JsonResponse({'html': html})

    return render(request, 'home.html', {'page_obj': page_obj})

@login_required
def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user  # ✅ assign user here
            post.save()

            try:
                image = Image.open(post.image.path).convert('RGB')
                embedding = extract_embedding(image)
                add_to_index(embedding, post.id)
            except Exception as e:
                print(f"Error processing image for embedding: {e}")
            return redirect('home')
    else:
        form = PostForm()
        # post = form.save(commit=False)
        # post.user = request.user
        # post.save()

    

    return render(request, 'create_post.html', {'form': form})

def image_search(request):
    # if request.method == 'POST' and 'image' in request.FILES:
    #     image_file = request.FILES['image']
    #     results = search_similar_images(image_file)
    #     return render(request, 'search_results.html', {'results': results})
    # return render(request, 'search.html')
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return render(request, 'search.html', {'error': 'No image uploaded.'})

        try:
            image = Image.open(image_file).convert("RGB")
            embedding = extract_embedding(image)
            post_ids = search(embedding, k=12)
            results = Post.objects.filter(id__in=post_ids)
            return render(request, 'search_results.html', {'results': results})
        except Exception as e:
            return render(request, 'search.html', {'error': f"Failed to process image: {e}"})

    return render(request, 'search.html')

def dev_login(request):
    user = User.objects.get_or_create(username='gau147')[0]
    user.backend = 'django.contrib.auth.backends.ModelBackend'
    login(request, user)
    return redirect('create_post')

# def image_search(request):
#     if request.method == 'POST':
#         image_file = request.FILES['image']
#         image = Image.open(image_file).convert('RGB')
#         embedding = extract_embedding(image)
#         ids = search(embedding, k=10)
#         posts = Post.objects.filter(id__in=ids)
#         return render(request, 'search_results.html', {'results': posts})