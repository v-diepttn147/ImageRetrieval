from django.db import models
from django.contrib.auth.models import User

# class Post(models.Model):
#     image = models.ImageField(upload_to='posts/')
#     caption = models.TextField()
#     uploaded_at = models.DateTimeField(auto_now_add=True)

class Post(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='media/img/')
    caption = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
