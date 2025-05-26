from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('create/', views.create_post, name='create_post'),
    path('search/', views.image_search, name='image_search'),
    path('dev-login/', views.dev_login, name='dev_login'),
]
