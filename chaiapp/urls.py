from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),   # Serves index.html as homepage
    path('upload/', views.upload_images, name='upload_images'),
]
