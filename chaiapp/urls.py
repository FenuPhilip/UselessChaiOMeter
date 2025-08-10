from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze', views.analyze_chai, name='analyze_chai'),
    #path('upload/', views.upload_images, name='upload_images'),
]
