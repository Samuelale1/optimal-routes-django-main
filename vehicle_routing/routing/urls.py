from django.urls import path
from . import views

urlpatterns = [
    #path('upload/', views.upload_file, name='upload_file'),
    path('', views.landing, name='landing'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('results/<int:route_id>/', views.results, name='results'),
]

