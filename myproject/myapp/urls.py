from django.urls import path
from .views import ImageAnalysisView

urlpatterns = [
    path('analyze/', ImageAnalysisView.as_view(), name='analyze_image'),
]