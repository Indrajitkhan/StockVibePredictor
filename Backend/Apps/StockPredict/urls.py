from django.urls import path
from .views import predict_stock_trend

urlpatterns = [
    path('predict/', predict_stock_trend, name='predict_stock_trend'),
]
