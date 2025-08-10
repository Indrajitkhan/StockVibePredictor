from django.urls import path
# Import from enhanced_views for better functionality
from .enhanced_views import predict_stock_trend, redis_check, model_status

urlpatterns = [
    path("predict/", predict_stock_trend, name="predict_stock_trend"),
    path("redis-check/", redis_check, name="redis_check"),
    path("models/status/", model_status, name="model_status"),
]
