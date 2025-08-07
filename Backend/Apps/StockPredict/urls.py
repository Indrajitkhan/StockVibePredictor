from django.urls import path
from .views import predict_stock_trend, redis_check

urlpatterns = [
    path("predict/", predict_stock_trend, name="predict_stock_trend"),
    path("redis-check/", redis_check, name="redis_check"),
]
