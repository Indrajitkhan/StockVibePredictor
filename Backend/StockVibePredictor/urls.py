from django.contrib import admin
from django.http import HttpRequest
from django.urls import path, include
from Apps.StockPredict.views import predict_stock_trend

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("Apps.StockPredict.urls")),
]
