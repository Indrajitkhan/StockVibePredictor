from django.urls import path, include
from .views import (
    predict_multi_timeframe,
    batch_predictions,
    predict_stock_trend,
    simulate_trade,
    get_portfolio,
    get_trade_history,
    place_real_trade,
    create_watchlist,
    get_watchlist_predictions,
    market_overview,
    analytics_dashboard,
    get_model_performance,
    system_health,
    redis_check,
)

# API versioning for future compatibility
app_name = "StockPredict"

# Main URL patterns
urlpatterns = [
    path("predict/multi/", predict_multi_timeframe, name="predict_multi_timeframe"),
    path("predict/batch/", batch_predictions, name="batch_predictions"),
    path("predict/", predict_stock_trend, name="predict_stock_trend"),  # Legacy
    path("trading/simulate/", simulate_trade, name="simulate_trade"),
    path("trading/portfolio/", get_portfolio, name="get_portfolio"),
    path("trading/history/", get_trade_history, name="get_trade_history"),
    path("trading/real/", place_real_trade, name="place_real_trade"),  # Future
    path("watchlist/create/", create_watchlist, name="create_watchlist"),
    path(
        "watchlist/predictions/",
        get_watchlist_predictions,
        name="get_watchlist_predictions",
    ),
    path("market/overview/", market_overview, name="market_overview"),
    path("market/analytics/", analytics_dashboard, name="analytics_dashboard"),
    path("system/health/", system_health, name="system_health"),
    path(
        "system/models/performance/",
        get_model_performance,
        name="get_model_performance",
    ),
    path("redis-check/", redis_check, name="redis_check"),
]

# Alternative: Versioned API patterns (optional for future use)
v1_patterns = [
    path("api/v1/", include(urlpatterns)),
]

# Alternative: Organized by functionality (optional structure)
prediction_patterns = [
    path("multi/", predict_multi_timeframe, name="predict_multi_timeframe"),
    path("batch/", batch_predictions, name="batch_predictions"),
    path("single/", predict_stock_trend, name="predict_stock_trend"),
]

trading_patterns = [
    path("simulate/", simulate_trade, name="simulate_trade"),
    path("portfolio/", get_portfolio, name="get_portfolio"),
    path("history/", get_trade_history, name="get_trade_history"),
    path("real/", place_real_trade, name="place_real_trade"),
]

market_patterns = [
    path("overview/", market_overview, name="market_overview"),
    path("analytics/", analytics_dashboard, name="analytics_dashboard"),
]

watchlist_patterns = [
    path("create/", create_watchlist, name="create_watchlist"),
    path("predictions/", get_watchlist_predictions, name="get_watchlist_predictions"),
]

system_patterns = [
    path("health/", system_health, name="system_health"),
    path("models/performance/", get_model_performance, name="get_model_performance"),
    path("redis-check/", redis_check, name="redis_check"),  # Legacy
]

# INFO: Uncomment below for organized URL structure (optional)
# urlpatterns = [
#     path("predict/", include(prediction_patterns)),
#     path("trading/", include(trading_patterns)),
#     path("market/", include(market_patterns)),
#     path("watchlist/", include(watchlist_patterns)),
#     path("system/", include(system_patterns)),
# ]
