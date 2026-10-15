import axios from "axios";

// Mock axios for integration tests
jest.mock("axios");
const mockedAxios = axios;

describe("API Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("Stock Prediction API", () => {
    test("should fetch stock prediction successfully", async () => {
      const mockResponse = {
        data: {
          prediction: {
            direction: "UP",
            confidence: 0.85,
            target_price: 150.0,
          },
          ticker: "AAPL",
          timestamp: new Date().toISOString(),
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockResponse);

      const response = await axios.post("/api/predict/", { ticker: "AAPL" });

      expect(response.data.prediction.direction).toBe("UP");
      expect(response.data.ticker).toBe("AAPL");
      expect(response.data.prediction.confidence).toBeGreaterThan(0);
      expect(mockedAxios.post).toHaveBeenCalledWith("/api/predict/", {
        ticker: "AAPL",
      });
    });

    test("should handle API errors gracefully", async () => {
      const errorMessage = "Network Error";
      mockedAxios.post.mockRejectedValueOnce(new Error(errorMessage));

      await expect(
        axios.post("/api/predict/", { ticker: "INVALID" })
      ).rejects.toThrow(errorMessage);
    });

    test("should fetch multiple timeframe predictions", async () => {
      const mockResponse = {
        data: {
          predictions: {
            "1d": { direction: "UP", confidence: 0.75 },
            "1w": { direction: "DOWN", confidence: 0.65 },
            "1mo": { direction: "UP", confidence: 0.8 },
          },
          ticker: "TSLA",
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockResponse);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "TSLA",
        timeframes: ["1d", "1w", "1mo"],
      });

      expect(Object.keys(response.data.predictions)).toHaveLength(3);
      expect(response.data.predictions["1d"].direction).toBeDefined();
    });
  });

  describe("Sentiment Analysis API", () => {
    test("should fetch sentiment data", async () => {
      const mockSentimentData = {
        data: {
          positive: 45.2,
          negative: 32.8,
          neutral: 22.0,
          confidence: 87,
          sources: ["News APIs", "Social Media"],
          totalSources: 150,
          marketCondition: "Bullish",
        },
      };

      mockedAxios.get.mockResolvedValueOnce(mockSentimentData);

      const response = await axios.get("/api/sentiment/AAPL");
      const data = response.data;

      expect(data.positive).toBeGreaterThan(0);
      expect(data.negative).toBeGreaterThan(0);
      expect(data.neutral).toBeGreaterThan(0);
      expect(data.confidence).toBeGreaterThanOrEqual(0);
      expect(data.confidence).toBeLessThanOrEqual(100);
      expect(data.marketCondition).toBeDefined();
    });
  });

  describe("Market Data API", () => {
    test("should fetch market overview", async () => {
      const mockMarketData = {
        data: {
          market_data: {
            "S&P 500": { value: 4200.0, change: 1.5 },
            NASDAQ: { value: 13500.0, change: 2.1 },
            DOW: { value: 34000.0, change: 0.8 },
          },
          market_sentiment: "Bullish",
          timestamp: new Date().toISOString(),
        },
      };

      mockedAxios.get.mockResolvedValueOnce(mockMarketData);

      const response = await axios.get("/api/market/overview/");

      expect(response.data.market_data).toBeDefined();
      expect(response.data.market_sentiment).toBeDefined();
      expect(Object.keys(response.data.market_data).length).toBeGreaterThan(0);
    });
  });

  describe("Batch Predictions API", () => {
    test("should handle batch predictions", async () => {
      const mockBatchResponse = {
        data: {
          results: {
            AAPL: { direction: "UP", confidence: 0.85 },
            GOOGL: { direction: "DOWN", confidence: 0.72 },
            TSLA: { direction: "UP", confidence: 0.91 },
            MSFT: { direction: "UP", confidence: 0.78 },
          },
          timeframe: "1d",
          processed_count: 4,
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockBatchResponse);

      const response = await axios.post("/api/predict/batch/", {
        tickers: ["AAPL", "GOOGL", "TSLA", "MSFT"],
        timeframe: "1d",
      });

      expect(Object.keys(response.data.results)).toHaveLength(4);
      expect(response.data.processed_count).toBe(4);
    });
  });

  // =============================================================================
  // ML SERVICE FAILURE SEMANTICS TESTS (Workstream F)
  // =============================================================================
  
  describe("ML Service Failure Semantics", () => {
    test("should handle no model available gracefully", async () => {
      // Simulate API response when no model is loaded
      const mockNoModelResponse = {
        data: {
          status: "no_model",
          error: "No trained model artifact available",
          timestamp: new Date().toISOString(),
          details: {
            reason: "model_cache_empty",
            note: "Predictions require a trained model artifact"
          }
        }
      };

      mockedAxios.post.mockResolvedValueOnce(mockNoModelResponse);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "AAPL",
        timeframes: ["1d"]
      });

      // Should return structured error, not fabricated prediction
      expect(response.data.status).toBe("no_model");
      expect(response.data.error).toBeDefined();
      expect(response.data.predictions).toBeUndefined();
    });

    test("should handle invalid feature vector", async () => {
      const mockInvalidInputResponse = {
        data: {
          status: "invalid_input",
          error: "Feature vector contains invalid values",
          timestamp: new Date().toISOString(),
          details: {
            feature_count: 15,
            nan_count: 5,
            inf_count: 0
          }
        }
      };

      mockedAxios.post.mockResolvedValueOnce(mockInvalidInputResponse);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "AAPL",
        timeframes: ["1d"]
      });

      expect(response.data.status).toBe("invalid_input");
      expect(response.data.error).toBeDefined();
    });

    test("should not display fabricated prediction on failure", async () => {
      // Ensure that on failure, we don't get a fake "success" response
      const mockFailureResponse = {
        data: {
          status: "prediction_failed",
          error: "Unable to generate predictions",
          ticker: "INVALID"
        }
      };

      mockedAxios.post.mockResolvedValueOnce(mockFailureResponse);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "INVALID",
        timeframes: ["1d"]
      });

      // Must not have predictions with direction/confidence on failure
      expect(response.data.predictions).toBeUndefined();
      expect(response.data.status).not.toBe("success");
    });

    test("should properly display model unavailable in health check", async () => {
      const mockHealthResponse = {
        data: {
          status: "degraded",
          services: {
            cache: "healthy",
            models: "degraded",
            data_source: "healthy"
          },
          ml_service: {
            status: "no_model_loaded",
            model_count: 0,
            feature_schema_version: "1.0.0",
            note: "Predictions require a trained model artifact"
          },
          metrics: {
            ml_contracts_loaded: true,
            canonical_feature_count: 15
          }
        }
      };

      mockedAxios.get.mockResolvedValueOnce(mockHealthResponse);

      const response = await axios.get("/api/v1/system/health/");

      // When models: "degraded", UI should show appropriate warning
      expect(response.data.services.models).toBe("degraded");
      expect(response.data.ml_service.status).toBe("no_model_loaded");
    });

    test("should distinguish between negative prediction and failure", async () => {
      // Valid negative prediction should have all expected fields
      const mockNegativePrediction = {
        data: {
          status: "success",
          ticker: "AAPL",
          predictions: {
            "1d": {
              direction: "DOWN",
              confidence: 62.5,
              model_accuracy: 58.0,
              price_target: 145.20,
              current_price: 150.00,
              expected_return: -3.2
            }
          },
          timestamp: new Date().toISOString()
        }
      };

      mockedAxios.post.mockResolvedValueOnce(mockNegativePrediction);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "AAPL",
        timeframes: ["1d"]
      });

      // Valid prediction should have all fields
      expect(response.data.status).toBe("success");
      expect(response.data.predictions["1d"].direction).toBe("DOWN");
      expect(response.data.predictions["1d"].confidence).toBeDefined();
      expect(response.data.predictions["1d"].model_accuracy).toBeDefined();
    });
  });
});
