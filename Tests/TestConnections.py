#!/usr/bin/env python3
"""
EnterpriseTest.py - StockVibePredictor API Testing Suite
Comprehensive test for all enterprise endpoints and features
"""
import requests
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000/api"
TIMEOUT = 30
COLORS = {
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "BOLD": "\033[1m",
    "END": "\033[0m",
}


def print_colored(text, color="WHITE"):
    """Print colored text"""
    print(f"{COLORS[color]}{text}{COLORS['END']}")


def print_header(title):
    """Print test section header"""
    print_colored(f"\n{'='*60}", "CYAN")
    print_colored(f"🧪 {title}", "BOLD")
    print_colored(f"{'='*60}", "CYAN")


def print_result(test_name, success, details=None, response_time=None):
    """Print test result with formatting"""
    status = "✅ PASS" if success else "❌ FAIL"
    color = "GREEN" if success else "RED"

    time_info = f" ({response_time:.2f}s)" if response_time else ""
    print_colored(f"{status} {test_name}{time_info}", color)

    if details:
        print_colored(f"    📋 {details}", "WHITE")


class APITester:
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
        }
        self.auth_token = None

    def record_result(self, test_name, success, details=None, response_time=None):
        """Record test result"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
        else:
            self.results["failed_tests"] += 1

        self.results["test_details"].append(
            {
                "test": test_name,
                "success": success,
                "details": details,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print_result(test_name, success, details, response_time)

    def make_request(
        self, method, endpoint, data=None, headers=None, auth_required=False
    ):
        """Make HTTP request with error handling"""
        url = f"{BASE_URL}{endpoint}"

        # Add auth header if required and available
        if auth_required and self.auth_token:
            if not headers:
                headers = {}
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            start_time = time.time()

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(
                    url, json=data, headers=headers, timeout=TIMEOUT
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time
            return response, response_time

        except requests.exceptions.ConnectionError:
            return None, None
        except requests.exceptions.Timeout:
            return "TIMEOUT", None
        except Exception as e:
            return str(e), None

    def test_system_health(self):
        """Test system health endpoint"""
        print_header("SYSTEM HEALTH TESTS")

        response, response_time = self.make_request("GET", "/system/health/")

        if response is None:
            self.record_result(
                "System Health Check",
                False,
                "Connection failed - Django server not running?",
            )
            return False
        elif response == "TIMEOUT":
            self.record_result("System Health Check", False, "Request timed out")
            return False
        elif isinstance(response, str):
            self.record_result(
                "System Health Check", False, f"Request error: {response}"
            )
            return False

        success = response.status_code == 200
        if success:
            data = response.json()
            status = data.get("status", "unknown")
            services = data.get("services", {})
            details = f"Status: {status}, Services: {services}"
        else:
            details = f"HTTP {response.status_code}: {response.text[:100]}"

        self.record_result("System Health Check", success, details, response_time)
        return success

    def test_legacy_endpoints(self):
        """Test legacy endpoints for backward compatibility"""
        print_header("LEGACY ENDPOINT TESTS")

        # Test Redis check
        response, response_time = self.make_request("GET", "/redis-check/")
        success = response and response.status_code == 200
        details = "Redis connectivity" if success else "Redis connection failed"
        self.record_result("Legacy Redis Check", success, details, response_time)

        # Test legacy prediction
        test_data = {"ticker": "AAPL"}
        response, response_time = self.make_request("POST", "/predict/", test_data)

        if response and response.status_code == 200:
            data = response.json()
            direction = data.get("prediction", {}).get("direction", "unknown")
            details = f"AAPL prediction: {direction}"
            success = True
        else:
            details = "Legacy prediction failed"
            success = False

        self.record_result("Legacy Prediction API", success, details, response_time)

    def test_multi_timeframe_prediction(self):
        """Test multi-timeframe prediction endpoint"""
        print_header("MULTI-TIMEFRAME PREDICTION TESTS")

        # Test single timeframe
        test_data = {"ticker": "AAPL", "timeframes": ["1d"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        success = response and response.status_code == 200
        if success:
            data = response.json()
            predictions = data.get("predictions", {})
            details = f"1d prediction for AAPL: {predictions.get('1d', {}).get('direction', 'unknown')}"
        else:
            details = "Single timeframe prediction failed"

        self.record_result("Single Timeframe (1d)", success, details, response_time)

        # Test multiple timeframes
        test_data = {"ticker": "TSLA", "timeframes": ["1d", "1w", "1mo"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        success = response and response.status_code == 200
        if success:
            data = response.json()
            predictions = data.get("predictions", {})
            timeframe_count = len(predictions)
            details = f"TSLA predictions for {timeframe_count} timeframes"
        else:
            details = "Multi-timeframe prediction failed"

        self.record_result(
            "Multi-Timeframe (1d,1w,1mo)", success, details, response_time
        )

        # Test with analysis
        test_data = {"ticker": "GOOGL", "timeframes": ["1d"], "include_analysis": True}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        success = response and response.status_code == 200
        if success:
            data = response.json()
            analysis = data.get("analysis", {})
            has_technical = "technical" in analysis
            has_risk = "risk" in analysis
            details = f"Analysis included: technical={has_technical}, risk={has_risk}"
        else:
            details = "Analysis prediction failed"

        self.record_result("Prediction with Analysis", success, details, response_time)

        # Test international stock (NIFTY)
        test_data = {"ticker": "NIFTY", "timeframes": ["1d"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        success = response and response.status_code == 200
        if success:
            data = response.json()
            normalized = data.get("normalized_ticker", "")
            details = f"NIFTY normalized to: {normalized}"
        else:
            details = "International stock prediction failed"

        self.record_result(
            "International Stock (NIFTY)", success, details, response_time
        )

    def test_batch_predictions(self):
        """Test batch prediction endpoint"""
        print_header("BATCH PREDICTION TESTS")

        test_data = {"tickers": ["AAPL", "GOOGL", "TSLA", "MSFT"], "timeframe": "1d"}
        response, response_time = self.make_request(
            "POST", "/predict/batch/", test_data
        )

        success = response and response.status_code == 200
        if success:
            data = response.json()
            results = data.get("results", {})
            successful_predictions = len(
                [r for r in results.values() if "direction" in r]
            )
            details = f"Batch: {successful_predictions}/{len(test_data['tickers'])} successful"
        else:
            details = "Batch prediction failed"

        self.record_result("Batch Predictions", success, details, response_time)

    def test_market_intelligence(self):
        """Test market intelligence endpoints"""
        print_header("MARKET INTELLIGENCE TESTS")

        # Market overview
        response, response_time = self.make_request("GET", "/market/overview/")
        success = response and response.status_code == 200
        if success:
            data = response.json()
            market_data = data.get("market_data", {})
            sentiment = data.get("market_sentiment", "unknown")
            details = f"Market sentiment: {sentiment}, indices: {len(market_data)}"
        else:
            details = "Market overview failed"

        self.record_result("Market Overview", success, details, response_time)

        # Analytics dashboard
        response, response_time = self.make_request("GET", "/market/analytics/")
        success = response and response.status_code == 200
        if success:
            data = response.json()
            system_metrics = data.get("system_metrics", {})
            details = f"System uptime: {system_metrics.get('uptime', 'unknown')}"
        else:
            details = "Analytics dashboard failed"

        self.record_result("Analytics Dashboard", success, details, response_time)

    def test_trading_simulation(self):
        """Test paper trading endpoints (requires auth simulation)"""
        print_header("TRADING SIMULATION TESTS")

        # Test without auth first (should fail gracefully)
        test_data = {"ticker": "AAPL", "action": "buy", "quantity": 10}
        response, response_time = self.make_request(
            "POST", "/trading/simulate/", test_data
        )

        # This should fail due to authentication, but gracefully
        if response and response.status_code in [401, 403]:
            details = "Authentication required (expected)"
            success = True
        elif response and response.status_code == 200:
            details = "Trading simulation working (auth disabled)"
            success = True
        else:
            details = "Trading simulation endpoint error"
            success = False

        self.record_result("Trading Simulation Auth", success, details, response_time)

        # Test portfolio endpoint
        response, response_time = self.make_request("GET", "/trading/portfolio/")

        if response and response.status_code in [401, 403]:
            details = "Portfolio authentication required (expected)"
            success = True
        elif response and response.status_code == 200:
            details = "Portfolio endpoint working (auth disabled)"
            success = True
        else:
            details = "Portfolio endpoint error"
            success = False

        self.record_result("Portfolio Access", success, details, response_time)

    def test_model_performance(self):
        """Test model performance and monitoring"""
        print_header("MODEL PERFORMANCE TESTS")

        # Model performance endpoint
        response, response_time = self.make_request(
            "GET", "/system/models/performance/?timeframe=1d"
        )
        success = response and response.status_code == 200
        if success:
            data = response.json()
            accuracy = data.get("metrics", {}).get("accuracy", 0)
            details = (
                f"Model accuracy: {accuracy:.2%}"
                if accuracy
                else "Performance data available"
            )
        else:
            details = "Model performance endpoint failed"

        self.record_result("Model Performance Metrics", success, details, response_time)

    def test_error_handling(self):
        """Test error handling and edge cases"""
        print_header("ERROR HANDLING TESTS")

        # Invalid ticker
        test_data = {"ticker": "INVALID123"}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and response.status_code in [400, 404]
        details = (
            "Invalid ticker properly rejected" if success else "Error handling failed"
        )
        self.record_result("Invalid Ticker Handling", success, details, response_time)

        # Missing ticker
        test_data = {}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and response.status_code == 400
        details = "Missing ticker properly rejected" if success else "Validation failed"
        self.record_result("Missing Ticker Validation", success, details, response_time)

        # Invalid timeframe
        test_data = {"ticker": "AAPL", "timeframes": ["invalid"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and response.status_code == 200  # Should default to 1d
        details = (
            "Invalid timeframe handled gracefully"
            if success
            else "Timeframe validation failed"
        )
        self.record_result(
            "Invalid Timeframe Handling", success, details, response_time
        )

    def test_performance_stress(self):
        """Test system performance under load"""
        print_header("PERFORMANCE STRESS TESTS")

        # Multiple rapid requests
        start_time = time.time()
        successful_requests = 0
        total_requests = 5

        for i in range(total_requests):
            test_data = {"ticker": f"AAPL", "timeframes": ["1d"]}
            response, _ = self.make_request("POST", "/predict/multi/", test_data)
            if response and response.status_code == 200:
                successful_requests += 1

        total_time = time.time() - start_time
        avg_time = total_time / total_requests

        success = (
            successful_requests >= total_requests * 0.8
        )  # 80% success rate acceptable
        details = f"{successful_requests}/{total_requests} requests succeeded, avg: {avg_time:.2f}s"

        self.record_result("Rapid Fire Requests", success, details, total_time)

    def generate_report(self):
        """Generate comprehensive test report"""
        print_header("TEST SUMMARY REPORT")

        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0

        print_colored(f"📊 Total Tests: {total}", "BOLD")
        print_colored(f"✅ Passed: {passed}", "GREEN")
        print_colored(f"❌ Failed: {failed}", "RED")
        print_colored(f"📈 Success Rate: {success_rate:.1f}%", "CYAN")

        if success_rate >= 90:
            print_colored("\n🎉 EXCELLENT! Your API is working perfectly!", "GREEN")
        elif success_rate >= 75:
            print_colored("\n✅ GOOD! Most features are working well.", "YELLOW")
        elif success_rate >= 50:
            print_colored("\n⚠️ PARTIAL! Some features need attention.", "YELLOW")
        else:
            print_colored("\n❌ CRITICAL! Major issues found.", "RED")

        # Save detailed report
        report_file = Path("api_test_report.json")
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print_colored(f"\n📄 Detailed report saved: {report_file}", "BLUE")

        # Recommendations
        print_colored("\n🔧 NEXT STEPS:", "PURPLE")

        if failed > 0:
            print_colored(
                "   1. Check Django server is running: python manage.py runserver",
                "WHITE",
            )
            print_colored(
                "   2. Verify all models are trained: python TrainModel.py full",
                "WHITE",
            )
            print_colored("   3. Check Redis is running (if using caching)", "WHITE")
            print_colored("   4. Review failed tests in the detailed report", "WHITE")
        else:
            print_colored(
                "   1. Start your frontend: cd Frontend && npm start", "WHITE"
            )
            print_colored("   2. Your API is ready for production! 🚀", "WHITE")


def main():
    """Main test execution"""
    print_colored("🧪 StockVibePredictor Enterprise API Test Suite", "BOLD")
    print_colored("=" * 60, "CYAN")
    print_colored(f"🌐 Testing API at: {BASE_URL}", "BLUE")
    print_colored(f"⏱️  Request timeout: {TIMEOUT}s", "BLUE")
    print_colored(
        f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "BLUE"
    )

    tester = APITester()

    # Run all test suites
    if not tester.test_system_health():
        print_colored(
            "\n⚠️ System health check failed. Some tests may not work properly.", "RED"
        )

    tester.test_legacy_endpoints()
    tester.test_multi_timeframe_prediction()
    tester.test_batch_predictions()
    tester.test_market_intelligence()
    tester.test_trading_simulation()
    tester.test_model_performance()
    tester.test_error_handling()
    tester.test_performance_stress()

    # Generate final report
    tester.generate_report()

    return tester.results["failed_tests"] == 0


if __name__ == "__main__":
    success = main()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
