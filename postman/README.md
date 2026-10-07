# STOCKVIBEPREDICTOR API - Postman Setup Guide

VERSION: 2.0 <br />
LAST UPDATED: Task 05 (2026-10-07) — paths and retention decision only. The body below is leftover 2024 setup copy and is **not** a verified feature list.

## Current status (KEEP / MOVE)

* **KEEP** the canonical collection, three environments, examples, and scripts under `postman/`.
* Newman is the live API regression suite while Django `tests.py` remains an empty scaffold.
* Task 04 baseline: **15 requests / 17 assertions**, all passing against a live `127.0.0.1:8000` — **prediction HTTP 200 is masked by fallback `.pkl` files** (R-ML-02), not genuine trained models.
* Removed as dead: `postman/Tests/*.json` (not consumed by Newman) and duplicate `docs/API-Endpoints.json`.
* **UNVERIFIED in this guide:** authentication/token endpoints, GitHub Actions / Jenkins CI, staging/production hosts, and expected JSON bodies copied below.
* Local Development needs **no auth**. Ignore credential/token sections unless you add auth later.


---

# TABLE OF CONTENTS

1. OVERVIEW
2. PREREQUISITES
3. INSTALLATION & SETUP
4. API CREDENTIALS
5. ENVIRONMENT CONFIGURATION
6. RUNNING REQUESTS
7. TESTING
8. COMMON USE CASES
9. TROUBLESHOOTING
10. CI/CD INTEGRATION
11. SECURITY NOTES
12. SUPPORT

---

## OVERVIEW

This is a complete API testing suite for StockVibePredictor - a stock market
prediction API. This guide helps you test all API endpoints, whether you're a
developer, tester, or someone learning API testing.

---

## API Structure :

- Predictions (4 endpoints): Multi-timeframe, Batch, Single Stock, Status.
- Models (3 endpoints): Train, List, Create Test.
- Trading (2 endpoints): Simulate Trade, Portfolio.
- Market Data (3 endpoints): Overview, Chart Data, Multi-Chart.
- System (3 endpoints): Health Check, Memory Status, Debug.

---

## PREREQUISITES

### Required :

- Postman installed (https://www.postman.com/downloads/).
- StockVibePredictor API running locally or access to staging/production.
- Basic understanding of REST APIs (GET, POST, etc.).

---

## INSTALLATION & SETUP

### STEP 1: Import Collection

1. Open Postman application.
2. Click Import button (top-left corner).
3. Choose File tab.
4. Navigate to: postman/Collections/.
5. Select: StockVibePredictor_API_v2.postman_collection.json.
6. Click Open → Import.

Alternative: Drag and drop the collection file into Postman window

### STEP 2: Import Environments

1. Click Import again.
2. Select all 3 environment files from postman/Environments/ :
   - Development.postman_environment.json
   - Staging.postman_environment.json
   - Production.postman_environment.json
3. Click Import.

### STEP 3: Verify Import

After importing, you should see:

- Left Sidebar: "StockVibePredictor API v2.0" collection.
- Top-Right: Environment dropdown with all three environments.

---

## API CREDENTIALS

### LOCAL DEVELOPMENT (No Authentication)

If running locally without authentication:

```bash
{
"username": "not_required",
"password": "not_required",
"apiKey": "not_required"
}
```

### DEVELOPMENT WITH AUTHENTICATION

#### Option A: Create Test Credentials

1. Open Django project terminal:
   $ cd StockVibePredictor
   $ python manage.py createsuperuser

2. Follow prompts:
   Username: testuser
   Email: test@example.com
   Password: TestPass123!

#### Option B: Get API Key (If Enabled)

Via Django Admin :

```bash
$ python manage.py runserver
Browse to: http://localhost:8000/admin/
Login → Navigate to "API Keys" → Create new key
```

Via Django Shell :

```bash
$ python manage.py shell
```

```bash
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
user = User.objects.get(username='testuser')
token, created = Token.objects.get_or_create(user=user)
print(f"Your API Token: {token.key}")
```

Via API Endpoint :

```bash
$ curl -X POST http://localhost:8000/api/auth/token/ \
 -H "Content-Type: application/json" \
 -d '{"username": "testuser", "password": "TestPass123!"}'
```

---

## ENVIRONMENT CONFIGURATION

### SELECT ENVIRONMENT

#### 1. Look at top-right corner of Postman.

#### 2. Click Environment dropdown.

#### 3. Select "Development" for local testing.

### EDIT VARIABLES

#### 1. Click eye icon next to environment dropdown.

#### 2. Click Edit to modify values.

### Development Environment Variables :

| Variable   | Current Value  | Description             | Example                   |
| ---------- | -------------- | ----------------------- | ------------------------- |
| protocol   | http           | Use http for local      | http                      |
| host       | localhost      | Your server address     | localhost                 |
| port       | 8000           | Django default port     | 8000                      |
| baseUrl    | Auto-generated | Combines above values   | http://localhost:8000/api |
| username   | testuser       | From credentials setup  | testuser                  |
| password   | TestPass123!   | From credentials setup  | TestPass123!              |
| apiKey     | your-api-key   | From Step 2 (if needed) | abc123xyz                 |
| authToken  | Leave empty    | Auto-filled after login |
| testTicker | AAPL           | Any stock symbol        | AAPL, GOOGL, TSLA         |
| debugMode  | true           | Enable console logs     | true                      |

#### 3. Click Save after editing.

---

## RUNNING REQUESTS

### FIRST REQUEST: Health Check

#### 1. Expand "System" folder in collection.

#### 2. Click on "Health Check".

#### 3. Verify URL shows: [ {{baseUrl}}/system/health/ ].

#### 4. Click Send button.

#### 5. Expected Response :

```bash
   {
   "status": "healthy",
   "services": {
   "cache": "healthy",
   "models": "healthy",
   "data_source": "healthy"
   }
   }
```

### GET AUTHENTICATION TOKEN (If Required) :

#### 1. Create new request or find "Get Auth Token".

#### 2. Set method to POST.

#### 3. URL: [ {{baseUrl}}/auth/token/ ].

#### 4. Body (raw JSON) :

```bash
   {
   "username": "{{username}}",
   "password": "{{password}}"
   }
```

#### 5. Send the request (token auto-saves to environment) :

### MAKE PREDICTION

#### 1. Open Predictions → Multi-Timeframe Prediction

#### 2. Check request body:

```bash
   {
   "ticker": "AAPL",
   "timeframes": ["1d", "1w", "1mo"],
   "include_analysis": true
   }
```

#### 3. Click Send

#### 4. View prediction results

### REQUEST TYPES :

#### GET Requests (Fetching Data) :

- Health Check: GET {{baseUrl}}/system/health/
- List Models: GET {{baseUrl}}/models/list/
- Market Overview: GET {{baseUrl}}/market/overview/

#### POST Requests (Sending Data):

- Predictions: POST {{baseUrl}}/predict/multi-timeframe/
- Train Model: POST {{baseUrl}}/models/train/

---

## TESTING

### INDIVIDUAL REQUEST TESTS :

#### 1. Send any request

#### 2. Click on Tests tab (next to Body)

#### 3. View test results (Pass/Fail)

### COLLECTION RUNNER :

#### 1. Hover over collection name

#### 2. Click three dots → Run Collection

#### 3. Configure:

```bash
   - Environment: Development
   - Iterations: 1
   - Delay: 0ms
```

#### 4. Click Run StockVibePredictor API

#### 5. View results for all endpoints

### COMMAND LINE TESTING (Newman) :

#### Install Newman :

```bash
$ npm install -g newman
```

#### Run tests :

```bash
$ cd postman/Scripts
$ ./run_tests.sh
```

#### Or manually :

```bash
$ newman run ../Collections/StockVibePredictor_API_v2.postman_collection.json \
 -e ../Environments/Development.postman_environment.json
```

---

## COMMON USE CASES

### TESTING STOCK PREDICTIONS :

Request:
POST {{baseUrl}}/predict/multi-timeframe/

```bash
{
"ticker": "AAPL",
"timeframes": ["1d", "1w", "1mo"],
"include_analysis": true
}
```

### Verify:

- predictions object exists.
- confidence > 50%.
- price_target is reasonable.

### TESTING MULTIPLE STOCKS :

#### Request :

```bash
POST {{baseUrl}}/predict/batch/
```

```bash
{
"tickers": ["AAPL", "MSFT", "GOOGL"],
"timeframe": "1d"
}
```

### Verify :

- All tickers have predictions.
- No errors in results.

### GETTING CHART DATA :

### Request :

```bash
GET {{baseUrl}}/market/chart/AAPL/?timeframe=1mo&indicators=sma20,rsi
```

### Verify :

- Data array is not empty.
- Indicators are included.
- Dates are in correct range.

---

## TROUBLESHOOTING

### PROBLEM: "Could not get response"

### Solution:

- Check if server is running: python manage.py runserver.
- Verify URL in environment (should be: http://localhost:8000/api).

### PROBLEM: "401 Unauthorized"

### Solution:

- Get new auth token.
- Check username/password in environment.
- Verify token is being sent in headers.

### PROBLEM: "404 Not Found"

### Solution:

- Check URL path is correct.
- Verify API version (v1 vs no version).
- Ensure endpoint exists in urls.py.

### PROBLEM: "500 Internal Server Error"

### Solution:

- Check Django console for errors.
- Common issues:
  - Missing models in Scripts/Models/
  - Database not migrated
  - Missing dependencies

---

## CI/CD INTEGRATION

### GITHUB ACTIONS :

#### Create .github/workflows/api-tests.yml :

```bash
name: API Tests
on: [push, pull_request]
jobs:
test:
runs-on: ubuntu-latest
steps: - uses: actions/checkout@v2 - name: Install Newman
run: npm install -g newman - name: Run API Tests
run: |
newman run postman/Collections/StockVibePredictor_API_v2.postman_collection.json \
 -e postman/Environments/Development.postman_environment.json \
 --reporters cli,json \
 --reporter-json-export results.json - name: Upload Results
uses: actions/upload-artifact@v2
with:
name: test-results
path: results.json
```

### JENKINS PIPELINE :

```bash
pipeline {
agent any
stages {
stage('API Tests') {
steps {
sh 'npm install -g newman'
sh 'newman run postman/Collections/\*.json -e postman/Environments/Development.postman_environment.json'
}
}
}
}
```

---

## SECURITY NOTES

### NEVER COMMIT :

- Real passwords in environment files
- Production API keys
- Personal authentication tokens

### BEST PRACTICES :

1. Use different credentials for each environment
2. Rotate API keys regularly
3. Use environment variables for sensitive data
4. Create .gitignore entries for local environment files

---

## SUPPORT

### GETTING HELP :

#### If stuck:

1. Check Console (bottom-left in Postman)
2. Review Django server logs
3. Verify environment variables are set
4. Try Health Check endpoint first
5. Run with debugMode: true for more logs

### USEFUL COMMANDS :

Start Django server:

```bash
$ python manage.py runserver
```

Create test models:

```bash
$ python manage.py shell
```

```bash
from Apps.StockPredict.views import create_dummy_models
create_dummy_models()
```

Check API is responding:

```bash
$ curl http://localhost:8000/api/system/health/
```

---

## PRO TIPS

1. Use Variables Everywhere.
   Instead of: "ticker": "AAPL"
   Use: "ticker": "{{testTicker}}"

2. Chain Requests.
   In Tests tab: pm.environment.set("stockPrice", pm.response.json().price);
   Use in next: "target_price": {{stockPrice}}

3. Debug with Console.
   console.log("Response:", pm.response.json());
   console.log("Status:", pm.response.code);

4. Save Common Responses.
   if (pm.response.code === 200) {
   pm.environment.set("lastGoodResponse", pm.response.json());
   }

---

## LEARNING RESOURCES

- Postman Learning Center: https://learning.postman.com/
- API Testing Guide: https://www.postman.com/api-testing/
- Environment Variables: https://learning.postman.com/docs/sending-requests/variables/

### StockVibePredictor API :

- Check /api/ endpoint for API documentation
- Review Apps/StockPredict/views.py for endpoint logic
- See Apps/StockPredict/urls.py for available routes

---

# END OF DOCUMENT
