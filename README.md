<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=300&section=header&text=StockVibePredictor&fontSize=90&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Stock%20Market%20Prediction%20System&descAlignY=51&descAlign=62" alt="header" />

<br/>

![StockVibe Banner](https://img.shields.io/badge/🚀_StockVibePredictor-AI_Powered_Trading-FF6B6B?style=for-the-badge&labelColor=4ECDC4&color=45B7D1)

# 📊 **StockVibePredictor**

### _🤖 Where AI Meets Wall Street 🏦_

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&pause=1000&color=45B7D1&center=true&vCenter=true&width=600&lines=Full-Stack+ML+Trading+Platform+🤖;Real-Time+Stock+Predictions+📈;Built+with+Django+%2B+React+⚛️;Powered+by+Random+Forest+AI+🌲" alt="Typing SVG" />

<br/>

[![Author](https://img.shields.io/badge/👨‍💻_Author-Dibakar-FF6B6B?style=for-the-badge&logo=github)](https://github.com/ThisIsDibakar)
[![License](https://img.shields.io/badge/📜_License-MIT_claimed_no_LICENSE_file-lightgrey?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/🐍_Python-3.11+-FFD93D?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/🎯_Django-5.0-092E20?style=for-the-badge&logo=django)](https://djangoproject.com)
[![React](https://img.shields.io/badge/⚛️_React-18.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Redis](https://img.shields.io/badge/🔥_Redis-Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![ML](https://img.shields.io/badge/🧠_ML-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/🔥_Status-Active-success?style=for-the-badge)](https://github.com/dibakar/StockVibePredictor)

<br/>

<pre align="center">
╔═══════════════════════════════════════════════════════════════╗
║  🎯 AI-Powered Stock Market Prediction System                ║
║  📅 Created with ❤️ • August 2025                            ║
║  🚀 Full-Stack Machine Learning Web Application              ║
╚═══════════════════════════════════════════════════════════════╝
</pre>

</div>

---

## Repository status (Task 06)

This repository is a **revived college prototype**, not a production trading platform.
No real capital, brokerage, wallet, or payment code is active.

| Status           | Meaning                                                                                                                                                                          |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CURRENT**      | Demonstrated locally: Django API under `api/`, CRA/CRACO React app under `ui/`, SQLite (`api/db.sqlite3`), live yfinance chart/company data, Newman API regression (`postman/`). |
| **EXPERIMENTAL** | Prediction endpoints exist but currently load **import-generated fallback `.pkl` files**, not independently trained artifacts.                                                   |
| **UNVERIFIED**   | Historical accuracy numbers, Redis-as-required, Next.js, Docker/Heroku/Vercel, production auth, and rate-limit/SLA claims.                                                       |
| **PLANNED**      | Investment-management platform direction in `docs/PRODUCT-VISION.md`. Live execution remains disabled.                                                                           |

Authoritative structure and facts: [`docs/README.md`](docs/README.md), [`docs/RUNTIME-BASELINE.md`](docs/RUNTIME-BASELINE.md), [`docs/CLEANUP-REGISTER.md`](docs/CLEANUP-REGISTER.md).

---

<div align="center">

## 🧩 **Built With**

<table>
<tr>
<td align="center" width="33%">

### 🧠 **Machine Learning**

![TensorFlow](https://img.shields.io/badge/Random_Forest-217346?style=flat-square&logo=forest&logoColor=white)
![Scikit](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

</td>
<td align="center" width="33%">

### ⚙️ **Backend**

![Django](https://img.shields.io/badge/Django-092E20?style=flat-square&logo=django&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)

</td>
<td align="center" width="33%">

### 🎨 **Frontend**

![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![CRA](https://img.shields.io/badge/Create_React_App-CRACO-09D3AC?style=flat-square)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)

</td>
</tr>
</table>

</div>

---

<div align="center">

## ✨ **Key Features**

</div>

| Feature                                   | Status       | Description                                                                                                                                                |
| :---------------------------------------- | :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔄 **Market data**                        | CURRENT      | Chart/company endpoints use yfinance locally.                                                                                                              |
| 🤖 **Predictions**                        | EXPERIMENTAL | Endpoints exist; they currently load import-generated fallback `.pkl` files, not independently trained models. Historical accuracy numbers are UNVERIFIED. |
| 📈 **Charts**                             | CURRENT      | Chart.js in the CRA frontend.                                                                                                                              |
| 🔌 **REST API**                           | CURRENT      | Django REST routes in `api/app/StockPredict/urls.py`. Canonical regression: `postman/`.                                                                   |
| 💰 **Live trading / brokerage / wallets** | NOT PRESENT  | Intentionally disabled. `/trading/real/` is a stub.                                                                                                        |

---

<div align="center">

## 🛠️ **Technology Stack**

<img src="https://skillicons.dev/icons?i=python,django,react,nodejs,git&theme=dark" />

<br/>

| **Layer**          | **Technology**                                                               | **Status**                                                |
| :----------------- | :--------------------------------------------------------------------------- | :-------------------------------------------------------- |
| 🎨 **Frontend**    | React 18 + CRA (`react-scripts` 5) + CRACO + Chart.js                        | **CURRENT**                                               |
| ⚙️ **Backend**     | Django 5.x + Django REST Framework                                           | **CURRENT**                                               |
| 🧠 **ML**          | scikit-learn VotingClassifier / Random Forest in `api/app/api/scripts/TrainModel.py` | **EXPERIMENTAL** — no genuine trained artifacts on disk   |
| 📊 **Data Source** | yfinance                                                                     | **CURRENT** for chart/company endpoints                   |
| 💾 **Database**    | SQLite (`api/db.sqlite3`)                                                    | **CURRENT** local baseline; MongoDB is **PLANNED**        |
| 🔥 **Cache**       | Redis (optional in settings)                                                 | **UNVERIFIED** as a required runtime                      |
| 🚀 **Deployment**  | none in this repository                                                      | **PLANNED** — no Dockerfile, compose file, or CI workflow |

</div>

---

<div align="center">

## 📦 **Project Architecture**

</div>

<details>
<summary><b>🗂️ Click to Expand Full Project Structure</b></summary>

```text
StockVibePredictor/
├── api/                         # Django 5 + DRF application
│   ├── StockVibePredictor/       # settings, urls, wsgi/asgi
│   ├── app/StockPredict/        # models, views, urls, CompanyEssentials
│   ├── app/api/scripts/          # TrainModel.py + model artifacts (gitignored)
│   ├── app/api/logs/             # runtime logs (gitignored)
│   ├── Makefile                  # backend commands
│   ├── manage.py
│   └── requirements.txt
├── ui/                          # Create React App + CRACO
│   ├── public/
│   ├── src/                       # JSX components under src/components/js/
│   ├── craco.config.js
│   ├── Makefile                  # frontend commands
│   └── package.json
├── docs/                        # current + historical documentation
│   ├── README.md                  # documentation index
│   ├── CLEANUP-REGISTER.md        # Task 05–07 cleanup decisions
│   ├── RUNTIME-BASELINE.md
│   └── archive/                   # college PDFs (historical)
├── tests/                       # Jest unit/integration + ad-hoc Python scripts
│   ├── backend/
│   ├── e2e/
│   ├── integration/
│   ├── unit/
│   └── SetupTests.jsx
├── postman/                     # Newman API regression (retained)
│   ├── Collections/
│   ├── Environments/
│   ├── Examples/
│   ├── Scripts/
│   └── README.md
├── .gitignore
├── AUTHORS.md
├── Makefile                     # root orchestrator (delegates to api/ and ui/)
├── pyproject.toml               # packaging + pytest config
└── README.md
```

**Not in this tree (intentionally):** `LICENSE` (MIT is claimed in badges/metadata but no license file exists), Docker/compose, CI workflows, `.github/` (removed in Task 08), `Deployment/`, `Security/`, `ui/Apps/Dashboard`, `api/setup.py`. Runtime artifacts such as `api/.venv/`, `ui/node_modules/`, `api/db.sqlite3`, logs, and fallback `*.pkl` models are gitignored, not source.

</details>

### Make commands

The repository now has a three-level Makefile system: the root `Makefile` orchestrates, `api/Makefile` owns backend/Django commands, and `ui/Makefile` owns frontend/React commands (architecture adopted from SHADE-ai/VisionCore).

```bash
make help            # list root targets (delegation map)
make install         # backend deps (api/.venv) + frontend deps
make setup           # install + apply database migrations
make dev             # backend (127.0.0.1:8000) + frontend (3000) together
make check           # Django system checks
make test            # backend suite + Jest suite
make build           # frontend production bundle (ui/build/)
make api-regression  # Newman API regression (backend must be running)
make logs            # tail the backend log
make status          # backend/frontend readiness
make stop            # stop dev servers on ports 8000/3000
make clean           # SAFE cleanup: caches, ui/node_modules/, ui/build/ only
```

`make clean` never touches `api/.env`, `api/db.sqlite3`, `api/app/api/logs/`, `api/app/api/scripts/` (models/performance/validation), source code, or Git metadata.

Backend- and frontend-specific targets live in `api/Makefile` and `ui/Makefile` — run `make help` inside those directories.

---

<div align="center">

## 💻 **Code Standards**

| Language          | Standards            | Tools                     |
| :---------------- | :------------------- | :------------------------ |
| 🐍 **Python**     | PEP 8, Type Hints    | `flake8`, `black`, `mypy` |
| 📜 **JavaScript** | ESLint Configuration | `eslint`, `prettier`      |
| 🔄 **Git**        | Conventional Commits | `commitizen`, `husky`     |

</div>

---

<div align="center">

# 🚀 **COMPLETE SETUP & RUNNING GUIDE**

</div>

## 📋 **Prerequisites Installation**

<details>
<summary><b>🔧 Click to View Installation Commands</b></summary>

### **Install Required Software First:**

```bash
# 🐍 Install Python 3.8+
# Mac:
brew install python@3.11

# Windows: Download from python.org
# Linux:
sudo apt update
sudo apt install python3.11 python3-pip

# 📦 Install Node.js (for frontend)
# Mac:
brew install node

# Windows: Download from nodejs.org
# Linux:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 🔥 Install Redis
# Mac:
brew install redis

# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Linux:
sudo apt update
sudo apt install redis-server
```

</details>

---

## 🔻 **Step 1: Clone & Setup Project**

```bash
# 📥 Clone the repository
git clone https://github.com/DibakarArchives/StockVibePredictor.git
cd StockVibePredictor

# 👀 Check you're in the right place
ls
# You should see: api/, ui/, docs/, tests/, postman/, README.md, etc...
```

---

## 🔴 **Step 2: Redis (OPTIONAL / UNVERIFIED)**

Local Task 02–04 baselines ran Django against SQLite **without Redis as a required service**. Redis is listed in settings; treat it as optional unless you are specifically exercising cache endpoints.

```bash
# 🖥️ Terminal 1 - Start Redis:
# Mac/Linux:
redis-server

# Windows:
redis-server.exe

# ✅ You should see:
# - Ready to accept connections
# - The server is now ready to accept connections on port 6379

# 🧪 Verify Redis is Running (Optional):
# In another terminal:
redis-cli ping
# Should return: PONG
```

> ⚠️ **KEEP THIS TERMINAL OPEN** - Redis must stay running!

---

## ⚙️ **Step 3: Backend Setup (Django)**

### **🐍 Terminal 2 - Backend Setup:**

```bash
# 📂 Navigate to project
cd StockVibePredictor

# 🌟 Create virtual environment
python -m venv venv

# 🔌 Activate environment
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# ✅ You should see (venv) in your terminal prompt
```

### **📦 Install Python Dependencies:**

```bash
# 📥 Install all dependencies
pip install -r api/requirements.txt
```

### **🔧 Setup Django Backend:**

```bash
# 📂 Navigate to api/
cd api

# 🔐 Create .env file for settings
echo "DEBUG=True" > .env
echo "SECRET_KEY=your-secret-key-here" >> .env
echo "REDIS_URL=redis://localhost:6379/0" >> .env

# 🗄️ Run migrations
python manage.py migrate
python manage.py makemigrations StockPredict
python manage.py migrate

# 👤 Create superuser (optional, for admin panel)
python manage.py createsuperuser

# 📁 Collect static files
python manage.py collectstatic --noinput
```

> 📝 **NOTE:** For Django-Secret-Key, contact the Owner

### **🚀 Start Django Server:**

```bash
# 🎯 Make sure you're in the api/ directory
python manage.py runserver

# ✅ You should see:
# Starting development server at http://127.0.0.1:8000/
# Quit the server with CONTROL-C.
```

<div align="center">

✅ **Backend is running at:** http://127.0.0.1:8000
⚠️ **KEEP THIS TERMINAL OPEN!**

</div>

---

## 🧠 **Step 4: Train ML Models (UNVERIFIED / HISTORICAL)**

There are **no independently trained model artifacts** in git. Importing `views.py` can persist dummy `universal_model_*.pkl` files under gitignored `api/app/api/scripts/Models/`. Those files are **not** a training success. The commands and accuracy numbers below are leftover college-era README copy and must not be treated as a current baseline.

### **🤖 Terminal 3 - Model Training:**

```bash
# 📂 Navigate to Scripts directory
cd StockVibePredictor/api/app/api/scripts

# 🔌 Activate virtual environment (if not already active)
# Mac/Linux (canonical backend venv lives at api/.venv — Task 08):
source ../../../.venv/bin/activate
# Windows:
..\..\..\.venv\Scripts\activate

# 🎯 Train models - Smart modes available (v5.0 ENHANCED):

# 🚀 RECOMMENDED: Essential training with validation
python TrainModel.py essential
# Trains: 1d, 1w, 1mo, 1y with pre-flight validation
# Features: Auto-validation, retry logic, quality scoring
# Duration: ~20-30 mins, Success rate: 99%+

# ⚡ Quick training (fastest)
python TrainModel.py minimal
# Trains: 1d, 1w, 1mo (covers 60% of use cases, ~10-15 mins)

# 📈 Extended training (thorough)
python TrainModel.py extended
# Trains: 1d, 5d, 1w, 1mo, 3mo, 1y (covers 95% of use cases, ~45-60 mins)

# 🏆 Complete training (everything)
python TrainModel.py complete
# Trains: All 9 timeframes (100% coverage, ~90-120 mins)

# 🔧 Skip validation for faster start (not recommended)
python TrainModel.py essential --no-validate

# 🤖 Use different model type
python TrainModel.py essential --model-type random_forest
```

### **🔧 Additional Training Options:**

```bash
# 🔍 Pre-Training Validation
python TrainModel.py validate
# Validates all timeframe configurations before training
# Checks: Data availability, API connectivity, minimum samples
# Creates: validation report in Validation/ folder

# 🏥 System Health Check
python TrainModel.py health
# Comprehensive system diagnostics:
# ✓ Directory structure
# ✓ Dependencies installed
# ✓ API connectivity
# ✓ Model inventory

# 🧹 Model Cleanup
python TrainModel.py cleanup
# Removes old model versions, keeps latest 3
python TrainModel.py cleanup --keep-models 5  # Keep latest 5

# 📤 Export Model Metadata
python TrainModel.py export
# Exports comprehensive model metadata to JSON
# Includes: versions, accuracy, training time, file sizes

# 🧪 Test Specific Timeframe
python TrainModel.py test 5d
# Tests data fetching for specific timeframe
# Shows: Data points, quality score, issues

# 🎯 Train Single Model with Options
python TrainModel.py single AAPL 1mo
python TrainModel.py single AAPL 1mo --model-type random_forest

# 🏃 Performance Benchmark
python TrainModel.py benchmark
# Quick benchmark on key models
# Shows: Training time, accuracy, performance
```

### **📊 Progress Monitoring:**

```bash
# 📊 You'll see enhanced progress logs:
========================================================================
🚀 STOCKVIBEPREDICTOR MODEL TRAINING SYSTEM v5.0
📅 Build Date: 2024-08-20
========================================================================
📋 Running pre-training validation...
🔍 VALIDATING TIMEFRAME CONFIGURATIONS
  ✅ PASS 1d: 721 samples
  ✅ PASS 5d: 320 samples (fixed!)
  ✅ PASS 1w: 65 samples
📊 Validation Summary: 9/9 timeframes valid

🎯 Training Mode: essential
⏱️ Timeframes: ['1d', '1w', '1mo', '1y']
🤖 Model Type: ensemble

📊 PHASE 1: TRAINING UNIVERSAL MODELS
✅ Universal 1d: Success (Accuracy: 52.56%)
✅ Universal 1w: Success (Accuracy: 54.02%)

🏢 PHASE 2: TRAINING INDIVIDUAL STOCK MODELS
🏷️ Category: mega_cap_tech
📊 Stocks: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
  Training AAPL_1d...
    ✅ Success (Accuracy: 38.71%)

🎉 TRAINING COMPLETED SUCCESSFULLY
📈 Total Models: 94
✅ Successful: 94
❌ Failed: 0
🎯 Success Rate: 100.0%
⏰ Duration: 2m 20s

📊 PERFORMANCE DISTRIBUTION:
  🌟 Excellent (≥60%): 25
  ✅ Good (≥55%): 18
  ⚠️ Acceptable (≥45%): 38
  ❌ Poor (<45%): 13

💾 Summary saved: training_summary_20240820_104536.json
```

### **✅ Model Verifications:**

```bash
# 📁 Check Models with versioning
ls Models/
# Now shows versioned models:
# universal_model_1d_v3fa2b1c9.pkl
# AAPL_model_1mo_va8c3d2f1.pkl
# (Version hash ensures unique identification)

# 📊 Model validation with details
python TrainModel.py check
# Shows each model with version and accuracy

# 📈 Comprehensive summary
python TrainModel.py summary
# Enhanced output:
# ================================================================================
# 📊 MODEL PORTFOLIO SUMMARY
# ================================================================================
# Total Models: 94
# Average Accuracy: 53.2%
# Accuracy Range: 16.0% - 76.0%
# Total Size: 245.3 MB
#
# 📈 Accuracy by Timeframe:
#   1d      : 48.5% ± 9.2% (20 models)
#   1w      : 52.3% ± 7.8% (20 models)
#   1mo     : 50.1% ± 11.3% (20 models)
#
# 🏆 Top 10 Models:
#   MSFT     1mo  : 76.0% (v3fa2b1c9)
#   JPM      1y   : 71.9% (va8c3d2f1)
```

### **📈 Training Mode Comparison:**

|   Mode    | Timeframes | Models |  Duration  | Accuracy |      Features      |     Best For     |
| :-------: | :--------: | :----: | :--------: | :------: | :----------------: | :--------------: |
|  minimal  |     3      |  ~20   | 10-15 min  |   ~50%   | Validation + Retry |     Testing      |
| essential |     4      |  ~35   | 20-30 min  |   ~53%   | Full v5.0 Features |   Recommended    |
| extended  |     6      |  ~55   | 45-60 min  |   ~54%   | Full v5.0 Features |    Production    |
| complete  |     9      |  ~94   | 90-100 min |   ~55%   | Full v5.0 Features | Maximum coverage |

### **🔧 Configuration Options:**

```bash
# Customize training behavior
--no-validate       # Skip pre-training validation
--model-type        # Choose: ensemble (default) or random_forest
--keep-models       # Number of versions to keep in cleanup (default: 3)

# Examples:
python TrainModel.py essential --no-validate --model-type random_forest
python TrainModel.py cleanup --keep-models 10
```

### **📊 Data Quality Monitoring:**

```bash
# Quality scores are now tracked (0-100):
# 100: Perfect data
# 80-99: Excellent quality
# 60-79: Good quality
# 40-59: Acceptable quality
# <40: Poor quality (logged as warning)

# Issues detected and reported:
# • Missing values > 5%
# • Duplicate timestamps
# • Data gaps
# • Extreme price moves (>20%)
# • High zero volume days (>10%)
```

### **🚀 Workflow Recommendations:**

- Pre-Flight Check :

```bash
python TrainModel.py health      # System check
python TrainModel.py validate    # Timeframe validation
```

- Training :

```bash
python TrainModel.py essential   # With auto-validation
```

- Post-Training :

```bash
python TrainModel.py summary     # Review results
python TrainModel.py export      # Export metadata
```

- Maintenance :

```bash
python TrainModel.py cleanup     # Weekly
python TrainModel.py health      # Monthly
```

### **💡 Pro Tips:**

- ✅ Always run with validation (default) for reliability.
- 🔄 Automatic retry logic handles temporary API failures.
- 📊 Data quality scores help identify problematic stocks.
- 🏷️ Model versioning prevents accidental overwrites.
- 🧹 Regular cleanup keeps storage optimized.
- 📤 Export metadata for external analysis.
- 🏥 Health checks prevent silent failures.
- 📈 Performance categories help prioritize improvements.

### **🔍 Troubleshooting:**

```bash
# Check system health first
python TrainModel.py health

# Validate specific timeframe
python TrainModel.py test 5d

# View detailed validation report
cat Validation/timeframe_validation_*.json

# Check model versions
ls Models/*AAPL* | sort -t_ -k4

# View latest training summary
cat Performance/training_summary_latest.json | python -m json.tool
```

---

## 💻 **Step 5: Frontend Setup (React / CRA + CRACO)**

The UI is Create React App (`react-scripts` 5) with CRACO, not Next.js. The API base is hardcoded as `http://localhost:8000/api` in `ui/src/App.jsx` — there is no `NEXT_PUBLIC_API_URL`.

Port 3000 must be free (`craco.config.js` pins it). If another process holds 3000, the dev server will fail (R-FE-01).

### **⚛️ Terminal 4 - Frontend:**

```bash
# 📂 Navigate to ui/ directory
cd StockVibePredictor/ui

# 📦 Install dependencies
npm install

# If you get errors, try:
npm install --legacy-peer-deps

# 🚀 Start the development server
npm start

# ✅ You should see:
# Local: http://localhost:3000
# ready - started server on 0.0.0.0:3000
```

<div align="center">

✅ **Frontend is running at:** http://localhost:3000
⚠️ **KEEP THIS TERMINAL OPEN!**

</div>

---

## 🧪 **Step 6: Test with Postman**

<details>
<summary><b>📮 Click to View Postman Setup</b></summary>

### **📥 Install & Setup Postman:**

```bash
# Download Postman from: https://www.postman.com/downloads/
# Or use web version: https://web.postman.co/
```

### **📋 Import Endpoints to Postman:**

1. Open Postman
2. Click "Import" → "Raw text"
3. Import `postman/Collections/StockVibePredictor_API_v2.postman_collection.json`
4. Click "Continue" → "Import"

### **🧪 Test Basic Endpoints:**

```bash
# 1. First check system health
GET http://127.0.0.1:8000/api/system/health/

# 2. Check Redis connection
GET http://127.0.0.1:8000/api/redis-check/

# 3. List available models
GET http://127.0.0.1:8000/api/models/list/

# 4. Make a prediction
POST http://127.0.0.1:8000/api/predict/
Headers: Content-Type: application/json
Body:
{
    "ticker": "AAPL",
    "timeframes": ["1d"]
}
```

</details>

---

## 🎯 **Step 7: Quick Verification Checklist**

```bash
# ✅ Terminal 1 - Check Redis
redis-cli ping
# Expected: PONG

# ✅ Terminal 2 - Check Django API
curl http://127.0.0.1:8000/api/system/health/
# Expected: {"status": "healthy", ...}

# ✅ Terminal 3 - Check models exist
ls api/app/api/scripts/Models/*.pkl
# Expected: List of .pkl files

# ✅ Terminal 4 - Check Frontend
curl http://localhost:3000
# Expected: HTML content

# 🎯 Test prediction via curl
curl -X POST http://127.0.0.1:8000/api/predict/ \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}'
# Expected: {"ticker": "AAPL", "predictions": {...}}
```

---

<div align="center">

## 🛠️ **Troubleshooting Common Issues**

</div>

<details>
<summary><b>⚠️ Click to View Solutions</b></summary>

### **Issue 1: "Redis connection failed"**

```bash
# Solution: Make sure Redis is running
redis-server  # Start Redis
redis-cli ping  # Test connection
```

### **Issue 2: "No models available"**

```bash
# Solution: Train models first
cd api/app/api/scripts
python TrainModel.py
```

### **Issue 3: "Port already in use"**

```bash
# Kill existing processes
# Mac/Linux:
lsof -i :8000  # Find process using port 8000
kill -9 <PID>  # Kill the process

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### **Issue 4: "Module not found" errors**

```bash
# Make sure virtual environment is activated
# You should see (venv) in terminal

# Reinstall dependencies
pip install -r api/requirements.txt

# For frontend:
rm -rf node_modules package-lock.json
npm install
```

</details>

---

<div align="center">

## 📱 **Access Points Summary**

| Service                 | URL                                                                     | Terminal   |
| :---------------------- | :---------------------------------------------------------------------- | :--------- |
| 🔥 **Redis**            | `localhost:6379`                                                        | Terminal 1 |
| ⚙️ **Django Backend**   | `http://127.0.0.1:8000`                                                 | Terminal 2 |
| 📊 **API Docs**         | `http://127.0.0.1:8000/api/`                                            | Terminal 2 |
| 👤 **Django Admin**     | `http://127.0.0.1:8000/admin/`                                          | Terminal 2 |
| 🎨 **Frontend App**     | `http://localhost:3000`                                                 | Terminal 3 |
| 📮 **Postman / Newman** | `postman/Collections/StockVibePredictor_API_v2.postman_collection.json` | -          |

</div>

---

<div align="center">

## 📂 **Regenerating Ignored Files (.gitignore)**

</div>

<details>
<summary><b>📦 Click to View Ignored Files Guide</b></summary>

Our `.gitignore` file keeps the repo clean by excluding files like `node_modules`, `venv`, and `stock_model.pkl`. When you clone the repo, you'll need to recreate these files locally.

### **Ignored files and how to Recreate them:**

#### **`venv/`** - Python virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r api/requirements.txt
```

#### **`node_modules/`** - Node.js dependencies

```bash
cd ui
npm install
```

#### **`stock_model.pkl`** - Trained ML model

```bash
python TrainModel.py
```

#### **`__pycache__/`, `*.pyc`** - Compiled Python files

- Automatically generated when running Python scripts
- No manual action needed

#### **`build/`, `dist/`** - React build output

```bash
cd ui
npm run build
```

#### **`.env`** - Environment variables

```bash
echo "SECRET_KEY=your-django-secret-key" > api/.env
```

#### **`migrations/`** - Django migrations

```bash
cd api
python manage.py makemigrations
python manage.py migrate
```

#### **`*.sqlite3`, `*.db`** - Local database

```bash
cd api
python manage.py migrate
```

</details>

---

<div align="center">

## 🔄 **Daily Startup Sequence**

</div>

```bash
# 🖥️ Terminal 1
redis-server

# ⚙️ Terminal 2
cd StockVibePredictor/api
source .venv/bin/activate  # Mac/Linux — local baseline uses api/.venv (Task 08)
python manage.py runserver 127.0.0.1:8000

# 🧠 Terminal 3 (optional)
# Do not treat leftover TrainModel.py output as a verified training baseline.
# Current predictions use import-generated fallback .pkl files (R-ML-02).

# 🎨 Terminal 4
cd StockVibePredictor/ui
npm start
```

---

<div align="center">

## 🛑 **Shutdown Sequence**

</div>

```bash
# Terminal 4: Press Ctrl+C to stop Frontend
# Terminal 2: Press Ctrl+C to stop Django
# Terminal 1: Press Ctrl+C to stop Redis

# Or kill all at once (Mac/Linux):
pkill -f redis-server
pkill -f "python manage.py"
pkill -f node
```

---

<div align="center">

## ✅ **Success Indicators**

| Check              | Expected Result                           |
| :----------------- | :---------------------------------------- |
| ✅ Redis ping      | Returns `PONG`                            |
| ✅ Django server   | Shows `Starting development server`       |
| ✅ Frontend        | Shows `Compiled successfully!`            |
| ✅ Health endpoint | Returns `{"status": "healthy"}`           |
| ✅ Models list     | Shows at least one model                  |
| ✅ Prediction test | Returns data for AAPL                     |
| ✅ Browser         | Frontend loads at `http://localhost:3000` |

</div>

---

<div align="center">

## 👥 **Team Collaboration Tips**

| Tip                       | Description                                                                          |
| :------------------------ | :----------------------------------------------------------------------------------- |
| **🔍 Verify Setup**       | After cloning, run `git status` to ensure ignored files don't appear                 |
| **♻️ Regenerate Locally** | Each team member must recreate `venv`, `node_modules`, and `stock_model.pkl` locally |
| **📦 Large Files**        | Use Git LFS for large model files if needed                                          |
| **🔄 Consistency**        | Ensure all use same Python (3.8+) and Node.js (16+) versions                         |

</div>

---

<div align="center">

## 📦 **How to Contribute to Project**

</div>

<details>
<summary><b>🚀 Click to View Contribution Guide</b></summary>

### **Steps to Upload to `Group` Branch**

#### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/DibakarArchives/StockVibePredictor.git
cd StockVibePredictor
```

#### **2️⃣ Fetch Latest Branches**

```bash
git fetch origin
```

#### **3️⃣ Switch to `Group` Branch**

```bash
git checkout Group
# OR if doesn't exist locally:
git checkout -b Group origin/Group
```

#### **4️⃣ Add Your Files**

```bash
git add .
```

#### **5️⃣ Commit Changes**

```bash
git commit -m "Add my feature to Group branch"
```

#### **6️⃣ Push to Group Branch**

```bash
git push origin Group
```

### **💡 Notes**

- This **will not** affect the `main` branch
- Verify current branch: `git branch`
- Keep `Group` updated: `git pull origin Group`

</details>

---

<div align="center">

## 🧪 **ML Usage Guide**

</div>

1. **Access Application:** Navigate to http://localhost:3000
2. **Enter Stock Symbol:** Input a valid ticker (e.g., AAPL, TSLA, GOOGL)
3. **View Analysis:** The application displays:
   - 📊 Historical price charts
   - 📈 Technical indicators
   - 🔮 Next-day prediction (Up/Down)
   - 📊 Confidence metrics

---

<div align="center">

## 🎯 **API Endpoints**

| Method   | Endpoint                            | Description                | Notes                                                              |
| :------- | :---------------------------------- | :------------------------- | :----------------------------------------------------------------- |
| **POST** | `/api/predict/multi-timeframe/`     | Multi-timeframe prediction | Alias: `/api/predict/multi/`. Currently served by fallback models. |
| **POST** | `/api/predict/batch/`               | Batch predictions          |                                                                    |
| **GET**  | `/api/models/list/`                 | List models                | May include generated fallback `.pkl` files.                       |
| **GET**  | `/api/market/chart/<ticker>/`       | Chart data                 | Alias: `/api/chart/<ticker>/`. Live yfinance.                      |
| **GET**  | `/api/company/<ticker>/essentials/` | Company essentials         | Live yfinance.                                                     |
| **GET**  | `/api/system/health/`               | Health check               |                                                                    |
| **POST** | `/api/trading/real/`                | Real trade                 | **Stub / not live execution.**                                     |

</div>

> Source of truth for routes: [`api/app/StockPredict/urls.py`](api/app/StockPredict/urls.py). Newman collection: [`postman/Collections/StockVibePredictor_API_v2.postman_collection.json`](postman/Collections/StockVibePredictor_API_v2.postman_collection.json). `docs/API.md` is UNVERIFIED historical documentation. The duplicate `docs/API-Endpoints.json` was removed in Task 05.

---

<div align="center">

## ❌ **Common Issues and Fixes**

| Issue                    | Solution                                   |
| :----------------------- | :----------------------------------------- |
| **Missing stock models** | Run `python TrainModel.py`                 |
| **Dependency Errors**    | Re-run `pip install` or `npm install`      |
| **CORS Issues**          | Verify `django-cors-headers` is configured |
| **Git Conflicts**        | Pull latest (`git pull origin main`)       |

</div>

---

<div align="center">

## ✨ **Deployment (PLANNED / UNVERIFIED)**

There is **no** Dockerfile, docker-compose, GitHub Actions workflow, Heroku, or Vercel config in this repository. The commands below are leftover college-era notes, not a working deploy path.

</div>

<details>
<summary><b>🚀 Click to View Deployment Guide</b></summary>

### **Backend (Heroku):**

```bash
heroku create stock-vibe-predictor
git push heroku main
```

### **Frontend (Vercel):**

- Push `frontend/` to GitHub
- Connect to Vercel
- Update API URL in `App.js`

### **Team Task:**

Assign one member to handle deployment and test the live app

</details>

---

<div align="center">

## 🚀 **Running the App**

| Service            | URL                                |
| :----------------- | :--------------------------------- |
| 🔌 **Backend API** | http://localhost:8000/api/predict/ |
| 🌐 **Frontend UI** | http://localhost:3000              |

</div>

---

<div align="center">

## 🧪 **Usage**

</div>

1. Open the frontend in your browser
2. Enter a stock ticker (like `TSLA`, `GOOGL`, `AAPL`)
3. The app will:
   - 📊 Fetch real-time historical prices
   - 📈 Display a chart
   - 🤖 Predict if the stock will go **Up** or **Down** tomorrow

---

<div align="center">

## 🧭 **Roadmap**

</div>

- [ ] Add confidence scoring for predictions
- [ ] Implement portfolio tracking
- [ ] Advanced technical indicators (MACD, Bollinger Bands)
- [ ] User authentication and personalized dashboards
- [ ] Real-time WebSocket updates

---

<div align="center">

## 🔮 **Future Improvements**

| Feature                    | Description                              |
| :------------------------- | :--------------------------------------- |
| 📊 **Confidence Scores**   | Add ML prediction confidence metrics     |
| 📈 **Parallel Processing** | Support multiple stocks simultaneously   |
| 🧠 **Advanced Indicators** | Include MACD, Bollinger Bands, etc.      |
| ☁️ **Cloud Storage**       | Save and track predictions over time     |
| 🔐 **Authentication**      | Add user login & personalized dashboards |

</div>

---

<div align="center">

## 📖 **Support**

</div>

For issues and questions:

- 🐛 Create an issue on GitHub
- 📚 Check existing documentation
- 📝 Review API logs for error details

---

<div align="center">

## 🧾 **License**

</div>

<div align="center">

Metadata and badges claim **MIT**, but **no `LICENSE` file exists in the repository**.<br>
Treat the license as unverified until a file is added.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

<div align="center">

## 🎉 **Acknowledgments** 🎉

</div>

<div align="center">

| Thanks To                     | For                      |
| :---------------------------- | :----------------------- |
| 📊 **Yahoo Finance**          | Providing market data    |
| 🧠 **Scikit-learn Community** | Machine learning tools   |
| ⚙️ **Django Community**       | Excellent framework      |
| ⚛️ **React Community**        | Amazing frontend library |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=100&section=footer" alt="footer" />

## ⚠️ **DISCLAIMER** ⚠️

<p align="center">
This application is for educational and research purposes only.<br/>
Stock predictions are not guaranteed and should not be used as<br/>
the sole basis for investment decisions. Always consult with<br/>
financial professionals before making investment choices.
</p>

---

### **Made with 💖 Passion, deployed with 🎯 Precision**

### **and maintained with 💪 stubborn Optimism**

<br/>

<img src="https://img.shields.io/badge/Built_with-Love_&_Coffee-brown?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white" />

<br/>

**© 2025 StockVibePredictor | All Rights Reserved**

</div>

---
