# 📊 StockVibePredictor

**StockVibePredictor** is a full-stack machine learning web app that predicts stock price trends.
Enter a stock ticker (e.g., `AAPL`) to see historical price charts and get a prediction for whether the stock will go **Up** or **Down** the next trading day.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.0%2B-green.svg)](https://djangoproject.com)
[![React](https://img.shields.io/badge/React-18.0%2B-blue.svg)](https://reactjs.org)

# 🧩 Built with:

- 🧠 Machine Learning (Random Forest)
- ⚙️ Django (Backend + API)
- 🎨 React & Next (Frontend)

---

# 🌟 Features

- 🔄 **Real-Time Market Data** – Integration with Yahoo Finance (yfinance) API for accurate, up-to-date stock information.
- 🤖 **ML-Powered Predictions** – Trained on historical stock data with technical indicators (RSI, Moving Averages, etc.).
- 📈 **Interactive Visuals** – Uses Chart.js on the frontend to display trends and predictions.
- 🔌 **RESTful API** – Comprehensive backend API for data retrieval and predictions.
- 🍥 Responsive Design: Modern, mobile-first frontend interface

---

# 🛠️ Tech Stack

| Layer              | Technology                              |
| ------------------ | --------------------------------------- |
| **Backend**        | Django + Django REST Framework (Python) |
| **Frontend**       | React + Next.js + Chart.js              |
| **ML Model**       | scikit-learn (Random Forest Classifier) |
| **Data Source**    | yfinance API                            |
| **DatabaseSQLite** | (development) / PostgreSQL (production) |
| **Deployment**     | Heroku (backend) + Vercel (frontend)    |

---

# 📦 Project Architecture

```py
/StockVibePredictor/
│
│── /Backend/
│   │── /StockVibePredictor/
│   │   │── __init__.py
│   │   │── settings.py
│   │   │── urls.py
│   │   │── asgi.py
│   │   │── wsgi.py
│   │   │── middleware.py
│   │   │── schema.graphql
|   |
│   │── /Apps/
│   │   │── /StockPredict/
│   │   │   │── migrations/
│   │   │   │── __init__.py
│   │   │   │── models.py
│   │   │   │── views.py
│   │   │   │── serializers.py
│   │   │   │── urls.py
│   │   │   │── admin.py
│   │   │   │── forms.py
│   │   │   │── tests.py
│   │   │   │── permissions.py
│   │   │   │── tasks.py
│   │   │   │── signals.py
|   |   |
│   │   │── /Store/
│   │   │── /Blog/
|   |
│   │── /Logs/
│   │   │── /ModelTraining.log/
│   │   │── /StockPredict.log/
│   │   │── /TrainingErrors.log/
|   |
│   │── /Scripts/
│   │   │── /Models/
│   │   │── /Performance/
│   │   │── /TrainModel.py/
|   |
│   │── /Templates/
│   │   │── base.html
│   │   │── index.html
|   |
│   │── /Static/
|   |   |
│   │   │── /Css/
│   │   │── /Js/
│   │   │── /Images/
|   |
│   │── /Media/
|   |
│   │── /Config/
│   │   │── celery.py
│   │   │── logging.py
│   │   │── permissions.py
|   |
│   │── /Utils/
|   |
│   │── /Scripts/
│   │   │── backup_db.py
│   │   │── cron_jobs.py
|   |
│   │── manage.py
│   │── package-lock.json
│   │── package.json
│   │── requirements.txt
│   │── requirements-dev.txt
│   │── requirements-prod.txt
│   │── Dockerfile
│   │── docker-compose.yml
│   │── .env
│
│── /Frontend/
│   │── /Apps/
│   │   │── /Dashboard/
|   |   |   |-- package.lock.json
|   |   |   |-- package.json
|   |   |   |-- README.md
|   |
|   |-- /Public/
|   |   |-- favicon.ico
|   |   |-- index.html
|   |   |-- other essentials ...
|   |
|   |-- /Src/
|   |   |-- /Components/
|   |   |   |-- logo.svg
|   |   |   |-- Other essential components ...
|   |   |
|   |   |-- /Database/
|   |   |   |--StockDatabase.js
|   |   |
|   |   |-- App.css
|   |   |-- App.js
|   |   |-- index.css
|   |   |-- index.js
|   |
│   │── package.json
│   │── package-lock.json
│   │── webpack.config.js
│   │── vite.config.js
│   │── craco.config.js
│   │── webpack.config.js
│
│── /Tests/
│   │── /Unit/
│   │── /Integration/
│   │── /e2e/
│
│── /Docs/
│   │── API.md
│   │── README.md
│   │── CHANGELOG.md
│   │── architecture.md
│   │── Endpoints.txt
│
│── /Deployment/
│   │── nginx.conf
│   │── gunicorn.conf.py
│   │── supervisor.conf
│   │── aws_deploy.sh
│
│── /Security/
│   │── .htaccess
│   │── security.txt
│
│── /ci-cd/
│   │── .github/
│   │── .gitlab-ci.yml
│   │── jenkinsfile
│   │── docker-hub.yml
│
│── README.md
│── LICENSE
│── .pre-commit-config.yaml
│── .editorconfig
│── .flake8
│── .pylintrc
│── .babelrc
│── .eslintrc.json
│── .stylelintrc
│── .gitignore
│── .dockerignore
```

---

## 💻 Code Standards :

### - Python: Follow PEP 8 standards, use type hints where applicable.

### - JavaScript: ESLint configuration for consistent code style.

### - Git: Use conventional commit messages.

---

## 🚀 COMPLETE SETUP & RUNNING GUIDE

### 📋 Prerequisites Installation

### Install Required Software First :

```bash
# Install Python 3.8+
# Mac:
brew install python@3.11

# Windows: Download from python.org
# Linux:
sudo apt update
sudo apt install python3.11 python3-pip

# Install Node.js (for frontend)
# Mac:
brew install node

# Windows: Download from nodejs.org
# Linux:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Redis
# Mac:
brew install redis

# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Linux:
sudo apt update
sudo apt install redis-server
```

## 🔻 Step 1: Clone & Setup Project

### Clone the repository

```bash
git clone https://github.com/ThisIsDibakar/StockVibePredictor.git
cd StockVibePredictor
```

### Check you're in the right place

```bash
ls
```

You should see: Backend, Frontend, README.md, etc ...

## 🔴 Step 2: Start Redis Server (IMPORTANT!)

### Start Redis in a NEW Terminal :

```bash
# Mac/Linux - Terminal 1:
redis-server

# Windows - Terminal 1:
redis-server.exe

# You should see:
# - Ready to accept connections
# - The server is now ready to accept connections on port 6379
```

### Verify Redis is Running (Optional) :

#### In another terminal:

```bash
redis-cli ping

# Should return: PONG
```

#### ⚠️ KEEP THIS TERMINAL OPEN - Redis must stay running!

## ⚙️ Step 3: Backend Setup (Django)

### Open Terminal 2 for Backend :

```bash
# Navigate to project
cd StockVibePredictor

# Create virtual environment
python -m venv venv

# Activate environment
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### You should see (venv) in your terminal prompt

### Install Python Dependencies :

```bash
# Make sure you're in StockVibePredictor directory with (venv) active
pip install -r requirements.txt
```

### Setup Django Backend :

```bash
# Navigate to Backend
cd Backend

# Create .env file for settings
echo "DEBUG=True" > .env
echo "SECRET_KEY=your-secret-key-here" >> .env
echo "REDIS_URL=redis://localhost:6379/0" >> .env

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional, for admin panel)
python manage.py createsuperuser
# Enter username: <yourName>
# Enter email: <yourName>@example.com
# Enter password: (your choice)

# Collect static files
python manage.py collectstatic --noinput
```

### Start Django Server :

#### Make sure you're in Backend directory ...

```bash
python manage.py runserver
#### You should see :
#### Starting development server at http://127.0.0.1:8000/
#### Quit the server with CONTROL-C.
```

✅ Backend is running at: http://127.0.0.1:8000 <br />
⚠️ KEEP THIS TERMINAL OPEN!

## 🧠 Step 4: Train ML Models

### Open Terminal 3 for Training :

```bash
# Navigate to Scripts directory
cd StockVibePredictor/Backend/Scripts

# Activate virtual environment (if not already active)
# Mac/Linux:
source ../../venv/bin/activate

# Windows:
..\..\venv\Scripts\activate

# Train models - Choose one:

# Option 1: Quick training (just universal models)
python TrainModel.py

# Option 2: Train specific category
python TrainModel.py category mega_cap_tech 1d,1w

# Option 3: Full training (takes longer)
python TrainModel.py full

# You should see progress logs:
# INFO: Training universal model for 1d...
# INFO: Model saved: Models/universal_model_1d.pkl
```

### Verify Models are Created :

```bash
# Check Models directory

ls Models/

# Should show: universal_model_1d.pkl, universal_model_1w.pkl, etc.
```

## 💻 Step 5: Frontend Setup (React/Next.js)

### Open Terminal 4 for Frontend :

```bash
# Navigate to Frontend directory
cd StockVibePredictor/Frontend

# Install dependencies
npm install

# If you get errors, try:
npm install --legacy-peer-deps

# Create .env.local file for API endpoint
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api" > .env.local

# Start the development server
npm run dev
# or
npm start

# You should see:
# Local: http://localhost:3000
# ready - started server on 0.0.0.0:3000
```

✅ Frontend is running at: http://localhost:3000 <br />
⚠️ KEEP THIS TERMINAL OPEN!

## 🧪 Step 6: Test with Postman

### Install & Setup Postman :

```bash
# Download Postman from: https://www.postman.com/downloads/
# Or use web version: https://web.postman.co/
```

### Import Endpoints to Postman :

1. Open Postman
2. Click "Import" → "Raw text"
3. Paste all endpoints from the Endpoints.txt file
4. Click "Continue" → "Import"

### Test Basic Endpoints:

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

## 🎯 Step 7: Quick Verification Checklist

### Run these commands to verify everything works :

```bash
# Terminal 1 - Check Redis
redis-cli ping
# Expected: PONG

# Terminal 2 - Check Django API
curl http://127.0.0.1:8000/api/system/health/
# Expected: {"status": "healthy", ...}

# Terminal 3 - Check models exist
ls Backend/Scripts/Models/*.pkl
# Expected: List of .pkl files

# Terminal 4 - Check Frontend
curl http://localhost:3000
# Expected: HTML content

# Test prediction via curl
curl -X POST http://127.0.0.1:8000/api/predict/ \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}'
# Expected: {"ticker": "AAPL", "predictions": {...}}
```

## 🛠️ Troubleshooting Common Issues :

### Issue 1: "Redis connection failed"

```bash
# Solution: Make sure Redis is running
redis-server  # Start Redis
redis-cli ping  # Test connection
```

### Issue 2: "No models available"

```bash
# Solution: Train models first
cd Backend/Scripts
python TrainModel.py
# Then reload models in Django
curl -X POST http://127.0.0.1:8000/api/models/reload/
```

### Issue 3: "Port already in use"

```bash
# Kill existing processes
# Mac/Linux:
lsof -i :8000  # Find process using port 8000
kill -9 <PID>  # Kill the process

lsof -i :3000  # Find process using port 3000
kill -9 <PID>  # Kill the process

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Issue 4: "Module not found" errors

```bash
# Make sure virtual environment is activated
# You should see (venv) in terminal

# Reinstall dependencies
pip install -r requirements.txt

# For frontend:
rm -rf node_modules package-lock.json
npm install
```

## 📱 Access Points Summary

After everything is running, you can access : <br />
• Redis: ⁠localhost:6379 (running in Terminal 1) <br />
• Django Backend API: ⁠http://127.0.0.1:8000 (Terminal 2) <br />
• API Documentation: ⁠http://127.0.0.1:8000/api/ <br />
• Django Admin: ⁠http://127.0.0.1:8000/admin/ <br />
• Frontend Application: ⁠http://localhost:3000 (Terminal 4) <br />
• Postman: Test all endpoints from Endpoints.txt <br />

---

## 📂 Regenerating Ignored Files (.gitignore) :

Our .gitignore file keeps the repo clean by excluding files like node_modules, venv, and stock_model.pkl. When you clone the repo, you’ll need to recreate these files locally.

## Ignored files and how to Recreate them :

venv/ : Python virtual environment for Django and ML dependencies.

## Recreate :

```bash
python -m venv venv
source venv/bin/activate                # Mac/Linux
venv\Scripts\activate                   # Windows
pip install -r requirements.txt
```

This sets up the environment and installs all Python dependencies (e.g., django, scikit-learn, yfinance).

node_modules/, frontend/node_modules/: Node.js dependencies for the React front-end.

```bash
cd Frontend
npm install
```

This regenerates node_modules based on package.json.

stock_model.pkl: The trained ML model for stock predictions.

Recreate:

```bash
python TrainModel.py
```

Runs the training script to generate the Random Forest model and moves it to the Django folder.

**pycache**/, _.pyc, _.pyo, \*.pyd : Compiled Python files ...

Recreate: Automatically generated when you run Python scripts (e.g., python manage.py runserver).
No manual action needed.

frontend/build/, frontend/dist/ : React build output for deployment.

```bash
cd Frontend
npm run build
```

Generates the production-ready front-end files.

.env, _.env._ : Environment files for sensitive settings (e.g., API keys) ...

#### If needed, create a .env file in StockVibePredictor/ with your settings (e.g., SECRET_KEY for Django) :

```bash
echo "SECRET_KEY=your-django-secret-key" > StockVibePredictor/.env
```

Generate a Django secret key using a tool like " djecrety.ir " if required.

migrations/ : Django migration files ...

```bash
cd StockVibePredictor
python manage.py makemigrations
python manage.py migrate
```

This generates and applies migrations for your Django app.

_.sqlite3, _.db : Local SQLite database.

```bash
cd StockVibePredictor
python manage.py migrate
```

Creates a fresh SQLite database if needed (not used in this project unless you add models).

Other Ignored Files : Files like .DS_Store, .vscode/, .coverage, etc., are user-specific or temporary and don’t need recreation.

---

## Why These Files Are Ignored ??

node_modules/: Huge folder, regenerated with npm install.
venv/: User-specific, avoids conflicts across machines.
stock_model.pkl: Large file, easily recreated with train_model.py.
migrations/: Environment-specific, prevents merge conflicts.
**pycache**/, \*.pyc: Temporary compiled files.
.env: May contain sensitive keys.
Others: Editor files (.vscode/), OS files (.DS_Store), or test outputs (.coverage) are irrelevant to the repo.

---

## 🔄 Daily Startup Sequence

### Once everything is installed, daily startup is :

```bash
# Terminal 1
redis-server

# Terminal 2
cd StockVibePredictor/Backend
source ../venv/bin/activate  # Mac/Linux
# or
..\venv\Scripts\activate  # Windows
python manage.py runserver

# Terminal 3 (optional - for model updates)
cd StockVibePredictor/Backend/Scripts
python TrainModel.py daily

# Terminal 4
cd StockVibePredictor/Frontend
npm run dev
```

---

## 🛑 Shutdown Sequence

#### To properly shutdown :

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

## ✅ Success Indicators

### You know everything is working when :

✅ Redis responds with "PONG" to ping <br />
✅ Django shows no errors and says "Starting development server" <br />
✅ Frontend compiles successfully with "Compiled successfully!" <br />
✅ System health endpoint returns {"status": "healthy"} <br />
✅ Models list shows at least one model <br />
✅ Prediction endpoint returns data for AAPL <br />
✅ Frontend loads at http://localhost:3000 <br />

---

## Team Tips :

Verify Setup: After cloning, run git status to ensure ignored files don’t appear.
Regenerate Locally: Each team member must recreate venv, node_modules, and stock_model.pkl locally.
Large Files: If you need to include stock_model.pkl in the repo (e.g., for deployment), remove it from .gitignore and use Git LFS :

```bash
git lfs install
git lfs track "\*.pkl"
git add .gitattributes
git add stock_model.pkl
git commit -m "Track ML model with Git LFS"
git push origin main
```

### Consistency: Ensure all team members use the same Python (3.8+) and Node.js (16+) versions to avoid dependency issues.

---

## 🚨 Pro Tip :

Run :

```bash
pip freeze > requirements.txt
```

After the installation of dependencies to keep requirements.txt updated for the team.

---

## ML Usage :

Access the Application: Navigate to http://localhost:3000 <br />
Enter Stock Symbol: Input a valid ticker symbol (e.g., AAPL, TSLA, GOOGL) <br />
View Analysis: The application will display :

- Historical price charts
- Technical indicators
- Next-day prediction (Up/Down)
- Confidence metrics

---

## 🎯 API Endpoints :

| Method   | Endpoint                 | Description          |
| -------- | ------------------------ | -------------------- |
| **POST** | **/api/predict/**        | Get stock prediction |
| **GET**  | **/api/stock/{ticker}/** | Retrieve stock data  |
| **GET**  | **/api/health/**         | Health check         |

---

## ❌ Common Issues and Fixes :

- Missing stock models: Run python TrainModel.py. <br />
- Dependency Errors: Ensure requirements.txt and package.json are up-to-date. Re-run pip install or npm install. <br />
- CORS Issues: Verify django-cors-headers is installed and configured in StockVibePredictor/settings.py. <br />
- Git Conflicts: Pull latest changes (git pull origin main) and resolve conflicts in VS Code or git mergetool. <br />

---

## ✨ Deployment (OPTIONAL) :

### Backend (Heroku):

```bash
heroku create stock-vibe-predictor
git push heroku main
```

Ensure stock_model.pkl is in **StockVibePredictor/Backend/Scripts** or regenerated during deployment. <br />
Front-end (Vercel):Push frontend/ to a GitHub repo, connect to Vercel, and update App.js with the Heroku API URL. <br />
Team Task: Assign one member to handle deployment and test the live app. <br />

---

## 🚀 Running the App :

- 🔌 **Backend API** : [http://localhost:8000/api/predict/](http://localhost:8000/api/predict/)
- 🌐 **Frontend UI** : [http://localhost:3000](http://localhost:3000)

---

## 🧪 Usage :

1. Open the frontend in your browser.
2. Enter a stock ticker (like `TSLA`, `GOOGL`, `AAPL`).
3. The app will:
   - 📊 Fetch real-time historical prices
   - 📈 Display a chart
   - 🤖 Predict if the stock will go **Up** or **Down** tomorrow

---

## 🧭 Roadmap

- Add confidence scoring for predictions.
- Implement portfolio tracking.
- Advanced technical indicators (MACD, Bollinger Bands).
- User authentication and personalized dashboards.
- Real-time WebSocket updates.

  ***

## 🔮 Future Improvements :

- 📊 Add confidence scores to ML predictions ...
- 📈 Support multiple stocks in parallel ...
- 🧠 Include advanced indicators like MACD, Bollinger Bands, etc.
- ☁️ Save and track predictions over time ...
- 🔐 Add user login & personalized dashboards ...

---

## 📖 Support

### For issues and questions :

- Create an issue on GitHub.
- Check existing documentation.
- Review API logs for error details.

---

## 🧾 License :

This project is licensed under the **MIT License**.
Feel free to fork, remix, and use — just don’t forget to credit. 😎

---

## 🎉 Acknowledgments :

- Yahoo Finance for providing market data.
- scikit-learn community for machine learning tools.
- Django and React communities for excellent frameworks.

---

## ⚠️ Disclaimer:

This application is for educational and research purposes only. Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

---

### ☕ Made with Passion, deployed with Precision, and maintained with stubborn Optimism ☕

---
