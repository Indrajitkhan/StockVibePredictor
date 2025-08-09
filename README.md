# 📊 StockVibePredictor

**StockVibePredictor** is a full-stack machine learning web app that predicts stock price trends.
Enter a stock ticker (e.g., `AAPL`) to see historical price charts and get a prediction for whether the stock will go **Up** or **Down** the next trading day.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.0%2B-green.svg)](https://djangoproject.com)
[![React](https://img.shields.io/badge/React-18.0%2B-blue.svg)](https://reactjs.org)

## 🧩 Built with:

- 🧠 Machine Learning (Random Forest)
- ⚙️ Django (Backend + API)
- 🎨 React & Next (Frontend)

---

## 🌟 Features

- 🔄 **Real-Time Market Data** – Integration with Yahoo Finance (yfinance) API for accurate, up-to-date stock information.
- 🤖 **ML-Powered Predictions** – Trained on historical stock data with technical indicators (RSI, Moving Averages, etc.).
- 📈 **Interactive Visuals** – Uses Chart.js on the frontend to display trends and predictions.
- 🔌 **RESTful API** – Comprehensive backend API for data retrieval and predictions.
- 🍥 Responsive Design: Modern, mobile-first frontend interface

---

## 🛠️ Tech Stack

| Layer              | Technology                              |
| ------------------ | --------------------------------------- |
| **Backend**        | Django + Django REST Framework (Python) |
| **Frontend**       | React + Next.js + Chart.js              |
| **ML Model**       | scikit-learn (Random Forest Classifier) |
| **Data Source**    | yfinance API                            |
| **DatabaseSQLite** | (development) / PostgreSQL (production) |
| **Deployment**     | Heroku (backend) + Vercel (frontend)    |

---

## 📦 Project Architecture

```py
/StockVibePredictor/                              # INFO: Root directory
│
│── /Backend/                                     # INFO: Backend (Django)
│   │── /StockVibePredictor/                      # INFO: Main Django project folder
│   │   │── __init__.py
│   │   │── settings.py                           # INFO: Django settings
│   │   │── urls.py                               # INFO: Main URL config
│   │   │── asgi.py
│   │   │── wsgi.py
│   │   │── middleware.py                         # INFO: Custom middleware (optional)
│   │   │── schema.graphql                        # INFO: GraphQL Schema (if using GraphQL)
|   |
│   │── /Apps/                                    # INFO: Custom Django apps (Modular)
│   │   │── /StockPredict/                        # INFO: Stock Prediction backend app
│   │   │   │── migrations/                       # INFO: Migrations for the app
│   │   │   │── __init__.py
│   │   │   │── models.py                         # INFO: Stock models
│   │   │   │── views.py                          # INFO: Stock views (API)
│   │   │   │── serializers.py                    # INFO: DRF Serializers
│   │   │   │── urls.py                           # INFO: App-specific URLs
│   │   │   │── admin.py                          # INFO: Django admin
│   │   │   │── forms.py                          # INFO: Django forms
│   │   │   │── tests.py                          # INFO: Unit tests
│   │   │   │── permissions.py                    # INFO: Custom permissions (DRF)
│   │   │   │── tasks.py                          # INFO: Celery tasks (if using)
│   │   │   │── signals.py                        # INFO: Django signals
|   |   |
│   │   │── /Store/                               # INFO: Example apps
│   │   │── /Blog/                                # INFO: Other Apps
|   |
│   │── /Logs/
│   │   │── /stockpredict.log/                    # INFO: Logs messages of Backend
|   |
│   │── /Scripts/
│   │   │── /TrainModel.py/                      # INFO: Python Model for ML training
│   │   │── /stock_model.pkl/                    # INFO: Actual Model
|   |
│   │── /Templates/                               # INFO: Global HTML templates (Jinja)
│   │   │── base.html                             # INFO: Base template
│   │   │── index.html                            # INFO: Homepage
|   |
│   │── /Static/                                  # INFO: Global static files (CSS, JS)
|   |   |
│   │   │── /Css/
│   │   │── /Js/
│   │   │── /Images/
|   |
│   │── /Media/                                   # INFO: Uploaded media files
|   |
│   │── /Config/                                  # INFO: Additional settings (optional)
│   │   │── celery.py                             # INFO: Celery config (if using)
│   │   │── logging.py                            # INFO: Logging settings
│   │   │── permissions.py                        # INFO: Global API permissions (if using DRF)
|   |
│   │── /Utils/                                   # INFO: Utility functions
|   |
│   │── /Scripts/                                 # INFO: Management scripts (e.g., backup, cronjobs)
│   │   │── backup_db.py                          # INFO: Script to backup database
│   │   │── cron_jobs.py                          # INFO: Automate scheduled tasks
|   |
│   │── manage.py                                 # INFO: Django CLI tool
│   │── package-lock.json                         # INFO: Dependency lock file
│   │── package.json                              # INFO: Backend dependencies
│   │── requirements.txt                          # INFO: Python dependencies
│   │── requirements-dev.txt                      # INFO: Dev-only dependencies
│   │── requirements-prod.txt                     # INFO: Production-only dependencies
│   │── Dockerfile                                # INFO: Docker config (optional)
│   │── docker-compose.yml                        # INFO: Docker Compose (optional)
│   │── .env                                      # INFO: Environment variables
│
│── /Frontend/                                    # INFO: Frontend (React, Vue, etc.)
|   |
│   │── /Apps/                                    # INFO: Apps Folder
|   |   |
│   │   │── /Dashboard/                           # INFO: Dashboard App
|   |   |   |
|   |   |   |-- package.lock.json
|   |   |   |-- package.json
|   |   |   |-- README.md
|   |   |
|   |   |
|   |-- /Components/
|   |   |
|   |   |-- LoadingSpinner.css
|   |   |-- LoadingSpinner.js
|   |   |-- logo.svg
|   |   |-- PredictionResult.css
|   |   |-- PredictionResult.js
|   |   |-- reportWebVitals.js
|   |   |-- setupTests.js
|   |   |-- StockChart.css
|   |   |-- StockChart.js
|   |   |-- StockInput.css
|   |   |-- StockInput.js
|   |
|   |-- /Public/
|   |   |
|   |   |-- favicon.ico
|   |   |-- index.html
|   |   |-- logo192.png
|   |   |-- logo152.png
|   |   |-- manifest.json
|   |   |-- robots.txt
|   |
|   |-- /Src/
|   |   |
|   |   |-- App.css
|   |   |-- App.js
|   |   |-- index.css
|   |   |-- index.js
|   |
│   │── package.json                              # INFO: Frontend dependencies
│   │── package-lock.json                         # INFO: Dependency lock file
│   │── webpack.config.js                         # INFO: Webpack config (if using)
│   │── vite.config.js                            # INFO: Vite config (if using)
│
│── /Tests/                                       # INFO: Global test directory
│   │── /Unit/                                    # INFO: Unit tests
│   │── /Integration/                             # INFO: Integration tests
│   │── /e2e/                                     # INFO: End-to-end tests
│
│── /Docs/                                        # INFO: Documentation
│   │── API.md                                    # INFO: API Docs
│   │── README.md                                 # INFO: Project documentation
│   │── CHANGELOG.md                              # INFO: Changelog (if needed)
│   │── architecture.md                           # INFO: Architecture documentation
│
│── /Deployment/                                  # INFO: Deployment configs
│   │── nginx.conf                                # INFO: Nginx reverse proxy settings
│   │── gunicorn.conf.py                          # INFO: Gunicorn settings
│   │── supervisor.conf                           # INFO: Process manager config
│   │── aws_deploy.sh                             # INFO: AWS Deployment script
│
│── /Security/                                    # INFO: Security-related files
│   │── .htaccess                                 # INFO: Apache security config (if needed)
│   │── security.txt                              # INFO: Security policies
│
│── /ci-cd/                                       # INFO: CI/CD Pipeline setup
│   │── .github/                                  # INFO: GitHub Actions workflows
│   │── .gitlab-ci.yml                            # INFO: GitLab CI/CD config (if using GitLab)
│   │── jenkinsfile                               # INFO: Jenkins config (if using Jenkins)
│   │── docker-hub.yml                            # INFO: Docker Hub auto-builds
│
│── README.md                                     # INFO: Project documentation
│── LICENSE                                       # INFO: License file (if needed)
│── .pre-commit-config.yaml                       # INFO: Pre-commit hooks config
│── .editorconfig                                 # INFO: Code formatting rules
│── .flake8                                       # INFO: Python linting config
│── .pylintrc                                     # INFO: Pylint config
│── .babelrc                                      # INFO: Babel config (if using Babel)
│── .eslintrc.json                                # INFO: ESLint config (for frontend)
│── .stylelintrc                                  # INFO: Stylelint config (for frontend)
│── .gitignore                                    # INFO: Git ignore file
│── .dockerignore                                 # INFO: Docker ignore file
```

---

## 📜 Prerequisites :

- Python 3.8 or higher
- Node.js 16.0 or higher
- NPM or yarn package manager
- Git

---

## 💻 Code Standards :

- Python: Follow PEP 8 standards, use type hints where applicable.
- JavaScript: ESLint configuration for consistent code style.
- Git: Use conventional commit messages.

---

## 🧰 Setup Instructions

##### 🔻 Clone the Repository :

```bash
git clone https://github.com/your-username/StockVibePredictor.git
cd StockVibePredictor
```

### ⚙️ Backend Setup (Django)

#### Create virtual environment :

```bash
python -m venv venv
```

#### Activate environment :

#### On Mac/Linux :

```bash
source venv/bin/activate
```

#### On Windows :

```bash
venv\Scripts\activate
```

#### Install Python Dependencies :

```bash
pip install -r requirements.txt
```

#### Navigate to Django project :

```bash
cd StockVibePredictor
cd Backend
```

#### Configure Django :

```bash
cd Backend
python manage.py migrate
python manage.py collectstatic --noinput
```

#### Start Development Server :

```bash
python manage.py runserver
```

#### The API will be available at http://localhost:8000

### 💻 Frontend Setup (React & Next JS) :

#### Go to frontend folder :

```bash
cd Frontend
cd Src
cd dashboard
```

#### Install node dependencies :

```bash
npm install
```

#### Run the React app :

```bash
npm start
```

#### The application will be available at http://localhost:3000

---

## 🧠 Train the ML Model

#### From Scripts directory :

```bash
cd Backend
cd Scripts
python TrainModel.py
```

#### This generates stock_model.pkl which is automatically loaded by the Django application.

#### 📦 The backend will load this file to make predictions.

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

## 📂 Regenerating Ignored Files (.gitignore) :

Our .gitignore file keeps the repo clean by excluding files like node_modules, venv, and stock_model.pkl. When you clone the repo, you’ll need to recreate these files locally.

### Ignored files and how to Recreate them :

venv/ : Python virtual environment for Django and ML dependencies.

### Recreate :

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

### Why These Files Are Ignored ??

node_modules/: Huge folder, regenerated with npm install.
venv/: User-specific, avoids conflicts across machines.
stock_model.pkl: Large file, easily recreated with train_model.py.
migrations/: Environment-specific, prevents merge conflicts.
**pycache**/, \*.pyc: Temporary compiled files.
.env: May contain sensitive keys.
Others: Editor files (.vscode/), OS files (.DS_Store), or test outputs (.coverage) are irrelevant to the repo.

---

### Team Tips :

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

#### Consistency: Ensure all team members use the same Python (3.8+) and Node.js (16+) versions to avoid dependency issues.

---

## 🚨 Pro Tip :

Run :

```bash
pip freeze > requirements.txt
```

After the installation of dependencies to keep requirements.txt updated for the team.

---

## 🎯 API Endpoints :

| Method   | Endpoint                 | Description          |
| -------- | ------------------------ | -------------------- |
| **POST** | **/api/predict/**        | Get stock prediction |
| **GET**  | **/api/stock/{ticker}/** | Retrieve stock data  |
| **GET**  | **/api/health/**         | Health check         |

---

## 🧪 Testing the Setup :

### Backend : Test the API with Postman or curl :

#### Example Request :

```bash
curl -X POST -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

#### Example Response :

```json
{
  "ticker": "AAPL",
  "prediction": "Up",
  "confidence": 0.78,
  "current_price": 150.25,
  "technical_indicators": {
    "rsi": 65.2,
    "ma_50": 148.5,
    "ma_200": 145.8
  }
}
```

### Frontend :

Enter a ticker (e.g., TSLA), and check for a chart and prediction. <br />
ML Model: Verify stock_model.pkl works by running the API and checking predictions.

---

## ❌ Common Issues and Fixes :

- Missing stock_model.pkl: Run python train_model.py and move the file to StockVibePredictor/. <br />
- Dependency Errors: Ensure requirements.txt and package.json are up-to-date. Re-run pip install or npm install. <br />
- CORS Issues: Verify django-cors-headers is installed and configured in StockVibePredictor/settings.py. <br />
- Git Conflicts: Pull latest changes (git pull origin main) and resolve conflicts in VS Code or git mergetool. <br />

---

## ✨ Deployment (OPTIONAL) :

#### Backend (Heroku):

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
