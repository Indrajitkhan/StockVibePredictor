# 📊 StockVibePredictor

**StockVibePredictor** is a full-stack machine learning web app that predicts stock price trends.
Enter a stock ticker (e.g., `AAPL`) to see historical price charts and get a prediction for whether the stock will go **Up** or **Down** the next trading day.

Built with:

- 🧠 Machine Learning (Random Forest)
- ⚙️ Django (Backend + API)
- 🎨 React & Next (Frontend)

---

## 🌟 Features

- 🔄 **Real-Time Stock Data** – Fetches accurate market data using `yfinance`.
- 🤖 **ML-Powered Predictions** – Trained on historical stock data with technical indicators (RSI, Moving Averages, etc.).
- 📈 **Interactive Visuals** – Uses Chart.js on the frontend to display trends and predictions.
- 🔌 **REST API** – Exposes endpoints to fetch predictions and chart data.

---

## 🛠️ Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| **Backend**     | Django + Django REST Framework (Python) |
| **Frontend**    | React + Chart.js                        |
| **ML Model**    | scikit-learn (Random Forest Classifier) |
| **Data Source** | yfinance API                            |
| **Deployment**  | Heroku (backend) + Vercel (frontend)    |

---

## 📦 Project Structure

```
/StockVibePredictor/              # INFO: Root directory
│
│── /Backend/                    # INFO: Backend (Django)
│   │── /StockVibePredictor/             # INFO: Main Django project folder
│   │   │── __init__.py
│   │   │── settings.py          # INFO: Django settings
│   │   │── urls.py              # INFO: Main URL config
│   │   │── asgi.py
│   │   │── wsgi.py
│   │   │── middleware.py        # INFO: Custom middleware (optional)
│   │   │── schema.graphql       # INFO: GraphQL Schema (if using GraphQL)
|   |
│   │── /Apps/                   # INFO: Custom Django apps (Modular)
│   │   │── /Users/              # INFO: User management app
│   │   │   │── migrations/      # INFO: Migrations for the app
│   │   │   │── __init__.py
│   │   │   │── models.py        # INFO: User models
│   │   │   │── views.py         # INFO: User views (API)
│   │   │   │── serializers.py   # INFO: DRF Serializers
│   │   │   │── urls.py          # INFO: App-specific URLs
│   │   │   │── admin.py         # INFO: Django admin
│   │   │   │── forms.py         # INFO: Django forms
│   │   │   │── tests.py         # INFO: Unit tests
│   │   │   │── permissions.py   # INFO: Custom permissions (DRF)
│   │   │   │── tasks.py         # INFO: Celery tasks (if using)
│   │   │   │── signals.py       # INFO: Django signals
|   |   |
│   │   │── /Store/              # INFO: Example app (e.g., eCommerce)
│   │   │── /Blog/               # INFO: Blog module
|   |
│   │── /Templates/              # INFO: Global HTML templates (Jinja)
│   │   │── base.html            # INFO: Base template
│   │   │── index.html           # INFO: Homepage
|   |
│   │── /Static/                 # INFO: Global static files (CSS, JS)
|   |   |
│   │   │── /Css/
│   │   │── /Js/
│   │   │── /Images/
|   |
│   │── /Media/                  # INFO: Uploaded media files
|   |
│   │── /Config/                 # INFO: Additional settings (optional)
│   │   │── celery.py            # INFO: Celery config (if using)
│   │   │── logging.py           # INFO: Logging settings
│   │   │── permissions.py       # INFO: Global API permissions (if using DRF)
|   |
│   │── /Utils/                  # INFO: Utility functions
|   |
│   │── /Scripts/                # INFO: Management scripts (e.g., backup, cronjobs)
│   │   │── backup_db.py         # INFO: Script to backup database
│   │   │── cron_jobs.py         # INFO: Automate scheduled tasks
|   |
│   │── manage.py                 # INFO: Django CLI tool
│   │── requirements.txt          # INFO: Python dependencies
│   │── requirements-dev.txt      # INFO: Dev-only dependencies
│   │── requirements-prod.txt     # INFO: Production-only dependencies
│   │── Dockerfile                # INFO: Docker config (optional)
│   │── docker-compose.yml        # INFO: Docker Compose (optional)
│   │── .env                      # INFO: Environment variables
│   │── .gitignore                # INFO: Git ignore file
│
│── /Frontend/                    # INFO: Frontend (React, Vue, etc.)
│   │── /Src/                     # INFO: Source code
│   │   │── /Components/          # INFO: Reusable UI components
│   │   │── /Pages/               # INFO: Page components
│   │   │── /Services/            # INFO: API service handlers
│   │   │── /Redux/               # INFO: Redux store (if using Redux)
│   │   │── app.js                # INFO: Main app component
│   │   │── index.js              # INFO: Entry point
│   │   │── hooks.js              # INFO: Custom React hooks
|   |
│   │── /Public/                  # INFO: Public assets
│   │── package.json              # INFO: Frontend dependencies
│   │── package-lock.json         # INFO: Dependency lock file
│   │── webpack.config.js         # INFO: Webpack config (if using)
│   │── vite.config.js            # INFO: Vite config (if using)
│
│── /Tests/                       # INFO: Global test directory
│   │── /Unit/                    # INFO: Unit tests
│   │── /Integration/             # INFO: Integration tests
│   │── /e2e/                     # INFO: End-to-end tests
│
│── /Docs/                        # INFO: Documentation
│   │── API.md                    # INFO: API Docs
│   │── README.md                  # INFO: Project documentation
│   │── CHANGELOG.md               # INFO: Changelog (if needed)
│   │── architecture.md            # INFO: Architecture documentation
│
│── /Deployment/                   # INFO: Deployment configs
│   │── nginx.conf                 # INFO: Nginx reverse proxy settings
│   │── gunicorn.conf.py           # INFO: Gunicorn settings
│   │── supervisor.conf            # INFO: Process manager config
│   │── aws_deploy.sh              # INFO: AWS Deployment script
│
│── /Security/                     # INFO: Security-related files
│   │── .htaccess                  # INFO: Apache security config (if needed)
│   │── security.txt               # INFO: Security policies
│
│── /ci-cd/                        # INFO: CI/CD Pipeline setup
│   │── .github/                   # INFO: GitHub Actions workflows
│   │── .gitlab-ci.yml             # INFO: GitLab CI/CD config (if using GitLab)
│   │── jenkinsfile                # INFO: Jenkins config (if using Jenkins)
│   │── docker-hub.yml             # INFO: Docker Hub auto-builds
│
│── README.md                      # INFO: Project documentation
│── LICENSE                         # INFO: License file (if needed)
│── .pre-commit-config.yaml         # INFO: Pre-commit hooks config
│── .editorconfig                   # INFO: Code formatting rules
│── .flake8                         # INFO: Python linting config
│── .pylintrc                       # INFO: Pylint config
│── .babelrc                        # INFO: Babel config (if using Babel)
│── .eslintrc.json                  # INFO: ESLint config (for frontend)
│── .stylelintrc                    # INFO: Stylelint config (for frontend)
│── .gitignore                      # INFO: Git ignore file
│── .dockerignore                   # INFO: Docker ignore file
```

---

## 🧰 Setup Instructions

##### 🔻 Clone the Repo :

```
git clone https://github.com/your-username/StockVibePredictor.git
cd StockVibePredictor
```

### ⚙️ Backend Setup (Django)

#### Create virtual environment :

```
python -m venv venv
```

#### Activate environment :

#### On Mac/Linux :

```
source venv/bin/activate
```

#### On Windows :

```
venv\Scripts\activate
```

#### Install dependencies :

```
pip install -r requirements.txt
```

#### Navigate to Django project :

```
cd StockVibePredictor
```

#### Run development server :

```
python manage.py runserver
```

### 💻 Frontend Setup (React & Next JS) :

#### Go to frontend folder :

```
cd Frontend
```

#### Install node dependencies :

npm install

#### Run the React app :

```
npm start
```

🧠 Train the ML Model

#### From root directory :

```
python train_model.py
```

This creates a file called stock_model.pkl.
Then move it to the Django app folder:
mv stock_model.pkl StockVibePredictor/

📦 The backend will load this file to make predictions.

---

## 📂 Regenerating Ignored Files (.gitignore) :

Our .gitignore file keeps the repo clean by excluding files like node_modules, venv, and stock_model.pkl. When you clone the repo, you’ll need to recreate these files locally.

### Ignored files and how to Recreate them :

venv/ : Python virtual environment for Django and ML dependencies.

### Recreate :

```
python -m venv venv
source venv/bin/activate                # Mac/Linux
venv\Scripts\activate                   # Windows
pip install -r requirements.txt
```

This sets up the environment and installs all Python dependencies (e.g., django, scikit-learn, yfinance).

node_modules/, frontend/node_modules/: Node.js dependencies for the React front-end.

```
cd Frontend
npm install
```

This regenerates node_modules based on package.json.

stock_model.pkl: The trained ML model for stock predictions.

Recreate:

```
python train_model.py
mv stock_model.pkl StockVibePredictor/
```

Runs the training script to generate the Random Forest model and moves it to the Django folder.

**pycache**/, _.pyc, _.pyo, \*.pyd : Compiled Python files ...

Recreate: Automatically generated when you run Python scripts (e.g., python manage.py runserver).
No manual action needed.

frontend/build/, frontend/dist/ : React build output for deployment.

```
cd Frontend
npm run build
```

Generates the production-ready front-end files.

.env, _.env._ : Environment files for sensitive settings (e.g., API keys) ...

#### If needed, create a .env file in StockVibePredictor/ with your settings (e.g., SECRET_KEY for Django) :

```
echo "SECRET_KEY=your-django-secret-key" > StockVibePredictor/.env
```

Generate a Django secret key using a tool like " djecrety.ir " if required.

migrations/ : Django migration files ...

```
cd StockVibePredictor
python manage.py makemigrations
python manage.py migrate
```

This generates and applies migrations for your Django app.

_.sqlite3, _.db : Local SQLite database.

```
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

```
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

```
pip freeze > requirements.txt
```

After the installation of dependencies to keep requirements.txt updated for the team.

---

## 🔧 Additional Setup Instructions :

### Testing the Setup :

#### Backend : Test the API with Postman or curl :

```
curl -X POST -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

http://localhost:8000/api/predict/

#### Frontend :

http://localhost:3000

Enter a ticker (e.g., TSLA), and check for a chart and prediction.
ML Model: Verify stock_model.pkl works by running the API and checking predictions.

### Common Issues and Fixes :

Missing stock_model.pkl: Run python train_model.py and move the file to StockVibePredictor/.
Dependency Errors: Ensure requirements.txt and package.json are up-to-date. Re-run pip install or npm install.
CORS Issues: Verify django-cors-headers is installed and configured in StockVibePredictor/settings.py.
Git Conflicts: Pull latest changes (git pull origin main) and resolve conflicts in VS Code or git mergetool.

### Deployment (OPTIONAL) :

Backend (Heroku):heroku create stock-vibe-predictor
git push heroku main

Ensure stock_model.pkl is in StockVibePredictor/ or regenerated during deployment.
Front-end (Vercel):Push frontend/ to a GitHub repo, connect to Vercel, and update App.js with the Heroku API URL.
Team Task: Assign one member to handle deployment and test the live app.

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

## 🔮 Future Improvements :

- 📊 Add confidence scores to ML predictions
- 📈 Support multiple stocks in parallel
- 🧠 Include advanced indicators like MACD, Bollinger Bands, etc.
- ☁️ Save and track predictions over time
- 🔐 Add user login & personalized dashboards

---

## 🧾 License :

This project is licensed under the **MIT License**.
Feel free to fork, remix, and use — just don’t forget to credit. 😎

---

### ☕ Made with Passion, deployed with Precision, and maintained with stubborn Optimism.

---
