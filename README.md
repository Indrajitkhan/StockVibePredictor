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
│   │   │── /stockpredict.log/
|   |
│   │── /Scripts/
│   │   │── /TrainModel.py/
│   │   │── /stock_model.pkl/
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
|   |
│   │── /Apps/
|   |   |
│   │   │── /Dashboard/
|   |   |   |
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
|   |   |-- App.css
|   |   |-- App.js
|   |   |-- index.css
|   |   |-- index.js
|   |
│   │── package.json
│   │── package-lock.json
│   │── webpack.config.js
│   │── vite.config.js
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
