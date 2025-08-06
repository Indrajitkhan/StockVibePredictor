# 📊 StockVibePredictor

**StockVibePredictor** is a full-stack machine learning web app that predicts stock price trends.
Enter a stock ticker (e.g., `AAPL`) to see historical price charts and get a prediction for whether the stock will go **Up** or **Down** the next trading day.

Built with:

- 🧠 Machine Learning (Random Forest)
- ⚙️ Django (Backend + API)
- 🎨 React (Frontend)

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
StockVibePredictor/
├── frontend/              # React frontend
├── stockpredictor/        # Django project
│   └── stock_model.pkl    # Trained ML model
├── train_model.py         # Script to train the model
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

🧰 Setup Instructions

🔻 Clone the Repo
git clone https://github.com/your-username/StockVibePredictor.git
cd StockVibePredictor

⚙️ Backend Setup (Django)

# Create virtual environment

python -m venv venv

# Activate environment

# On Mac/Linux:

source venv/bin/activate

# On Windows:

venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Navigate to Django project

cd stockpredictor

# Run development server

python manage.py runserver

💻 Frontend Setup (React)

# Go to frontend folder

cd frontend

# Install node dependencies

npm install

# Run the React app

npm start

🧠 Train the ML Model

# From root directory

python train_model.py

This creates a file called stock_model.pkl.
Then move it to the Django app folder:
mv stock_model.pkl stockpredictor/

📦 The backend will load this file to make predictions.

📂 Regenerating Ignored Files (.gitignore)
Our .gitignore file keeps the repo clean by excluding files like node_modules, venv, and stock_model.pkl. When you clone the repo, you’ll need to recreate these files locally. Here’s how:
Ignored Files and How to Recreate Them

venv/: Python virtual environment for Django and ML dependencies.

Recreate:python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt

This sets up the environment and installs all Python dependencies (e.g., django, scikit-learn, yfinance).

node_modules/, frontend/node_modules/: Node.js dependencies for the React front-end.

Recreate:cd frontend
npm install

This regenerates node_modules based on package.json.

stock_model.pkl: The trained ML model for stock predictions.

Recreate:python train_model.py
mv stock_model.pkl stockpredictor/

Runs the training script to generate the Random Forest model and moves it to the Django folder.

**pycache**/, _.pyc, _.pyo, \*.pyd: Compiled Python files.

Recreate: Automatically generated when you run Python scripts (e.g., python manage.py runserver).
No manual action needed.

frontend/build/, frontend/dist/: React build output for deployment.

Recreate:cd frontend
npm run build

Generates the production-ready front-end files.

.env, _.env._: Environment files for sensitive settings (e.g., API keys).

Recreate: If needed, create a .env file in stockpredictor/ with your settings (e.g., SECRET_KEY for Django).echo "SECRET_KEY=your-django-secret-key" > stockpredictor/.env

Generate a Django secret key using a tool like djecrety.ir if required.

migrations/: Django migration files.

Recreate:cd stockpredictor
python manage.py makemigrations
python manage.py migrate

This generates and applies migrations for your Django app.

_.sqlite3, _.db: Local SQLite database.

Recreate:cd stockpredictor
python manage.py migrate

Creates a fresh SQLite database if needed (not used in this project unless you add models).

Other Ignored Files: Files like .DS_Store, .vscode/, .coverage, etc., are user-specific or temporary and don’t need recreation.

Why These Files Are Ignored

node_modules/: Huge folder, regenerated with npm install.
venv/: User-specific, avoids conflicts across machines.
stock_model.pkl: Large file, easily recreated with train_model.py.
migrations/: Environment-specific, prevents merge conflicts.
**pycache**/, \*.pyc: Temporary compiled files.
.env: May contain sensitive keys.
Others: Editor files (.vscode/), OS files (.DS_Store), or test outputs (.coverage) are irrelevant to the repo.

Team Tips

Verify Setup: After cloning, run git status to ensure ignored files don’t appear.
Regenerate Locally: Each team member must recreate venv, node_modules, and stock_model.pkl locally.
Large Files: If you need to include stock_model.pkl in the repo (e.g., for deployment), remove it from .gitignore and use Git LFS:git lfs install
git lfs track "\*.pkl"
git add .gitattributes
git add stock_model.pkl
git commit -m "Track ML model with Git LFS"
git push origin main

Consistency: Ensure all team members use the same Python (3.8+) and Node.js (16+) versions to avoid dependency issues.

🚨 Pro Tip: Run pip freeze > requirements.txt after installing dependencies to keep requirements.txt updated for the team.

🔧 Additional Setup Instructions
Testing the Setup

Backend: Test the API with Postman or curl:curl -X POST -H "Content-Type: application/json" -d '{"ticker":"AAPL"}' http://localhost:8000/api/predict/

Front-end: Open http://localhost:3000, enter a ticker (e.g., TSLA), and check for a chart and prediction.
ML Model: Verify stock_model.pkl works by running the API and checking predictions.

Common Issues and Fixes

Missing stock_model.pkl: Run python train_model.py and move the file to stockpredictor/.
Dependency Errors: Ensure requirements.txt and package.json are up-to-date. Re-run pip install or npm install.
CORS Issues: Verify django-cors-headers is installed and configured in stockpredictor/settings.py.
Git Conflicts: Pull latest changes (git pull origin main) and resolve conflicts in VS Code or git mergetool.

Deployment (Optional)

Backend (Heroku):heroku create stock-vibe-predictor
git push heroku main

Ensure stock_model.pkl is in stockpredictor/ or regenerated during deployment.
Front-end (Vercel):Push frontend/ to a GitHub repo, connect to Vercel, and update App.js with the Heroku API URL.
Team Task: Assign one member to handle deployment and test the live app.

## 🚀 Running the App

- 🔌 **Backend API**: [http://localhost:8000/api/predict/](http://localhost:8000/api/predict/)
- 🌐 **Frontend UI**: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Usage

1. Open the frontend in your browser.
2. Enter a stock ticker (like `TSLA`, `GOOGL`, `AAPL`).
3. The app will:
   - 📊 Fetch real-time historical prices
   - 📈 Display a chart
   - 🤖 Predict if the stock will go **Up** or **Down** tomorrow

---

## 🔮 Future Improvements

- 📊 Add confidence scores to ML predictions
- 📈 Support multiple stocks in parallel
- 🧠 Include advanced indicators like MACD, Bollinger Bands, etc.
- ☁️ Save and track predictions over time
- 🔐 Add user login & personalized dashboards

---

## 🧾 License

This project is licensed under the **MIT License**.
Feel free to fork, remix, and use — just don’t forget to credit. 😎

---

> Built with ☕, 📈, and a love for clean code.
