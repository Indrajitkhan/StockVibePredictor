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

## 🧰 Setup Instructions

### 🔻 Clone the Repo

```bash
git clone https://github.com/your-username/StockVibePredictor.git
cd StockVibePredictor
```

---

### ⚙️ Backend Setup (Django)

```bash
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
```

---

### 💻 Frontend Setup (React)

```bash
# Go to frontend folder
cd frontend

# Install node dependencies
npm install

# Run the React app
npm start
```

---

### 🧠 Train the ML Model

```bash
# From root directory
python train_model.py
```

This creates a file called `stock_model.pkl`.

Then move it to the Django app folder:

```bash
mv stock_model.pkl stockpredictor/
```

> 📦 The backend will load this file to make predictions.

---

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
