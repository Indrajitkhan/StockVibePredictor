StockVibePredictor
A full-stack web app that predicts stock price trends using machine learning. Users enter a stock ticker (e.g., AAPL) to see historical prices and a prediction for whether the stock will go up or down tomorrow. Built with Django, scikit-learn, and React for a modern, industry-ready project.
Features

Real-time Stock Data: Fetches data using yfinance.
ML Prediction: Random Forest model predicts up/down trends based on price, volume, and technical indicators (RSI, moving averages).
Interactive UI: React front-end with Chart.js for price charts.
REST API: Django backend serves predictions and data.

Tech Stack

Backend: Django + Django REST Framework (Python)
ML: scikit-learn (Random Forest Classifier)
Front-end: React + Chart.js
Data: yfinance API
Deployment (Optional) : Heroku (backend), Vercel (front-end)

Setup

Clone the repo:
git clone https://github.com/your-username/StockVibePredictor.git
cd StockVibePredictor

Backend Setup:
python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt
cd stockpredictor
python manage.py runserver

Front-end Setup:
cd frontend
npm install
npm start

Train ML Model:
python train_model.py

Move stock_model.pkl to stockpredictor/.

Run the App:

Backend: http://localhost:8000/api/predict/
Front-end: http://localhost:3000

Usage

Enter a stock ticker (e.g., TSLA) in the UI.
View historical prices and a prediction (Up/Down).

Future Improvements

Add confidence scores for predictions.
Support multiple stocks.
Include more technical indicators (e.g., MACD).

License
MIT License
