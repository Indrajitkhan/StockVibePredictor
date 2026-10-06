from pathlib import Path

# This is the exact code from your views.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

print(f"The calculated BASE_DIR is: {BASE_DIR}")
