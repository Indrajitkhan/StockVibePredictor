#!/bin/bash

# Run Postman tests using Newman

echo "🚀 Running StockVibe API Tests..."

# Install Newman if not installed
if ! command -v newman &> /dev/null; then
    echo "Installing Newman..."
    npm install -g newman newman-reporter-html
fi

# Run tests for each environment
echo "Testing Development Environment..."
mkdir -p ./results
newman run ../Collections/StockVibePredictor_API_v2.postman_collection.json \
    -e ../Environments/Development.postman_environment.json \
    --reporters cli,html \
    --reporter-html-export ./results/dev-test-results.html

echo "✅ Tests Complete! Check Scripts/results/ for reports"
