import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, render_template_string, jsonify

app = Flask(__name__)

# Load the pickle model safely
MODEL_PATH = "Bank_Customer.pkl"
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Expected features from Bank_Customer.pkl
FEATURE_NAMES = [
    'credit_score', 'country', 'gender', 'age', 'tenure', 
    'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary'
]

# Interactive CSS/HTML Single-file Template 
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Retention Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        body {
            background: radial-gradient(circle at 10% 20%, rgb(18, 28, 41) 0%, rgb(11, 15, 23) 100%);
            color: #f3f4f6;
            font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh;
        }
        .main-card {
            background: rgba(23, 37, 54, 0.65);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        }
        .form-control, .form-select {
            background-color: rgba(13, 22, 38, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            border-radius: 10px;
            padding: 12px;
            transition: all 0.3s ease;
        }
        .form-control:focus, .form-select:focus {
            background-color: rgba(13, 22, 38, 0.95);
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
            color: #fff;
        }
        .form-label {
            font-weight: 500;
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 6px;
        }
        .btn-gradient {
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
            border: none;
            color: white;
            padding: 14px;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        .btn-gradient:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
            color: white;
        }
        .result-box {
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            font-weight: 600;
            display: none;
            animation: fadeIn 0.5s ease forwards;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-xl-9 col-lg-10">
                
                <div class="text-center mb-5">
                    <h2 class="fw-bold text-white"><i class="bi bi-shield-check text-info me-2"></i>Bank Customer Retention Analytics</h2>
                    <p class="text-muted">Enter target customer data metrics below to query prediction diagnostics.</p>
                </div>

                <div class="main-card p-4 p-md-5">
                    <form id="predictionForm">
                        <div class="row g-4">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Credit Score</label>
                                    <input type="number" class="form-control" name="credit_score" min="300" max="850" placeholder="e.g. 650" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Geography / Country</label>
                                    <select class="form-select" name="country" required>
                                        <option value="France">France</option>
                                        <option value="Spain">Germany</option>
                                        <option value="Germany">Spain</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Gender</label>
                                    <select class="form-select" name="gender" required>
                                        <option value="Male">Male</option>
                                        <option value="Female">Female</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Age</label>
                                    <input type="number" class="form-control" name="age" min="18" max="100" placeholder="e.g. 34" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Tenure (Years)</label>
                                    <input type="number" class="form-control" name="tenure" min="0" max="10" placeholder="e.g. 5" required>
                                </div>
                            </div>

                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Account Balance ($)</label>
                                    <input type="number" step="0.01" class="form-control" name="balance" placeholder="e.g. 72000.50" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Number of Products</label>
                                    <input type="number" class="form-control" name="products_number" min="1" max="4" placeholder="e.g. 2" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Has Credit Card?</label>
                                    <select class="form-select" name="credit_card" required>
                                        <option value="1">Yes</option>
                                        <option value="0">No</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Is Active Member?</label>
                                    <select class="form-select" name="active_member" required>
                                        <option value="1">Yes</option>
                                        <option value="0">No</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Estimated Annual Salary ($)</label>
                                    <input type="number" step="0.01" class="form-control" name="estimated_salary" placeholder="e.g. 105000.00" required>
                                </div>
                            </div>
                        </div>

                        <div class="mt-4">
                            <button type="submit" class="btn btn-gradient w-100 fs-5">
                                Generate Risk Prediction Analytics
                            </button>
                        </div>
                    </form>

                    <div id="resultContainer" class="result-box mt-4"></div>

                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            const resultBox = document.getElementById('resultContainer');
            resultBox.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const resData = await response.json();
                
                if(resData.success) {
                    resultBox.style.display = 'block';
                    if (resData.prediction === 1) {
                        resultBox.style.background = 'rgba(239, 68, 68, 0.2)';
                        resultBox.style.border = '1px solid #ef4444';
                        resultBox.style.color = '#fca5a5';
                        resultBox.innerHTML = `<id class="bi bi-exclamation-triangle-fill me-2"></id> High Risk Alert: Customer is likely to Churn! (Probability: ${resData.probability}%)`;
                    } else {
                        resultBox.style.background = 'rgba(34, 197, 94, 0.2)';
                        resultBox.style.border = '1px solid #22c55e';
                        resultBox.style.color = '#86efac';
                        resultBox.innerHTML = `<id class="bi bi-check-circle-fill me-2"></id> Healthy Metrics: Customer is likely to Stay. (Probability: ${resData.probability}%)`;
                    }
                }
            } catch (err) {
                console.error(err);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Format mapping inputs explicitly to match string categories if label-encoded / expected by model
        # Adjust if your pipeline expects numeric inputs for string categories
        input_data = {
            'credit_score': float(data['credit_score']),
            'country': data['country'],
            'gender': data['gender'],
            'age': int(data['age']),
            'tenure': int(data['tenure']),
            'balance': float(data['balance']),
            'products_number': int(data['products_number']),
            'credit_card': int(data['credit_card']),
            'active_member': int(data['active_member']),
            'estimated_salary': float(data['estimated_salary'])
        }

        # Structure input into a unified pandas row dataframe 
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Predict Class & Probability Matrix
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        confidence = round(float(probabilities[prediction]) * 100, 2)

        return jsonify({'success': True, 'prediction': prediction, 'probability': confidence})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
