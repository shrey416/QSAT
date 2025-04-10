# QSAT - Quick Soil Analysis Tool

QSAT (Quick Soil Analysis Tool) is an innovative, machine learning-powered web application designed to empower farmers and soil analysts by predicting soil nutrient levels with high accuracy. Using water level data (0ml, 25ml, 50ml) and spectrometer readings from soil samples, QSAT delivers precise predictions for critical soil parameters: **Moisture Capacity**, **Temperature**, **Nitrogen**, **Electrical Conductivity**, **Phosphorus**, **Potassium**, and **pH**. Built with cutting-edge technologies, QSAT combines a robust regression model with an AI-driven chatbot to provide actionable insights for optimizing soil health and crop yield.

#### User Input Interface:
![image](https://github.com/user-attachments/assets/a793a836-91cb-484c-8c8b-df8d1c5e3d41)
#### Predicted Values:
![image](https://github.com/user-attachments/assets/f64c5974-627e-4d00-a7f3-a8b87f61581a)
#### Soil Health Assistant:
![image](https://github.com/user-attachments/assets/f139f58d-c508-440f-8985-2a8d258e33c7)

## Key Features

- **Soil Nutrient Prediction**: Accurately predicts seven key soil nutrients using water level and spectrometer data.
- **Flexible Input System**: Accepts a variable number of wavelength intensity values (minimum 3 out of 18) for customizable predictions.
- **Targeted Accuracy Optimization**: Enables users to maximize prediction accuracy for a specific nutrient (e.g., Nitrogen) by recommending the most correlated wavelengths (e.g., 610nm, 565nm, 435nm) when limited inputs are available.
- **Soil Health Assistant**: Integrates a Gemini-powered chatbot that analyzes soil data and offers tailored recommendations on crop selection, fertilizers, and farming techniques.
- **Performance Insights**: Features a dedicated page displaying model performance metrics (MAE, RMSE, R²) in a tabular format across water levels and target columns.

## Technologies Used

- **Backend**: Python, Flask, LightGBM, Optuna, Pandas, NumPy, Scikit-learn, Joblib, Gunicorn, Google Generative AI, Python-dotenv
- **Frontend**: ReactJS, TailwindCSS

## Installation and Setup

Follow these steps to set up and run QSAT locally:

### Backend

1. **Navigate to the Backend Directory**:
```
cd backend
```

2. **In Backend folder create a python virtual environment using:**
```
python -m venv model
```

3. **To activate the virtual environment:**
Windows : 
```
.\model\Scripts\activate
```

4. **After activating install dependencies:**
```
pip install flask flask-cors lightgbm pandas numpy scikit-learn optuna joblib gunicorn google-generativeai python-dotenv
```

5. **To run backend server:**
(while virtual environment in python is active)
```
flask run
```

### Frontend

1. **Go to frontend folder:**
```
cd frontend
```

2. **Install dependencies using npm:**
```
npm install
```

3. **Run the frontend server:**
```
npm run dev
```
##### Copyright (C) 2025 Shrey Parsania
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
