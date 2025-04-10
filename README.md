# QSAT
Quick Soil Analysis Tool 

## Technologies Used : 
Python, Flask, ReactJS, TailwindCSS, LightGBM, Gemini, Optuna

How to run the website along with model

## BACKEND =>

### 1. Go to backend folder=>
cd backend

### 2. In Backend folder create a python virtual environment using =>
python -m venv model

### 3. To activate the virtual environment =>
.\model\Scripts\activate 

### 4. After activating install dependencies =>
pip install flask flask-cors lightgbm pandas numpy scikit-learn optuna joblib gunicorn google-generativeai python-dotenv

### 5. To run backend server =>
(while virtual environment in python is active)
flask run

## FRONTEND => 

### 1. Go to frontend folder=>
cd frontend

### 2. Install dependencies using npm =>
npm install

### 3. Run the frontend server =>
npm run dev
