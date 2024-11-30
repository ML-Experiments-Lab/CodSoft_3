import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('credit_card_fraud_detection.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('frequency_encoding.pickle', 'rb') as freq_file:
    freq_encoding = pickle.load(freq_file)

scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    cc_num = request.form['cc_num']
    category = request.form['category']
    amt = float(request.form['amt'])
    gender = request.form['gender']
    city = request.form['city']
    state = request.form['state']
    zip_code = request.form['zip']

    # Apply frequency encoding for categorical features
    category_encoded = freq_encoding['category'].get(category, 0)
    gender_encoded = freq_encoding['gender'].get(gender, 0)
    city_encoded = freq_encoding['city'].get(city, 0)
    state_encoded = freq_encoding['state'].get(state, 0)

    # Prepare the feature vector (same features used for model training)
    input_features = np.array([
        [cc_num, category_encoded, amt, gender_encoded, city_encoded, state_encoded, zip_code]
    ], dtype=object)

    # If your model requires specific feature transformations (e.g., encoding)
    # Make sure these transformations are applied here, as they were in training.

    # Scale the input features
    input_features_scaled = scaler.fit_transform(input_features)  # or use scaler.transform() if fitting is already done

    # Make prediction
    prediction = model.predict(input_features_scaled)

    # Return prediction result
    prediction_text = "Fraudulent Transaction" if prediction[0] == 1 else "Non-Fraudulent Transaction"
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
