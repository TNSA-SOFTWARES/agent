from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer

app = Flask(__name__)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the Q-learning model using my_h5py and Keras
model_path = 'model.h5'
q_learning_model = tf.keras.models.load_model(model_path)

# Function to preprocess input text using GPT-2 tokenizer
def preprocess_input(text):
    text = text.replace(",", "").replace(".", "")
    text = text.replace("%", " percent")
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    padded_sequence = np.pad(tokenized_text, (0, 1024 - len(tokenized_text)), 'constant', constant_values=0)
    return np.array(padded_sequence)

# Function to predict with the pre-trained Q-learning model
def predict_price(state, user_entered_price):
    predicted_prices = q_learning_model.predict(np.expand_dims(state, axis=0))
    predicted_price = predicted_prices[0][0]
    final_predicted_price = predicted_price + user_entered_price
    return final_predicted_price

# Function to generate recommendation based on predicted price
def generate_recommendation(predicted_price, user_entered_price, risk_tolerance):
    price_difference = predicted_price - user_entered_price

    if price_difference > 0 and price_difference >= risk_tolerance * user_entered_price / 100:
        return "Consider buying. The model predicts a significant potential increase in Bitcoin's price."
    elif price_difference < 0 and abs(price_difference) >= risk_tolerance * user_entered_price / 100:
        return "Consider selling. The model predicts a significant potential decrease in Bitcoin's price."
    else:
        return "Hold. The model predicts no significant change or the change is within your risk tolerance."

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_entered_price = float(request.form['price'].replace(",", "").replace(".", ""))
        user_input = request.form['description']
        risk_tolerance = float(request.form['risk'].replace(",", "").replace(".", ""))
        
        state = preprocess_input(user_input)
        final_predicted_price = predict_price(state, user_entered_price)
        recommendation = generate_recommendation(final_predicted_price, user_entered_price, risk_tolerance)

        return render_template('index.html', predicted_price=final_predicted_price, recommendation=recommendation)

    except ValueError:
        return render_template('index.html', message="Invalid input. Please enter valid numeric values.")

if __name__ == '__main__':
    app.run(debug=True)
