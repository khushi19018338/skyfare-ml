from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("transformer.pkl", "rb") as f:
    transformer = pickle.load(f)

# Hardcoded list of airline names and their display names
airlines = [
    {"code": "Vistara", "display": "Vistara"},
    {"code": "Air_India", "display": "Air India"},
    {"code": "Indigo", "display": "Indigo"},
    {"code": "GO_FIRST", "display": "Go First"},
    {"code": "AirAsia", "display": "Air Asia"},
    {"code": "SpiceJet", "display": "Spice Jet"},
]

# Hardcoded arrays for arrival and departure times
time = [
    {"code": "Morning", "display": "Morning"},
    {"code": "Early_Morning", "display": "Early Morning"},
    {"code": "Evening", "display": "Evening"},
    {"code": "Night", "display": "Night"},
    {"code": "Afternoon", "display": "Afternoon"},
    {"code": "Late_Night", "display": "Late Night"},
]

# Hardcoded list of source cities
source_cities = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Chennai",
    "Kolkata",
    "Hyderabad"
]

# Hardcoded list of destination cities
destination_cities = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Chennai",
    "Kolkata",
    "Hyderabad"
]
unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY')

# Home route
@app.route('/')
def home():

    return render_template('index.html', 
                           unsplash_access_key=unsplash_access_key,
                           airlines=airlines, 
                           dep_arr_time=time, 
                           source_cities=source_cities,
                           destination_cities=destination_cities)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        airline = request.form.get("airline")
        source_city = request.form.get("source_city")
        departure_time = request.form.get("departure_time")
        stops = request.form.get("stops")  
        arrival_time = request.form.get("arrival_time")
        destination = request.form.get("destination")
        flight_class = request.form.get("class")



        # Prepare features array
        features = [airline, source_city, departure_time, stops, arrival_time, flight_class,destination]
        feature_columns = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 
                           'class', 'destination_city']
        features_df = pd.DataFrame([features], columns=feature_columns)
        # Transform features
        transformed_features = transformer.transform(features_df)

        # Predict using the model
        prediction = model.predict(transformed_features)
        price=abs(prediction[0]) if prediction[0]<0 else prediction[0]
        return render_template('index.html', 
                               unsplash_access_key=unsplash_access_key,
                               airlines=airlines, 
                               dep_arr_time=time, 
                               source_cities=source_cities,
                               destination_cities=destination_cities,
                               prediction_text=f'Predicted Flight Price: Rs.{price:.2f}')
    
    except Exception as e:
        return render_template('index.html',
                               unsplash_access_key=unsplash_access_key, 
                               airlines=airlines, 
                               dep_arr_time=time, 
                               source_cities=source_cities,
                               destination_cities=destination_cities,
                               prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
