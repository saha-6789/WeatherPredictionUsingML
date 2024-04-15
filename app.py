from flask import Flask, render_template, request
from datetime import datetime
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        date = request.form['date']
       
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day

        output = predict_weather(city, year, month, day)  
        return render_template('result.html', output=output)
    return render_template('index.html')

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
def predict_weather(city, year, month, day):
    # Load models and encoders
    model_temp = load_model('model_temp.sav')
    model_humidity = load_model('model_humidity.sav')
    model_conditions = load_model('model_conditions.sav')
    city_encoder = load_model('city_encoder.sav')
    label_encoder = load_model('label_encoder.sav')

 
    city_encoded = city_encoder.transform([[city]]).toarray()

    num_cities = city_encoded.shape[1]
    city_feature_names = [f'{i}' for i in range(num_cities)]

    feature_values = [year, month, day] + city_encoded[0].tolist()
    feature_names = ['year', 'month', 'day'] + city_feature_names
    features = pd.DataFrame([feature_values], columns=feature_names)

    predicted_temp = model_temp.predict(features)[0]
    predicted_humidity = model_humidity.predict(features)[0]
    predicted_condition_encoded = model_conditions.predict(features)[0]
    predicted_condition = label_encoder.inverse_transform([predicted_condition_encoded])[0]

    return f"The date you selected is {day}-{month}-{year}.The temperature is around {predicted_temp} degree Celcius, the humidty in {city} is {predicted_humidity}. The weather for the date you selected in {city} is {predicted_condition}. Have a nice day in {city}."

if __name__ == '__main__':
    app.run(debug=True)
