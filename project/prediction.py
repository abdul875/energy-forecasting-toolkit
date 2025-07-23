import joblib
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

# ==== 1. Load Label Encoder for Inverse Transform ====
# Use the same classes from training if known
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Hydropower', 'Solar', 'Wind']  # Adjust based on your dataset

# ==== 2. Load Your Saved Model ====
model_path = input("🧠 Enter the path to your saved model (e.g., F:/JU/MY_Project/JU_project/saved_models/random_forest_model.pkl): ")
try:
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==== 3. Function to Get Weather Data ====
def get_user_input_from_visualcrossing():
    location = input("📍 Enter your location (e.g., Dhaka, New York): ")
    api_key = "NWHFUVASRA2ZHRHXSJQYJ9VTB"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/today?unitGroup=metric&key={api_key}&contentType=json"

    try:
        response = requests.get(url)
        data = response.json()

        if 'days' not in data or len(data['days']) == 0:
            print("❌ No weather data available for this location.")
            return None

        day = data['days'][0]
        sunrise = pd.to_datetime(day.get('sunrise'), errors='coerce')
        sunset = pd.to_datetime(day.get('sunset'), errors='coerce')
        day_length_minutes = (sunset - sunrise).total_seconds() / 60 if pd.notnull(sunrise) and pd.notnull(sunset) else 0

        return pd.DataFrame([{
            'solarradiation': day.get('solarradiation', 0),
            'uvindex': day.get('uvindex', 0),
            'cloudcover': day.get('cloudcover', 0),
            'windspeed': day.get('windspeed', 0),
            'windgust': day.get('windgust', 0),
            'precip': day.get('precip', 0),
            'snow': day.get('snow', 0),
            'snowdepth': day.get('snowdepth', 0),
            'day_length_minutes': day_length_minutes
        }])

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ==== 4. Predict Using the Model ====
input_df = get_user_input_from_visualcrossing()

if input_df is not None:
    try:
        prediction = model.predict(input_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        print(f"\n✅ Predicted Renewable Energy Source: **{predicted_label}**")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
else:
    print("❌ No input data available.")
