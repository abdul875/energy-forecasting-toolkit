# energy-forecasting-toolkit
🌿 Renewable Energy Potential Predictor
This project uses weather data and machine learning techniques to analyze and predict the most suitable renewable energy source—Solar, Wind, or Hydro—for a specific location and time. By processing environmental indicators like solar radiation, wind speed, humidity, and precipitation, the system calculates energy scores and identifies the dominant renewable energy potential for each record.
--------------------------------------------------------------------------
🔍 Key Features
🌞 Solar Score based on solar radiation, UV index, cloud cover, and day length

💨 Wind Score using wind speed, gusts, pressure, and temperature

🌧️ Hydro Score from precipitation levels and coverage

📊 Automatic source classification: Solar, Wind, or Hydro

🧹 Handles missing data and normalizes values for fair scoring

📁 Supports district-wise weather reports (CSV format)

💾 Outputs annotated datasets with predicted energy source labels

⚙️ Tech Stack
------------------------------------------------------------------------
Python, Pandas, NumPy

Joblib (for model persistence if extended)

Easily extendable with ML/DL models for future energy forecasting

📌 Use Cases
------------------------------------------------------------------------
Renewable energy planning & policy making

Smart city energy grid analysis

Educational datasets for data science projects

Decision support for solar/wind/hydro infrastructure placement

📊 Data Source
---------------------------------------------------------------------------
The weather data used in this project is collected from:

🔗 Visual Crossing Weather
You can download historical and real-time weather data for specific districts, cities, or regions in CSV format. The data includes attributes like:

solarradiation, uvindex, cloudcover

windspeed, windgust, humidity

precip, precipprob, precipcover

sunrise, sunset, temperature, solarenergy

You may need to sign up for a free API key to access full historical datasets.


