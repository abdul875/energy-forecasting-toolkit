import pandas as pd
import numpy as np
import os
import joblib

df = pd.read_csv("F:/JU/MY_Project/JU_project/District_wise_weather_report.csv")



def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Step 1: Create additional required feature - day length in minutes
df['sunrise'] = pd.to_datetime(df['sunrise'])
df['sunset'] = pd.to_datetime(df['sunset'])
df['day_length_minutes'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 60

# Fill NaNs in required columns
fill_columns = [
    'solarradiation', 'uvindex', 'cloudcover', 'windspeed', 'windgust',
    'precip', 'precipprob', 'precipcover', 'day_length_minutes',
    'sealevelpressure', 'temp', 'humidity', 'solarenergy'
]

for col in fill_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Step 2: Normalize scores
df['solar_score'] = (
    0.4 * normalize(df['solarradiation']) +
    0.3 * normalize(df['uvindex']) +
    0.2 * (1 - normalize(df['cloudcover'])) +
    0.1 * normalize(df['day_length_minutes'])
)

df['wind_score'] = (
    0.4 * normalize(df['windspeed']) +
    0.2 * normalize(df['windgust']) +
    0.15 * (1 - normalize(df['sealevelpressure'])) +  # Lower pressure may suggest stronger wind
    0.15 * normalize(df['temp']) +                    # More heating → more convection
    0.05 * normalize(df['humidity']) +                # Air moisture affects buoyancy
    0.05 * normalize(df['solarenergy'])               # Thermal energy drives movement
)

df['hydro_score'] = (
    0.5 * normalize(df['precip']) +
    0.25 * normalize(df['precipprob']) +
    0.25 * normalize(df['precipcover'])
)

# Step 3: Determine dominant renewable source
def dominant_source(row):
    scores = {
        "Solar": row['solar_score'],
        "Wind": row['wind_score'],
        "Hydro": row['hydro_score']
    }
    return max(scores, key=scores.get)

df['renewable_source'] = df.apply(dominant_source, axis=1)

df.to_csv("F:/JU/MY_Project/JU_project/weather_with_renewable_source.csv", index=False)