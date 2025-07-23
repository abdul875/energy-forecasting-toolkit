import pandas as pd

# load CSV into DataFrame
df = pd.read_csv(r"F:/JU/MY_Project/JU_project/weather_with_renewable_source.csv")

solar_df = df[df['renewable_source'] == 'Hydro']
# Group by the 'label' column
grouped = solar_df.groupby('name').size()
print(grouped)


import pandas as pd
import numpy as np

def create_wind_samples(n=10):
    base = {
        'solarradiation': 50,
        'uvindex': 1,
        'cloudcover': 30,
        'windspeed': 30,
        'windgust': 50,
        'precip': 1,
        'snow': 0,
        'snowdepth': 0,
        'day_length_minutes': 650
    }
    samples = []
    for _ in range(n):
        sample = {k: max(0, v + np.random.randint(-5, 6)) for k, v in base.items()}
        samples.append(sample)
    return pd.DataFrame(samples)

wind_samples = create_wind_samples()

wind_samples.to_csv("F:/JU/MY_Project/JU_project/22.csv", index=False)