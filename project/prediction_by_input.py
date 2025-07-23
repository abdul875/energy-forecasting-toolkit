import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os  # for getting filename from path

# ==== 1. Label Encoder ====
label_encoder = LabelEncoder()
label_encoder.fit(['Hydro', 'Solar', 'Wind'])

# ==== 2. Load Multiple Models (Hard-coded paths) ====
model_paths = [
    r"F:/JU/MY_Project/JU_project/saved_models/adaboost_classifier_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/bagging_classifier_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/decision_tree_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/logistic_regression_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/random_forest_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/stacking_classifier_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/svm_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/voting_classifier_model.pkl",
    r"F:/JU/MY_Project/JU_project/saved_models/xgboost_model.pkl"
]

models = {}

for path in model_paths:
    try:
        loaded_model = joblib.load(path)
        models[path] = loaded_model
        print(f"✅ Loaded model: {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Error loading model {os.path.basename(path)}: {e}")

if not models:
    print("❌ No valid models loaded. Exiting.")
    exit()

# ==== 3. Input function to get numeric input safely ====
def input_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value.")

# ==== 4. Collect feature inputs from user ====
features = [
    'solarradiation', 'uvindex', 'cloudcover', 'windspeed',
    'windgust', 'precip', 'snow', 'snowdepth', 'day_length_minutes'
]

print("Please enter values for the following weather features:")

input_data = {}
for feature in features:
    val = input_float(f"- {feature}: ")
    input_data[feature] = val

input_df = pd.DataFrame([input_data])

print("\nInput data for prediction:")
print(input_df)

# ==== 5. Predict with all models ====
for path, model in models.items():
    try:
        prediction = model.predict(input_df)
        predicted_class = prediction[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        model_name = os.path.basename(path)
        print(f"Model: {model_name} -> Prediction: {predicted_label}")
    except Exception as e:
        print(f"❌ Prediction failed for model {os.path.basename(path)}: {e}")
