import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Convert decimal hours to HH:MM
# ----------------------------
def convert_to_hours_minutes(decimal_hours):
    hours = int(decimal_hours)
    minutes = int(round((decimal_hours - hours) * 60))
    return f"{hours}h {minutes}m"

# Dataset path
FILEPATH = r"C:\Users\amate\OneDrive\Desktop\stock price predictor 2\PLANE\flight_delay_dataset.csv"

# Load dataset
df = pd.read_csv(FILEPATH)

target = "hours_until_clear"
features = [
    "wind_speed_kts",
    "wind_direction_deg",
    "humidity_%",
    "pressure_hPa",
    "fog_density",
    "visibility_km"
]

X = df[features]
y = df[target]

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ]), features)
])

# Model
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Model evaluation
pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

# -------------- Streamlit UI ------------------

st.title("🌤 Weather Clearance Time Prediction App")
st.write("Enter weather parameters to predict **hours until visibility clears**")

# Display model performance
st.write(f"### 📈 Model Performance\nMSE: **{mse:.4f}**, R²: **{r2:.4f}**")

# Input Form
wind_speed = st.number_input("Wind Speed (kts)", min_value=0.0, step=0.1)
wind_direction = st.number_input("Wind Direction (deg)", min_value=0.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, step=0.1)
fog_density = st.number_input("Fog Density", min_value=0.0, step=0.01)
visibility = st.number_input("Visibility (km)", min_value=0.0, step=0.1)

if st.button("Predict"):
    user_data = pd.DataFrame([[wind_speed, wind_direction, humidity, pressure, fog_density, visibility]],
                             columns=features)

    result = pipeline.predict(user_data)[0]
    readable = convert_to_hours_minutes(result)

    st.success("Prediction Completed!")
    st.write(f"### ⏱ Decimal Predicted Hours: **{result:.2f}**")
    st.write(f"### ⏳ Readable Format: **{readable}**")