import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dead_alert.csv")

# Remove NaN
data = data.dropna()

# Features
X = data[['Heart_Rate','Oxygen_Saturation','Body_Temperature',
          'Systolic_Blood_Pressure','Respiratory_Rate',
          'Blood_pH','White_Blood_Cell_Count']]

# Target
y = data['Sepsis_Output']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Training Completed\n")

# -------- USER INPUT --------

hr = float(input("Enter Heart Rate: "))
o2 = float(input("Enter Oxygen Saturation: "))
temp = float(input("Enter Body Temperature: "))
sbp = float(input("Enter Systolic Blood Pressure: "))
resp = float(input("Enter Respiratory Rate: "))
ph = float(input("Enter Blood pH: "))
wbc = float(input("Enter White Blood Cell Count: "))

# Convert input to DataFrame (IMPORTANT)
input_data = pd.DataFrame([[hr,o2,temp,sbp,resp,ph,wbc]], columns=[
    'Heart_Rate',
    'Oxygen_Saturation',
    'Body_Temperature',
    'Systolic_Blood_Pressure',
    'Respiratory_Rate',
    'Blood_pH',
    'White_Blood_Cell_Count'
])

# Prediction
prediction = model.predict(input_data)

# Result
if prediction[0] == 1:
    print("\n⚠ Sepsis Detected")
else:
    print("\n✓ Patient Normal")