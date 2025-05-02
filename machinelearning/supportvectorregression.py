from google.colab import drive
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Load CSV from Google Drive
file_path = '/content/drive/My Drive/data.csv'  # Adjust path if in subfolder
df = pd.read_csv(file_path)

# Step 3: Features and target
X = df[['Size', 'Bedrooms']]
y = df['Price']

# Step 4: Feature scaling (SVM needs it!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Step 5: Train SVR model
model = SVR(kernel='linear')  # You can try 'linear', 'poly', or 'rbf'
model.fit(X_scaled, y_scaled)

# Step 6: User input
size = int(input("🏠 Enter house size in sqft (e.g., 2000): "))
bedrooms = int(input("🛏️  Enter number of bedrooms (e.g., 3): "))

# Step 7: Predict
input_scaled = scaler_X.transform([[size, bedrooms]])
predicted_scaled = model.predict(input_scaled)
predicted_price = scaler_y.inverse_transform([[predicted_scaled[0]]])

print(f"\n💰 Predicted House Price (SVM): ${int(predicted_price[0][0])}")
