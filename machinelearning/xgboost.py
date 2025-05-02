from google.colab import drive
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Load CSV (Update path if saved in a subfolder)
file_path = '/content/drive/My Drive/house_data_xgb.csv'
df = pd.read_csv(file_path)

# Step 3: Prepare features and target
X = df[['Size', 'Bedrooms']]
y = df['Price']

# Step 4: Train XGBoost model
model = XGBRegressor()
model.fit(X, y)

# Step 5: User input for prediction
size = int(input("🏠 Enter house size in sqft (e.g., 2200): "))
bedrooms = int(input("🛏️  Enter number of bedrooms (e.g., 3): "))

# Step 6: Make prediction
predicted_price = model.predict([[size, bedrooms]])
print(f"\n💰 Predicted House Price (XGBoost): ${int(predicted_price[0])}")
