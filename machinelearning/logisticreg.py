from google.colab import drive
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Load the data
file_path = '/content/drive/My Drive/data.csv'
df = pd.read_csv(file_path)

# Step 3: Convert prices into binary category
df['PriceCategory'] = df['Price'].apply(lambda price: 1 if price >= 300000 else 0)

# Step 4: Define features and labels
X = df[['Size', 'Bedrooms']]
y = df['PriceCategory']

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Step 6: Get user input
size = int(input("🏠 Enter house size in sqft (e.g., 2000): "))
bedrooms = int(input("🛏️  Enter number of bedrooms (e.g., 3): "))

# Step 7: Predict category
prediction = model.predict([[size, bedrooms]])
category = 'Expensive' if prediction[0] == 1 else 'Cheap'

print(f"\n🏷️  Predicted Category: {category}")
