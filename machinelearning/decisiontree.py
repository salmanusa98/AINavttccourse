import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Step 1: Create the dataset
data = {
    'Size': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 4, 4, 5],
    'Price': [150000, 200000, 250000, 275000, 300000]
}
df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['Size', 'Bedrooms']]  # Independent variables
y = df['Price']               # Dependent variable

# Step 3: Train the decision tree model
model = DecisionTreeRegressor()
model.fit(X, y)

# Step 4: Get user input
size = int(input("Enter house size in sqft (e.g., 1800): "))
bedrooms = int(input("Enter number of bedrooms (e.g., 3): "))

# Step 5: Predict price
predicted_price = model.predict([[size, bedrooms]])

# Step 6: Show result
print(f"\n💰 Predicted House Price: ${int(predicted_price[0])}")
