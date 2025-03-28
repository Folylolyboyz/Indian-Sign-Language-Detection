import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Data Processing/total.csv")

# Prepare features and target
X = df.dropna(axis=0, how='any')
X = X[X.count(axis=1) == 43]
label = X["label"]
X = X.drop(columns=["label"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(label)

print(X)

# Save label encoder classes
joblib.dump(label_encoder.classes_, "Train/label_encoder_classes.pkl")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "Train/random_forest_model.pkl")

print("Model training complete. Saved as 'random_forest_model.pkl'")