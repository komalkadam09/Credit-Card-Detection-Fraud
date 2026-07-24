import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
train_data = pd.read_csv("data/fraudTrain.csv")

# Remove unnecessary columns
train_data = train_data.drop(columns=[
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num",
    "trans_date_trans_time",
    "dob"
])

# Encode text columns
encoder = LabelEncoder()

text_columns = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job"
]

for column in text_columns:
    train_data[column] = encoder.fit_transform(train_data[column])

# Features and Target
X = train_data.drop("is_fraud", axis=1)
y = train_data["is_fraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Training Data After SMOTE:")
print(X_train.shape)
print(y_train.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Random Forest Accuracy:")
print(accuracy)

print("\n==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")