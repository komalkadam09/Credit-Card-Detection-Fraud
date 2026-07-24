import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Features (X)
X = train_data.drop("is_fraud", axis=1)

# Target (y)
y = train_data["is_fraud"]

print("Features Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

# ==========================
# Apply SMOTE
# ==========================
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Training Features Shape:", X_train.shape)
print("Training Target Shape:", y_train.shape)

# ==========================
# Create Logistic Regression Model
# ==========================
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Model Accuracy:")
print(accuracy)

print("\n==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("Classification Report:")
print(classification_report(y_test, y_pred))