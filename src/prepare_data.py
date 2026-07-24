import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Create Label Encoder
encoder = LabelEncoder()

# Convert text columns into numbers
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

print("Dataset Shape:")
print(train_data.shape)

print("\nFirst 5 Rows:")
print(train_data.head())