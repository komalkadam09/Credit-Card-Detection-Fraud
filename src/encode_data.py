import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_data = pd.read_csv("data/fraudTrain.csv")

# Remove unnecessary columns
train_data = train_data.drop(
    columns=[
        "Unnamed: 0",
        "first",
        "last",
        "street",
        "trans_num",
        "cc_num"
    ]
)

# Create LabelEncoder object
encoder = LabelEncoder()

# Columns that contain text
text_columns = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job"
]

# Convert text into numbers
for column in text_columns:
    train_data[column] = encoder.fit_transform(train_data[column])

print(train_data.head())