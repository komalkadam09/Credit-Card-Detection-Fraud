import pandas as pd

# Load the training dataset
train_data = pd.read_csv("data/fraudTrain.csv")

print("Original Shape:")
print(train_data.shape)

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

print("\nShape After Removing Columns:")
print(train_data.shape)

print("\nRemaining Columns:")
print(train_data.columns)