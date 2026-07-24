import pandas as pd

# Load dataset
train_data = pd.read_csv("data/fraudTrain.csv")

print("===== UNIQUE VALUES =====\n")

print("Gender:")
print(train_data["gender"].unique())

print("\nCategory:")
print(train_data["category"].unique())

print("\nState:")
print(train_data["state"].unique())