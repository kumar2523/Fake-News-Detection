import pandas as pd

# Load both files
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine both
data = pd.concat([fake, true])

# Save as one dataset
data.to_csv("dataset.csv", index=False)

print("dataset.csv created successfully!")