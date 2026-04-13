import pandas as pd
import pickle

# Load the dataset and indices
df = pd.read_parquet("./data/planner/train.parquet")
with open("./data/planner/answerable_indices.pkl", "rb") as f:
    indices = pickle.load(f)
model_answerable = indices["model_answerable"]

# Remove model_answerable indices
df_filtered = df.drop(df.index[model_answerable])

# Remove duplicated questions
df_filtered = df_filtered.drop_duplicates(subset=["question"])

# Save the filtered dataset
df_filtered.to_parquet("./data/planner/train_filtered.parquet")


# Get the original indices of the filtered dataset
filtered_indices = df_filtered.index.tolist()

print(f"Number of elements left: {len(df_filtered)}")
