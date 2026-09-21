import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # the registered dataset

# ---------------- Data cleaning ----------------
# 1. Remove unnecessary columns: "Unnamed: 0" is a leftover row index from the CSV export
#    and "CustomerID" is a unique identifier. Neither is a predictive feature.
df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore", inplace=True)

# 2. Fix inconsistent category labels ("Fe Male" is a typo for "Female")
df["Gender"] = df["Gender"].str.strip().replace({"Fe Male": "Female"})

# 3. Report missing values (this dataset has none, so no imputation is required)
print("Total missing values:", int(df.isna().sum().sum()))


target = "ProdTaken"  # column to predict: 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # keep the ProdTaken class ratio the same in both splits
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
