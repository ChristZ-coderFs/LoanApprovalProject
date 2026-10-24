import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_approval_dataset.csv", sep=";")
df.columns = df.columns.str.strip()
print(df.head(20).to_string(index=False))


df["loan_status"] = df["loan_status"].str.strip()
df["loan_status_encoded"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

numerical_data = df.select_dtypes(include="number")
numerical_data = numerical_data.drop(columns=["loan_id"])

corrmatrix = numerical_data.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    corrmatrix,
    annot=True,
    fmt=".2f",
    cmap="PiYG",
    vmin=-1,
    vmax=1
)

plt.tight_layout()
df["education"] = df["education"].str.strip()

graduates = df[df["education"] == "Graduate"]
non_graduates = df[df["education"] == "Not Graduate"]

print("\nGraduate:")
print("Approved:", graduates[graduates["loan_status"] == "Approved"].shape[0])
print("Rejected:", graduates[graduates["loan_status"] == "Rejected"].shape[0])
print("\nNot Graduate:")
print("Approved:", non_graduates[non_graduates["loan_status"] == "Approved"].shape[0])
print("Rejected:", non_graduates[non_graduates["loan_status"] == "Rejected"].shape[0])
plt.show()