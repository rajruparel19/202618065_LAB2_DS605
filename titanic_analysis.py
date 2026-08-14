import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

FIG_DIR = "figures"
DATA_DIR = "data"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved figure -> {path}")

np.random.seed(42)

A = np.random.randint(1, 101, size=100)
print("Array A (first 10 values):", A[:10])
print("Array A shape:", A.shape)

A_min = np.min(A)
A_max = np.max(A)
A_median = np.median(A)
A_mean = np.mean(A)
A_std = np.std(A)

print(f"Min    : {A_min}")
print(f"Max    : {A_max}")
print(f"Median : {A_median}")
print(f"Mean   : {A_mean:.4f}")
print(f"Std    : {A_std:.4f}")

B = np.arange(0, 100)  # 0,1,2,...,99 -> exactly 100 values
print("Array B:", B)
print("Array B shape:", B.shape, "| length:", len(B))

zeros_arr = np.zeros((4, 5))
ones_arr = np.ones((4, 5))

print("zeros_arr:\n", zeros_arr)
print("Shape:", zeros_arr.shape, "| dtype:", zeros_arr.dtype)
print()
print("ones_arr:\n", ones_arr)
print("Shape:", ones_arr.shape, "| dtype:", ones_arr.dtype)

lin_arr = np.linspace(0, 10, 5) 
print("linspace(0, 10, 5):", lin_arr)

arange_arr = np.arange(0, 10, 2)  
print("arange(0, 10, 2):  ", arange_arr)

arr_2d = np.arange(1, 13).reshape(3, 4)
print("2D array:\n", arr_2d)
print("Shape:", arr_2d.shape, "| ndim:", arr_2d.ndim)

print("\nElement at row 1, col 2:", arr_2d[1, 2])
print("Row 0:", arr_2d[0, :])
print("Column 1:", arr_2d[:, 1])
print("Slice (rows 0-1, cols 1-3):\n", arr_2d[0:2, 1:3])

arr_3d = np.arange(1, 25).reshape(2, 3, 4)
print("3D array:\n", arr_3d)
print("Shape:", arr_3d.shape, "| ndim:", arr_3d.ndim)

print("\nFirst 'block' (index 0):\n", arr_3d[0])
print("\nElement at [1, 2, 3]:", arr_3d[1, 2, 3])
print("\nRow 1 of block 0:", arr_3d[0, 1, :])
print("Column 2 of block 0:", arr_3d[0, :, 2])
print("\nSlice (block 0, rows 0-1, cols 0-2):\n", arr_3d[0, 0:2, 0:2])

flat = np.arange(1, 21)
matrix = flat.reshape(4, 5)
print("Original 1D array:", flat)
print("\nReshaped to 4x5 matrix:\n", matrix)

flattened_back = matrix.flatten()
print("\nFlattened back to 1D:", flattened_back)
print("Matches original:", np.array_equal(flat, flattened_back))

M1 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 10]])

M2 = np.array([[9, 8, 7],
               [6, 5, 4],
               [3, 2, 1]])

print("M1:\n", M1)
print("\nM2:\n", M2)

addition = M1 + M2
print("\nAddition (M1 + M2):\n", addition)

elementwise_mult = M1 * M2
print("\nElement-wise multiplication (M1 * M2):\n", elementwise_mult)

matmul_result = M1 @ M2          
print("\nMatrix multiplication (M1 @ M2):\n", matmul_result)

square = M1  

transpose = square.T
determinant = np.linalg.det(square)

print("Square matrix:\n", square)
print("\nTranspose:\n", transpose)
print(f"\nDeterminant: {determinant:.4f}")

if not np.isclose(determinant, 0):
    inverse = np.linalg.inv(square)
    identity_check = square @ inverse
    print("Inverse:\n", inverse)
    print("\nsquare @ inverse:\n", identity_check)
    print("\nIs square @ inverse ≈ Identity matrix?",
          np.allclose(identity_check, np.eye(square.shape[0])))
else:
    print("Matrix is singular (determinant ≈ 0); no inverse exists.")

chosen_mean, chosen_std = 50, 10
n_samples = 5000

np.random.seed(42)
normal_data = np.random.normal(loc=chosen_mean, scale=chosen_std, size=n_samples)

print(f"Chosen population mean (mu): {chosen_mean}")
print(f"Chosen population std (sigma): {chosen_std}")
print(f"Number of samples: {n_samples}")

sample_mean = np.mean(normal_data)
sample_std = np.std(normal_data, ddof=1)  # sample std (Bessel's correction)

print(f"Sample mean: {sample_mean:.4f}  (chosen mean: {chosen_mean})")
print(f"Sample std : {sample_std:.4f}  (chosen std: {chosen_std})")
print(f"\nDifference in mean: {abs(sample_mean - chosen_mean):.4f}")
print(f"Difference in std : {abs(sample_std - chosen_std):.4f}")
print("\nThe sample statistics are very close to the chosen population "
      "parameters, as expected for a large (n=5000) random sample "
      "(law of large numbers).")

plt.figure(figsize=(8, 5))
plt.hist(normal_data, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
plt.axvline(sample_mean, color="#C44E52", linestyle="--", linewidth=2,
            label=f"Sample mean = {sample_mean:.2f}")
plt.title(f"Histogram of {n_samples} Normally Distributed Values (mu={chosen_mean}, sigma={chosen_std})")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
savefig("task3_normal_distribution_histogram.png")
plt.show()
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print("Data loaded. Shape:", df.shape)
df.head()
df.tail()
print("Shape (rows, columns):", df.shape)
print("Columns:", list(df.columns))

df.info()
df.describe(include="all")
loc_example = df.loc[0:4, ["Name", "Age", "Survived"]]
print("loc example (rows 0-4 inclusive, by label, selected columns):")
print(loc_example)
print()

iloc_example = df.iloc[0:5, 0:4]
print("iloc example (first 5 rows, first 4 columns, by integer position):")
print(iloc_example)
male_over_50 = df[(df["Sex"] == "male") & (df["Age"] > 50)]
print(f"Male passengers older than 50: {len(male_over_50)}")

female_first_class = df[(df["Sex"] == "female") & (df["Pclass"] == 1)]
pct_survived = female_first_class["Survived"].mean() * 100
print(f"Female first-class passengers: {len(female_first_class)}")
print(f"Percentage survived: {pct_survived:.2f}%")
overall_median_fare = df["Fare"].median()
cond3 = df[(df["Age"].between(20, 40)) & (df["Fare"] > overall_median_fare) & (df["Survived"] == 1)]
print(f"Overall median fare: {overall_median_fare:.4f}")
print(f"Age 20-40, Fare above median, survived: {len(cond3)}")

alone_young_died = df[(df["SibSp"] == 0) & (df["Parch"] == 0) & (df["Age"] < 30) & (df["Survived"] == 0)]
print(f"Traveling alone, age < 30, did not survive: {len(alone_young_died)}")

southampton_median_fare = df.loc[df["Embarked"] == "S", "Fare"].median()
cond5 = df[(df["Embarked"] == "S") & (df["Pclass"].isin([2, 3])) & (df["Fare"] > southampton_median_fare)]
print(f"Southampton median fare: {southampton_median_fare:.4f}")
print(f"Embarked=S, Pclass 2 or 3, Fare above Southampton median: {len(cond5)}")

survival_by_sex = df.groupby("Sex")["Survived"].mean()
print("Survival rate by Sex:")
print(survival_by_sex)

survival_by_pclass = df.groupby("Pclass")["Survived"].mean()
print("Survival rate by Pclass:")
print(survival_by_pclass)
age_fare_by_pclass = df.groupby("Pclass")[["Age", "Fare"]].mean()
print("Average Age and Fare by Pclass:")
print(age_fare_by_pclass)
sex_pclass_stats = df.groupby(["Sex", "Pclass"])["Survived"].agg(["count", "mean"])
sex_pclass_stats.columns = ["passenger_count", "survival_rate"]
print("Passenger count and survival rate by Sex-Pclass:")
print(sex_pclass_stats)
embarked_stats = df.groupby("Embarked").agg(
    passenger_count=("PassengerId", "count"),
    avg_fare=("Fare", "mean"),
    survival_rate=("Survived", "mean"),
)
print("Passenger count, average Fare, and survival rate by Embarked:")
print(embarked_stats)

missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
print(missing_summary)

plt.figure(figsize=(8, 5))
sns.barplot(x=missing_summary.index, y=missing_summary["missing_count"], color="#DD8452")
plt.title("Missing Values per Column")
plt.xlabel("Column")
plt.ylabel("Missing Count")
for i, v in enumerate(missing_summary["missing_count"]):
    plt.text(i, v + 5, str(int(v)), ha="center")
savefig("task7_missing_values_bar_chart.png")
plt.show()

print("Missing Age count BEFORE fill:", df["Age"].isnull().sum())

age_mean_value = df["Age"].mean()
df["Age"] = df["Age"].fillna(age_mean_value)

print("Missing Age count AFTER fill:", df["Age"].isnull().sum())
print(f"Filled with mean Age = {age_mean_value:.4f}")

raw_age = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))["Age"]

age_mean_imputed = raw_age.fillna(raw_age.mean())
age_median_imputed = raw_age.fillna(raw_age.median())
age_mode_imputed = raw_age.fillna(raw_age.mode()[0])

np.random.seed(42)
non_null_values = raw_age.dropna().values
random_fill = pd.Series(
    np.random.choice(non_null_values, size=raw_age.isnull().sum()),
    index=raw_age[raw_age.isnull()].index,
)
age_random_imputed = raw_age.copy()
age_random_imputed[age_random_imputed.isnull()] = random_fill

imputation_comparison = pd.DataFrame({
    "original_missing": raw_age.isnull().sum(),
    "mean_imputed_mean": age_mean_imputed.mean(),
    "median_imputed_mean": age_median_imputed.mean(),
    "mode_imputed_mean": age_mode_imputed.mean(),
    "random_imputed_mean": age_random_imputed.mean(),
}, index=["Age"])
print("Original missing Age values:", raw_age.isnull().sum())
print("\nComparison of resulting column mean under each imputation strategy:")
print(imputation_comparison)

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

fare_outliers = df[(df["Fare"] < lower_bound) | (df["Fare"] > upper_bound)]

print(f"Q1 (25th percentile): {Q1:.4f}")
print(f"Q3 (75th percentile): {Q3:.4f}")
print(f"IQR: {IQR:.4f}")
print(f"Lower bound: {lower_bound:.4f}")
print(f"Upper bound: {upper_bound:.4f}")
print(f"Number of Fare outliers: {len(fare_outliers)}")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df[["SibSp", "Parch", "FamilySize", "IsAlone"]].head(10)

pivot_survival = pd.pivot_table(df, index="Sex", columns="Pclass", values="Survived", aggfunc="mean")
print("Pivot table - mean Survived by Sex x Pclass:")
print(pivot_survival.round(4))

stacked = pivot_survival.stack()
highest = stacked.idxmax()
lowest = stacked.idxmin()
print(f"\nHighest survival group: Sex={highest[0]}, Pclass={highest[1]} -> {stacked.max():.4f}")
print(f"Lowest survival group : Sex={lowest[0]}, Pclass={lowest[1]} -> {stacked.min():.4f}")

numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=0.5)
plt.title("Correlation Heatmap - Numerical Titanic Features")
savefig("task9_correlation_heatmap.png")
plt.show()

trivial_pairs = {
    frozenset(p) for p in [
        ("SibSp", "FamilySize"), ("Parch", "FamilySize"),
        ("SibSp", "IsAlone"), ("Parch", "IsAlone"),
        ("FamilySize", "IsAlone"),
    ]
}

corr_pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
corr_pairs = corr_pairs[[frozenset(idx) not in trivial_pairs for idx in corr_pairs.index]]

strongest_positive = corr_pairs.idxmax()
strongest_negative = corr_pairs.idxmin()
print(f"Strongest positive relationship: {strongest_positive} -> r = {corr_pairs.max():.4f}")
print(f"Strongest negative relationship: {strongest_negative} -> r = {corr_pairs.min():.4f}")

print("\nFull correlation matrix:")
print(corr.round(3))

plt.figure(figsize=(6, 5))
survival_by_sex_plot = df.groupby("Sex")["Survived"].mean()
ax = sns.barplot(x=survival_by_sex_plot.index, y=survival_by_sex_plot.values,
                  hue=survival_by_sex_plot.index, palette=["#C44E52", "#4C72B0"], legend=False)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
for i, v in enumerate(survival_by_sex_plot.values):
    plt.text(i, v + 0.02, f"{v:.2%}", ha="center")
savefig("task9_survival_rate_by_sex.png")
plt.show()

print(f"Male survival rate: {survival_by_sex_plot['male']:.2%}")
print(f"Female survival rate: {survival_by_sex_plot['female']:.2%}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived",
                 palette={0: "#C44E52", 1: "#4C72B0"}, alpha=0.7)
plt.title("Age vs Fare by Survival Status")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived", labels=["No (0)", "Yes (1)"])
savefig("task9_age_vs_fare_by_survival.png")
plt.show()

cleaned_path = os.path.join(DATA_DIR, "train_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved -> {cleaned_path}")
print("Final shape:", df.shape)
df.head()