import pandas as pd
import numpy as np

np.random.seed(42)

# Developer profiles
dev_profiles = pd.DataFrame({
    "user_id": ["U01", "U02", "U03", "U04", "U05"],
    "name": ["Zoya", "Tariq", "Aaryaan", "Aafia", "Zain"],
    "age": [22, None, 20, 23, 21],
    "grade": ["A", "B", "C", "A", "B"]
})

# Daily logs
logs = pd.DataFrame({
    "user_id": ["U01", "U02", "U01", "U03", "U05", "U04", "U03", "U06"],
    "hours_coded": [5, 3, None, 2, 4, 8, 1, 6],
    "hours_slept": [7, 6, 8, np.nan, 6, 5, 7, 4],
    "bugs_fixed": [12, 7, 14, 5, 9, 20, 3, 16],
    "log_date": pd.date_range("2024-01-01", periods=8, freq="D")
})

print(f"\n first three rows of dev_profile: {dev_profiles.head(3)}")
print(f"\n last three rows of dev_profile: {dev_profiles.tail(3)}")
print(f"\n first three rows of logs {logs.head(3)}")
print(f"\n last three rows of dev_profile: {logs.tail(3)}")

print(f"\n information of dev_profile: {dev_profiles.info()}")
print(f"\n information of logs: {logs.info()}")

print(f"\n summary stats of developer profiles: {dev_profiles.describe()}")
print(f"\n summary stats of logs: {logs.describe()}")

print(f"\n the shape of developer profile: {dev_profiles.shape}")
print(f"\n the shape of logs: {logs.shape}")

print(f"\n columns info of developer profile: {dev_profiles.columns}")
print(f"\n columns info of logs: {logs.columns}")

print(f"\n the data types of developer profile: {dev_profiles.dtypes}")
print(f"\n the data types info of logs: {logs.dtypes}")

print(f"\n the index of developer profile: {dev_profiles.index}")
print(f"\n the index info of logs: {logs.index}")

dev_profiles = dev_profiles.dropna(subset="user_id")
dev_profiles = dev_profiles.dropna(subset="name")
logs = logs.dropna(subset="user_id")

dev_profiles["age"] = dev_profiles["age"].fillna(dev_profiles["age"].mean())

logs["hours_coded"] = logs["hours_coded"].fillna(logs["hours_coded"].mean())
logs["hours_slept"] = logs["hours_slept"].fillna(logs["hours_slept"].mean())

dev_profiles["grade"] = dev_profiles["grade"].astype("category")
logs["log_date"] = pd.to_datetime(logs["log_date"])

dev_profiles = dev_profiles.drop_duplicates()
logs = logs.drop_duplicates()

merge = pd.merge(dev_profiles,logs, on="user_id", how="left")
print(f"\n {merge}")

logs["productivity"] = logs["bugs_fixed"] / logs["hours_coded"]

logs["status"] = logs["bugs_fixed"].apply(lambda x: "pass" if x >= 10 else "fail")

logs["sleep_band"] = pd.cut(logs["hours_slept"], bins=[4,6,7,8], labels=["low","mid","high"])
logs["bug_q"] = pd.qcut(logs["bugs_fixed"], q=3, labels=["low","mid","high"])

grading = {"A": 4,
           "B": 3,
           "C": 2}

dev_profiles["grade_num"] = dev_profiles["grade"].map(grading)

filtering = (logs["bugs_fixed"] > 10) & (logs["hours_slept"] < 6)
print(f"\n Devs who fixed >10 bugs & slept <6 hrs:{logs[filtering]}")

dev_profiles = dev_profiles.set_index("name")
print(f"\n {dev_profiles.loc["Zoya"]}")
print(f"\n {logs.iloc[2]}")

avg_bugs = merge.groupby("grade")["bugs_fixed"].mean()
total_bugs = logs.groupby("sleep_band").agg({"bugs_fixed" :"sum"})

print(f"\n {avg_bugs}")
print(f"\n{total_bugs}")

merge["bugs_per_grade"] = (
    merge.groupby("grade")["bugs_fixed"].transform("median")
)

pivot = pd.pivot_table(logs, index = "user_id", columns="sleep_band", values="hours_coded", aggfunc = "mean")
print(f"\n{pivot}")

melt = pivot.reset_index().melt(id_vars="user_id",value_name="hours_coded")
print(f"\n{melt}")

dummies = pd.get_dummies(logs["sleep_band"])
print(f"\n{dummies}")

category = logs["sleep_band"].astype("category").cat.codes
print(f"\n{category}")

X = logs[["hours_coded", "productivity"]]
y = logs["bugs_fixed"]

X.to_csv("Xdata.csv", index=False)
y.to_csv("ydata.csv", index=False)

rolling = logs["bugs_fixed"].rolling(window=3).mean()
exp_wei_mean = logs["bugs_fixed"].ewm(alpha=0.2).mean()

print(f"\n{rolling}")
print(f"\n{exp_wei_mean}")
print(f"\n {logs}")

print(f"\n{merge}")
print(f"\n {dev_profiles}")

