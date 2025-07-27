import pandas as pd
import numpy as np

np.random.seed(42)

# Developer profiles
dev_profiles = pd.DataFrame({
    "user_id": ["U01", "U02", "U03", "U04", "U05", "U02"],
    "name": ["Aafia", "Aaryaan", "Jazib", "Zoya", None, "Aaryaan"],
    "age": [23, 21, 24, None, 22, 21],
    "grade": ["A", "B", "C", "A", "B", "B"]
})

# Daily logs
logs = pd.DataFrame({
    "user_id": ["U01", "U02", "U03", "U01", "U05", "U04", "U06"],
    "hours_coded": [5, None, 2, 6, 3, 4, 8],
    "hours_slept": [7, 6, 5, 8, None, 4, 6],
    "bugs_fixed": [12, 7, 5, 14, 9, 3, 11],
    "log_date": pd.date_range("2024-01-01", periods=7)
})

print(f"\n first 5 rows of dev profiles: {dev_profiles.head()} ")
print(f"\n first 5 rows of logs: {logs.head()} ")

print(f"\n last 5 rows of dev profiles: {dev_profiles.tail()} ")
print(f"\n last 5 rows of logs: {logs.tail()} ")

dev_profiles = dev_profiles.dropna(subset="name")
dev_profiles = dev_profiles.dropna(subset="user_id")

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

logs["statues"] = logs["bugs_fixed"].apply(lambda x: "pass" if x >= 10 else "fail")

logs["sleep_band"] = pd.cut(logs["hours_slept"], bins=[3, 5, 7, 9], labels=["low", "mid", "high"])
logs["bug_level"] = pd.qcut(logs["bugs_fixed"], q=3, labels=["low", "mid", "high"])

grading = {"A": 4,
           "B": 3,
           "C": 2}

dev_profiles["grade_num"] = dev_profiles["grade"].map(grading)

merge = merge.set_index("name")

print(f"\n{merge.loc['Aaryaan']}")
print(f"\n{merge.iloc[1]}")

filter =  (logs["hours_slept"] < 6) & (logs["bugs_fixed"] > 10) 
print(f"\n{logs[filter]}")

mean_bugs = merge.groupby("grade")["bugs_fixed"].mean()
total_bugs = logs.groupby("sleep_band")["bugs_fixed"].sum()
merge["bugs_per_grade"] = merge.groupby("grade")["bugs_fixed"].transform("median")


pivot  = pd.pivot_table(logs, index="user_id", columns="sleep_band", values="hours_coded", aggfunc="mean")
print(f"\n{pivot}")

melt = pivot.reset_index().melt(id_vars="user_id",value_name="hours_coded")
dummies = pd.get_dummies(logs["sleep_band"])
category_ = logs["sleep_band"].astype("category").cat.codes

print(F"\n{category_}")

logs["rolling"] = logs["bugs_fixed"].rolling(window=3).mean()
logs["exp_avg"] = logs["bugs_fixed"].ewm(alpha=0.2).mean()

X = logs[["hours_coded","hours_slept"]]
y = logs["bugs_fixed"]

X.to_csv("Xdata.csv", index=False)
y.to_csv("Ydata.csv" ,index=False)
print(f"\n{logs}")
print(f"\n {merge}")

