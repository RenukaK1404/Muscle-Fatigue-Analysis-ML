# ==========================================================
# PERSONALIZED MUSCLE FATIGUE & EXERCISE OPTIMIZATION MODEL
# FINAL RESEARCH GRADE VERSION
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from scipy.stats import ttest_ind, f_oneway

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(8,6)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

data = pd.read_excel("rawdata.xlsx")

print("Dataset Loaded")
print(data.head())
print("\nInitial Shape:", data.shape)

# ----------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------

force_cols = ["RFD","RFND","RHD","RHND"]

data = data.dropna(subset=force_cols)

data["Gender_numeric"] = data["Gender (M/F)"].replace({
"M":1,
"F":0,
1:1,
2:0
})

data["Gender_label"] = data["Gender_numeric"].map({
1:"Male",
0:"Female"
})

data["Packyears"] = data["Packyears"].fillna(0)
data["Activitylevel work (1-4)"] = data["Activitylevel work (1-4)"].fillna(2)
data["Activitylevel sparetime (1-4)"] = data["Activitylevel sparetime (1-4)"].fillna(2)

# ----------------------------------------------------------
# COMPUTE MVC
# ----------------------------------------------------------

data["MVC"] = data[force_cols].mean(axis=1)

print("\nMVC Summary Statistics")
print(data["MVC"].describe())

# ----------------------------------------------------------
# REMOVE OUTLIERS
# ----------------------------------------------------------

q1 = data["MVC"].quantile(0.25)
q3 = data["MVC"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

data = data[(data["MVC"] > lower) & (data["MVC"] < upper)]

print("\nAfter Outlier Removal:", data.shape)

# ----------------------------------------------------------
# AGE GROUPS
# ----------------------------------------------------------

def age_group(age):

    if age < 40:
        return "Young"
    elif age <= 60:
        return "Middle"
    else:
        return "Older"

data["Age_Group"] = data["Age (years)"].apply(age_group)

# ----------------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------------

plt.figure(figsize=(10,8))

sns.heatmap(
data.corr(numeric_only=True),
annot=True,
cmap="coolwarm",
fmt=".2f"
)

plt.title("Correlation Matrix of Physiological Variables")
plt.show()

# ----------------------------------------------------------
# MVC DISTRIBUTION
# ----------------------------------------------------------

plt.figure()

sns.histplot(data["MVC"],kde=True,bins=30)

plt.title("Distribution of Maximum Voluntary Contraction")
plt.xlabel("MVC")
plt.ylabel("Frequency")

plt.show()

# ----------------------------------------------------------
# AGE VS MVC
# ----------------------------------------------------------

plt.figure()

sns.scatterplot(
x="Age (years)",
y="MVC",
data=data,
alpha=0.6
)

sns.regplot(
x="Age (years)",
y="MVC",
data=data,
scatter=False,
color="red"
)

plt.title("Age vs Muscle Strength")
plt.xlabel("Age")
plt.ylabel("MVC")

plt.show()

# ----------------------------------------------------------
# MVC BY AGE GROUP
# ----------------------------------------------------------

plt.figure()

sns.boxplot(
x="Age_Group",
y="MVC",
data=data
)

sns.swarmplot(
x="Age_Group",
y="MVC",
data=data,
color="black",
size=3
)

plt.title("Muscle Strength Across Age Groups")

plt.show()

# ----------------------------------------------------------
# MVC BY GENDER
# ----------------------------------------------------------

plt.figure()

sns.boxplot(
x="Gender_label",
y="MVC",
data=data
)

sns.swarmplot(
x="Gender_label",
y="MVC",
data=data,
color="black",
size=3
)

plt.title("Muscle Strength by Gender")

plt.show()

# ----------------------------------------------------------
# ACTIVITY VS MVC
# ----------------------------------------------------------

plt.figure()

sns.boxplot(
x="Activitylevel work (1-4)",
y="MVC",
data=data
)

sns.swarmplot(
x="Activitylevel work (1-4)",
y="MVC",
data=data,
color="black",
size=3
)

plt.title("Effect of Work Activity on Muscle Strength")

plt.show()

# ----------------------------------------------------------
# SPARE TIME ACTIVITY VS MVC
# ----------------------------------------------------------

plt.figure()

sns.boxplot(
x="Activitylevel sparetime (1-4)",
y="MVC",
data=data
)

sns.swarmplot(
x="Activitylevel sparetime (1-4)",
y="MVC",
data=data,
color="black",
size=3
)

plt.title("Effect of Recreational Activity on Muscle Strength")

plt.show()

# ----------------------------------------------------------
# STATISTICAL TESTS
# ----------------------------------------------------------

young = data[data["Age_Group"]=="Young"]["MVC"]
middle = data[data["Age_Group"]=="Middle"]["MVC"]
old = data[data["Age_Group"]=="Older"]["MVC"]

anova = f_oneway(young,middle,old)

print("\nANOVA Test Across Age Groups")
print("F Statistic:",anova.statistic)
print("P Value:",anova.pvalue)

t,p = ttest_ind(young,old)

print("\nYoung vs Older T Test")
print("T statistic:",t)
print("P value:",p)

# ----------------------------------------------------------
# MACHINE LEARNING MODEL
# ----------------------------------------------------------

features = [
"Age (years)",
"Activitylevel work (1-4)",
"Activitylevel sparetime (1-4)",
"Packyears",
"Gender_numeric"
]

ml_data = data[features + ["MVC"]].dropna()

X = ml_data[features]
y = ml_data["MVC"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
X_scaled,y,test_size=0.2,random_state=42
)

rf = RandomForestRegressor(
n_estimators=600,
max_depth=10,
random_state=42
)

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

print("\nRandom Forest R2:",r2_score(y_test,pred_rf))

# ----------------------------------------------------------
# CROSS VALIDATION
# ----------------------------------------------------------

rkf = RepeatedKFold(n_splits=5,n_repeats=5,random_state=42)

cv_scores = cross_val_score(rf,X_scaled,y,cv=rkf,scoring="r2")

print("\nCross Validation Mean Score:",np.mean(cv_scores))

# ----------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------

importance = rf.feature_importances_

pd.Series(importance,index=features).sort_values().plot(kind="barh")

plt.title("Feature Importance for MVC Prediction")
plt.show()

# ----------------------------------------------------------
# FATIGUE MODEL
# ----------------------------------------------------------

time = np.linspace(0,120,300)

def fatigue_capacity(t,alpha):
    return np.exp(-alpha*t)

# ----------------------------------------------------------
# PERSONALIZED FATIGUE COEFFICIENT
# ----------------------------------------------------------

def compute_alpha(row):

    age = row["Age (years)"]
    activity = row["Activitylevel work (1-4)"]
    smoking = row["Packyears"]

    alpha = 0.006
    alpha += 0.00007 * age
    alpha -= 0.0006 * activity
    alpha += 0.00015 * smoking

    return max(alpha,0.002)

data["alpha_personal"] = data.apply(compute_alpha,axis=1)

# ----------------------------------------------------------
# SAFE EXERCISE TIME
# ----------------------------------------------------------

threshold = 0.5

data["safe_time_seconds"] = np.log(threshold)/(-data["alpha_personal"])

print("\nSafe Exercise Time Summary")
print(data["safe_time_seconds"].describe())

# ----------------------------------------------------------
# STRENGTH VS ENDURANCE
# ----------------------------------------------------------

plt.figure()

sns.scatterplot(
x="MVC",
y="safe_time_seconds",
hue="Age_Group",
data=data
)

plt.title("Strength vs Endurance Relationship")
plt.xlabel("MVC")
plt.ylabel("Safe Exercise Time")

plt.show()

# ----------------------------------------------------------
# USER PERSONALIZED PREDICTION
# ----------------------------------------------------------

print("\nENTER SUBJECT DETAILS FOR PERSONALIZED PREDICTION")

age=float(input("Age: "))
gender=input("Gender (M/F): ")

gender_num=1 if gender.upper()=="M" else 0

activity_work=float(input("Work Activity (1-4): "))
activity_spare=float(input("Spare Activity (1-4): "))
packyears=float(input("Smoking Packyears: "))

user=pd.DataFrame([{
"Age (years)":age,
"Activitylevel work (1-4)":activity_work,
"Activitylevel sparetime (1-4)":activity_spare,
"Packyears":packyears,
"Gender_numeric":gender_num
}])

user_scaled=scaler.transform(user)

predicted_mvc=rf.predict(user_scaled)[0]

alpha_user=0.006+0.00007*age-0.0006*activity_work+0.00015*packyears
alpha_user=max(alpha_user,0.002)

safe_time=np.log(threshold)/(-alpha_user)

contraction=safe_time*0.8
rest=contraction*0.5

risk_index=predicted_mvc/alpha_user

if risk_index<100000:
    risk="High Risk"
elif risk_index<200000:
    risk="Moderate Risk"
else:
    risk="Low Risk"

print("\n----- PERSONALIZED RESULTS -----")

print("Predicted MVC:",predicted_mvc)
print("Fatigue Coefficient:",alpha_user)
print("Safe Exercise Time:",safe_time)
print("Recommended Contraction:",contraction)
print("Recommended Rest:",rest)
print("Fatigue Risk:",risk)

print("\nPROJECT COMPLETE")