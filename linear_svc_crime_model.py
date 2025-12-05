# ============================================
# SAN FRANCISCO CRIME — LINEAR SVC OPTIMIZED
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, log_loss, classification_report
from scipy.special import softmax

# ============================================
# Load Data
# ============================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ============================================
# Feature Engineering
# Extract time features
# ============================================
def process_dates(df):
    df["Dates"] = pd.to_datetime(df["Dates"])
    df["Year"]  = df["Dates"].dt.year
    df["Month"] = df["Dates"].dt.month
    df["Day"]   = df["Dates"].dt.day
    df["Hour"]  = df["Dates"].dt.hour
    return df

train = process_dates(train)
test = process_dates(test)

# Create TimeOfDay bins
hour_bins = [-1, 6, 12, 18, 23]
hour_labels = ['Night', 'Morning', 'Afternoon', 'Evening']

for df in [train, test]:
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=hour_bins, labels=hour_labels)

# Create Day_Area interaction (DayOfWeek + AreaType)
downtown_districts = ['SOUTHERN', 'CENTRAL', 'TENDERLOIN', 'MISSION', 'NORTHERN']

for df in [train, test]:
    df['AreaType'] = df['PdDistrict'].apply(lambda x: 'downtown' if x in downtown_districts else 'suburban')
    df['Day_Area_Interaction'] = df['DayOfWeek'] + '_' + df['AreaType']

# ============================================
# Select Features
# Only the optimized subset
# ============================================
numerical_cols = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']

def encode_features(df, numerical_cols):
    numerical = df[numerical_cols]
    district_dummies = pd.get_dummies(df['PdDistrict'], prefix='District')
    time_of_day_dummies = pd.get_dummies(df['TimeOfDay'], prefix='Time')
    day_area_dummies = pd.get_dummies(df['Day_Area_Interaction'], prefix='DayArea')
    return pd.concat([numerical, district_dummies, time_of_day_dummies, day_area_dummies], axis=1)

X = encode_features(train, numerical_cols)
X_test_final = encode_features(test, numerical_cols)

# ============================================
# Encode target
# ============================================
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train['Category'])

# ============================================
# Train/Test Split
# ============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ============================================
# Scale Data (LinearSVC requires scaling)
# ============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_final = scaler.transform(X_test_final)

# ============================================
# Train LinearSVC
# ============================================
linear_svc = LinearSVC(max_iter=5000, random_state=42)
linear_svc.fit(X_train, y_train)

# ============================================
# Evaluate Model
# ============================================
y_val_pred = linear_svc.predict(X_val)

# F1-score
f1 = f1_score(y_val, y_val_pred, average='weighted')

# RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Log-loss (approximation using decision function + softmax)
y_val_scores = linear_svc.decision_function(X_val)
y_val_proba = softmax(y_val_scores, axis=1)
logloss = log_loss(y_val, y_val_proba)

print("\n===== LINEAR SVC METRICS (Optimized Features) =====")
print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"Log-Loss : {logloss:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_val, y_val_pred, target_names=target_encoder.classes_))

# ============================================
# Generate Submission
# ============================================
final_test_predictions = linear_svc.predict(X_test_final)
final_test_predictions = target_encoder.inverse_transform(final_test_predictions)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Category": final_test_predictions
})

submission.to_csv("submission_linear_svc_optimized.csv", index=False)
print("\nSubmission saved to submission_linear_svc_optimized.csv")