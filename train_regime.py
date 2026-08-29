import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("joined_dataset.csv")
df = df[df["regime"] != "unknown"]

df["valid_date"] = pd.to_datetime(df["valid_date"])
df["month"] = df["valid_date"].dt.month
df["doy"] = df["valid_date"].dt.dayofyear

features = ["forecast_precip_mm", "forecast_temp_c", "month", "doy"]
X = df[features]
y = df["regime"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(clf, "regime_classifier.pkl")