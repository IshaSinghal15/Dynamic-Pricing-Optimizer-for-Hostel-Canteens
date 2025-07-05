import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("synthetic_sales.csv")

le_item = LabelEncoder()
le_slot = LabelEncoder()
df["item_encoded"] = le_item.fit_transform(df["item"])
df["slot_encoded"] = le_slot.fit_transform(df["time_slot"])

features = ["price", "item_encoded", "slot_encoded", "is_event_day"]
X = df[features]
y = df["quantity_sold"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "demand_model.pkl")
joblib.dump(le_item, "label_encoder_item.pkl")
joblib.dump(le_slot, "label_encoder_slot.pkl")
