import pandas as pd
import joblib
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

df = pd.read_csv("synthetic_sales.csv")
model = joblib.load("demand_model.pkl")
le_item = joblib.load("label_encoder_item.pkl")
le_slot = joblib.load("label_encoder_slot.pkl")

sample_day = "2024-09-15"
sample_slot = "Lunch"
day_data = df[(df["date"] == sample_day) & (df["time_slot"] == sample_slot)]

price_options = {}
demand_predictions = {}

for idx, row in day_data.iterrows():
    item = row["item"]
    base_price = row["price"]
    item_enc = le_item.transform([item])[0]
    slot_enc = le_slot.transform([sample_slot])[0]
    is_event = row["is_event_day"]

    prices = np.linspace(base_price * 0.8, base_price * 1.2, 5).round(2)
    price_options[item] = prices
    demand_predictions[item] = [model.predict([[p, item_enc, slot_enc, is_event]])[0] for p in prices]

model_lp = LpProblem("Canteen-Pricing-Optimization", LpMaximize)
price_vars = {item: LpVariable.dicts(item, range(5), cat="Binary") for item in price_options}

for item in price_vars:
    model_lp += lpSum(price_vars[item][i] for i in range(5)) == 1

model_lp += lpSum(
    price_vars[item][i] * price_options[item][i] * demand_predictions[item][i]
    for item in price_vars
    for i in range(5)
)

model_lp.solve()

optimized_prices = {}
total_revenue = 0
for item in price_vars:
    for i in range(5):
        if price_vars[item][i].varValue == 1:
            price = price_options[item][i]
            demand = demand_predictions[item][i]
            optimized_prices[item] = (price, round(demand))
            total_revenue += price * demand

print("Optimized Prices:", optimized_prices)
print("Total Revenue:", round(total_revenue, 2))
