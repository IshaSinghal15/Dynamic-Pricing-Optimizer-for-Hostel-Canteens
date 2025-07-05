import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

items = [
    {"item": "Veg Biryani", "base_price": 50},
    {"item": "Paneer Butter Masala", "base_price": 60},
    {"item": "Poori Bhaji", "base_price": 30},
    {"item": "Masala Dosa", "base_price": 40},
    {"item": "Egg Curry", "base_price": 45},
    {"item": "Fried Rice", "base_price": 35},
    {"item": "Chole Bhature", "base_price": 35},
    {"item": "Dal Makhani", "base_price": 40},
    {"item": "Chicken Curry", "base_price": 70},
    {"item": "Aloo Paratha", "base_price": 25}
]

time_slots = ["Breakfast", "Lunch", "Snacks", "Dinner"]
start_date = datetime.strptime("2024-07-01", "%Y-%m-%d")
days = 180

data = []

for day in range(days):
    current_date = start_date + timedelta(days=day)
    is_event = np.random.binomial(1, 0.1)
    for slot in time_slots:
        for item in items:
            base_price = item["base_price"]
            price_variation = np.random.normal(0, 3)
            price = max(10, round(base_price + price_variation, 2))

            demand_mean = random.randint(30, 100)
            if is_event:
                demand_mean *= 1.3
            quantity_sold = int(np.random.poisson(lam=demand_mean * (base_price / price)))

            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "time_slot": slot,
                "item": item["item"],
                "price": price,
                "quantity_sold": quantity_sold,
                "is_event_day": is_event
            })

df = pd.DataFrame(data)
df.to_csv("synthetic_sales.csv", index=False)
