# Dynamic Pricing Optimizer for Hostel Canteens

🎯 A self-initiated project that applies Machine Learning and Operations Research to dynamically optimize food prices in hostel canteens — aiming to **maximize revenue** and **reduce food waste**.

---

## 🚀 Features

- 📊 Synthetic sales data generation for 180 days across 10+ menu items
- 🤖 Demand prediction using Random Forest Regressor (ML)
- 📈 Revenue maximization using Linear Programming (PuLP)
- 🔁 Adjustable price simulations across multiple slots
- 🧠 Ready for real-world extension with minimal changes

---

## 🛠 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate 7200-row synthetic sales data
python data_generator.py

# Step 2: Train Random Forest demand forecasting model
python demand_forecast.py

# Step 3: Run price optimization for a chosen slot (e.g. Lunch on 15th Sept)
python price_optimizer.py
