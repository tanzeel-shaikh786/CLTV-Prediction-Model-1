
import pandas as pd
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\Lenovo\Downloads\transactions.csv', encoding='ISO-8859-1')

# Step 2: Drop missing CustomerIDs
df = df.dropna(subset=["CustomerID"]).copy()

# Step 3: Remove canceled orders (InvoiceNo starts with 'C')
df = df[~df["InvoiceNo"].astype(str).str.startswith('C')]

# Step 4: Remove rows with Quantity <= 0 or UnitPrice <= 0
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Step 5: Convert data types
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Step 6: Calculate TotalPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Step 7: Reference date for Recency
reference_date = df["InvoiceDate"].max() + timedelta(days=1)

# Step 8: Group by CustomerID to compute RFM metrics
customer_metrics = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalPrice": "Monetary"
})

# Step 9: Compute AOV
customer_metrics["AOV"] = customer_metrics["Monetary"] / customer_metrics["Frequency"]

# Step 10: Prepare data for modeling
X = customer_metrics[["Recency", "Frequency", "AOV"]]
y = customer_metrics["Monetary"]

# Step 11: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Train XGBoost Regressor
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Step 13: Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Step 14: Predict LTV and Segment
customer_metrics["PredictedLTV"] = model.predict(X)
customer_metrics["Segment"] = pd.qcut(
    customer_metrics["PredictedLTV"], 
    q=4, 
    labels=["Low", "Mid-Low", "Mid-High", "High"],
    duplicates='drop'
)

# ✅ Reset index so 'CustomerID' is a column again
customer_metrics = customer_metrics.reset_index()


# Step 15: Save final output
customer_metrics.to_csv("customer_ltv_segments.csv")

print("\nSample of segmented customer data:")
print(customer_metrics[["Recency", "Frequency", "AOV", "PredictedLTV", "Segment"]].head())

customer_metrics["Segment"] = pd.qcut(customer_metrics["PredictedLTV"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"])
customer_metrics["Segment"] = pd.qcut(
    customer_metrics["PredictedLTV"], 
    q=4, 
    labels=["Low", "Mid-Low", "Mid-High", "High"],
    duplicates='drop'
)

ltv_only = customer_metrics[["Recency", "Frequency", "AOV", "PredictedLTV"]]
ltv_only.to_csv(r"C:\Users\Lenovo\Desktop\ltv_predictions.csv", index=True)

segmented_customers = customer_metrics[["Recency", "Frequency", "AOV", "PredictedLTV", "Segment"]]
segmented_customers.to_csv(r"C:\Users\Lenovo\Desktop\ltv_segments.csv", index=True)

output_path = r"C:\Users\Lenovo\Desktop\final_ltv_predictions.csv"
customer_metrics.to_csv(output_path)

print(f"✅ Final LTV predictions saved to: {output_path}")




#Visualizations

#Feature Importance
#what:Shows which customer behaviors (features) most impact LTV predictions.
# Why:Focus marketing on key behaviors — e.g., if Frequency is most important, encourage repeat purchases.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

# Make sure X and model are already defined
feature_names = X.columns
importances = model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(importances)
sorted_features = feature_names[sorted_idx]
sorted_importances = importances[sorted_idx]

# Create a gradient color map
colors = cm.plasma(np.linspace(0.1, 0.9, len(sorted_features)))

# 🎨 Advanced Horizontal Bar Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_importances, color=colors)

# Add text labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f"{width:.2f}", va='center', fontsize=10, color='black')

plt.title("🎯 Top Features Influencing LTV", fontsize=18, weight='bold', color='darkslateblue')
plt.xlabel("Feature Importance", fontsize=13)
plt.ylabel("Features", fontsize=13)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

input("✅ Press Enter to continue to Figure 2...")



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 7))

# 🎨 Histogram of Predicted LTV with KDE curve
sns.histplot(
    data=customer_metrics,
    x="PredictedLTV",
    bins=50,
    kde=True,
    stat="count",
    edgecolor=None,
    linewidth=0,
    color="mediumvioletred",
    alpha=0.6
)

# KDE curve separately
sns.kdeplot(
    data=customer_metrics["PredictedLTV"],
    color="darkred",
    linewidth=3,
    label="Density Curve"
)

# 🧠 Add vertical lines for percentiles
percentiles = [25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(customer_metrics["PredictedLTV"], p)
    plt.axvline(val, color="teal", linestyle="--", alpha=0.8)
    plt.text(val, plt.ylim()[1]*0.9, f"{p}%", color="teal", fontsize=10, ha="center")

# Final formatting
plt.title("📊 Predicted LTV Distribution with Density Curve", fontsize=18, weight='bold', color='darkslateblue')
plt.xlabel("Predicted Customer LTV", fontsize=13)
plt.ylabel("Number of Customers", fontsize=13)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
input("✅ Press Enter to continue to Figure 3...")







import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🎯 Create colorful segment labels with emojis
customer_metrics["Segment"] = pd.qcut(
    customer_metrics["PredictedLTV"],
    q=4,
    labels=["🧊 Low", "🟡 Mid-Low", "🟠 Mid-High", "🔥 High"]
)

# 💫 Set custom theme
sns.set_theme(style="whitegrid")

# 🎨 Plot with vivid colors
plt.figure(figsize=(9, 6))
bar = sns.countplot(
    x="Segment",
    data=customer_metrics,
    palette=["#b2df8a", "#fdbf6f", "#fb9a99", "#e31a1c"]
)

# 📌 Add annotations on bars
for p in bar.patches:
    count = int(p.get_height())
    bar.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=11, color='black', weight='bold')

# 🎀 Chart styling
plt.title("🎯 Customer Segments Based on Predicted LTV", fontsize=18, weight='bold', color="slateblue")
plt.xlabel("LTV Segment", fontsize=13)
plt.ylabel("Number of Customers", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

input("✅ Press Enter to continue to Figure 4...")


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 💡 Create pivot for heatmap
pivot = customer_metrics.pivot_table(
    index=pd.qcut(customer_metrics["Recency"], 5, duplicates='drop'),
    columns=pd.qcut(customer_metrics["Frequency"], 5, duplicates='drop'),
    values="PredictedLTV",
    aggfunc="mean"
)

# 🎨 Set theme
sns.set_theme(style="white")

# 🌈 Use a bold, perceptually uniform colormap
plt.figure(figsize=(11, 7))
heatmap = sns.heatmap(
    pivot,
    cmap="coolwarm",     # more colorful and appealing
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    linecolor='gray',
    square=True,
    cbar_kws={'label': '💰 Avg Predicted LTV'}
)

# 📌 Title and labels
plt.title("🧭 Heatmap: Recency vs Frequency vs Avg LTV", fontsize=18, weight='bold', color='navy')
plt.xlabel("🔁 Purchase Frequency", fontsize=13, weight='bold')
plt.ylabel("📅 Recency (days ago)", fontsize=13, weight='bold')

# 🎀 Layout and grid
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

input("✅ Press Enter to continue to Figure 5...")






import matplotlib.pyplot as plt
import seaborn as sns

# 🧼 Make sure style is set for clarity
sns.set_theme(style="whitegrid")

# ✅ Prepare data (already done above)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["FirstPurchase"] = df.groupby("CustomerID")["InvoiceDate"].transform("min")
df["CohortMonth"] = df["FirstPurchase"].dt.to_period("M")

df_ltv = df.merge(customer_metrics[["CustomerID", "PredictedLTV"]], on="CustomerID")
cohort_ltv = df_ltv.groupby("CohortMonth")["PredictedLTV"].mean()

# 🖼️ Convert Period to string for plotting
cohort_ltv.index = cohort_ltv.index.astype(str)

# 📈 Plot
plt.figure(figsize=(12, 6))
line = plt.plot(
    cohort_ltv.index, cohort_ltv.values,
    marker='o', markersize=8, linewidth=2.5, color="#1f77b4", label="Avg LTV"
)

# 🔘 Annotate values
for i, value in enumerate(cohort_ltv.values):
    plt.text(i, value + 0.02 * cohort_ltv.max(), f"{value:.0f}", ha='center', va='bottom', fontsize=9, color="black")

# 📝 Titles and labels
plt.title("📆 Average Predicted LTV by Customer Cohort", fontsize=18, weight='bold', color='#003366')
plt.xlabel("🗓️ Signup Cohort (Month)", fontsize=13)
plt.ylabel("💰 Avg Predicted LTV", fontsize=13)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.legend()
plt.show()

input("✅ Press Enter to continue to Figure 6...")



import plotly.express as px
import pandas as pd

# Use a smaller sample to reduce load
sample_df = customer_metrics.sample(n=1000, random_state=42)  # Reduced to 1000 rows

# Create histogram
fig = px.histogram(
    sample_df,
    x="PredictedLTV",
    nbins=25,
    color_discrete_sequence=["lightseagreen"]
)

# Update layout to be light and simple
fig.update_layout(
    title="📊 Lightweight Predicted Customer LTV Distribution",
    xaxis_title="Predicted LTV",
    yaxis_title="Customer Count",
    template="simple_white",  # Very light theme
    bargap=0.1,
    title_font=dict(size=18, color='black'),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)

# Display chart
fig.show()
plt.show()

fig.write_html(r"C:\Users\Lenovo\Downloads\interactive_ltv_chart.html")
path = r"C:\Users\Lenovo\Downloads\interactive_ltv_chart.html"
fig.write_html(path)
print(f"✅ Chart saved to: {path}")

