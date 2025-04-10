import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌦️ Weather Data Analysis and Prediction App")

# Load the dataset
df = pd.read_csv("weatherHistory.csv")  # Replace with your actual filename

# Rename columns for simplicity
df.rename(columns={
    "Formatted Date": "date",
    "Temperature (C)": "temperature",
    "Apparent Temperature (C)": "feels_like",
    "Humidity": "humidity",
    "Wind Speed (km/h)": "wind_speed",
    "Pressure (millibars)": "pressure",
    "Visibility (km)": "visibility",
    "Summary": "summary",
    "Precip Type": "precip_type",
    "Wind Bearing (degrees)": "wind_bearing",
    "Loud Cover": "cloud_cover",
    "Daily Summary": "daily_summary"
}, inplace=True)

# Convert date column
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

# Sidebar for date filtering
st.sidebar.header("📅 Date Range Filter")
start_date = pd.to_datetime(st.sidebar.date_input("Start Date", df['date'].min().date()))
end_date = pd.to_datetime(st.sidebar.date_input("End Date", df['date'].max().date()))

# Filter the dataframe
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

if df.empty:
    st.warning("⚠️ No data found for the selected date range.")
    st.stop()

# Show available columns
st.subheader("📌 View Specific Weather Columns")
columns = st.multiselect("Choose columns to display:", df.columns.tolist(), default=["date", "temperature", "humidity", "pressure"])

if columns:
    st.dataframe(df[columns])
else:
    st.info("ℹ️ No columns selected.")

# Line chart of temperature
st.subheader("📈 Temperature Over Time")
fig, ax = plt.subplots()
ax.plot(df['date'], df['temperature'], color='orange', label='Temperature (C)')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

# Linear Regression
st.subheader("📉 Temperature Prediction using Linear Regression")

features = ["humidity", "pressure", "wind_speed", "visibility"]
df[features] = df[features].ffill().bfill()
df['temperature'] = df['temperature'].ffill().bfill()

if len(df) >= 2:
    X = df[features]
    y = df['temperature']
    
    model = LinearRegression()
    model.fit(X, y)
    df['predicted_temp'] = model.predict(X)
    
    st.success("✅ Linear Regression Model Trained!")
    
    fig2, ax2 = plt.subplots()
    ax2.plot(df['date'], df['temperature'], label='Actual', color='blue')
    ax2.plot(df['date'], df['predicted_temp'], label='Predicted', color='red', linestyle='dashed')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("⚠️ Not enough data to make predictions.")

# Map-based visualization
st.subheader("🌍 Weather Map (Random Coordinates Example)")

# Generate fake lat/lon for demonstration (use real data if available)
np.random.seed(42)
df['lat'] = 19 + np.random.rand(len(df)) * 1  # Example: around Maharashtra
df['lon'] = 72 + np.random.rand(len(df)) * 1

fig_map = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="temperature",
    size="humidity",
    hover_name="summary",
    hover_data=["temperature", "humidity", "pressure"],
    zoom=5,
    height=500
)

fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map)

# Footer
st.markdown("---")
st.markdown("✅ Built with ❤️ for DSBDA & Cloud Computing Mini Project")

