import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from sklearn.ensemble import IsolationForest
sns.set(style='dark')

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQTcwsuCjwd0SPQzb1ks6K9Gh26MOhhY_m-2yyQ0JeE_SOeEGT2F1wgiFVEdX50C3sIQY9r4wbpIQ7u/pub?gid=1662792157&single=true&output=csv"
all_df = pd.read_csv(url)

# Create a datetime column from 'year', 'month', 'day', and 'hour'
all_df['date'] = pd.to_datetime(all_df[['year', 'month', 'day', 'hour']])

# Get min and max date
min_date = all_df["date"].min().date()
max_date = all_df["date"].max().date()

with st.sidebar:
    # add Text
    st.header("Air Quality")
    # add logo
    st.image("https://st2.depositphotos.com/4520249/7571/v/950/depositphotos_75718829-stock-illustration-blowing-wind-icon.jpg")
    
    # Get start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Convert start_date and end_date to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data for Guanyuan and Wanshouxigong stations
guanyuan_data = all_df[(all_df['station'] == 'Guanyuan') & (all_df['date'] >= start_date) & (all_df['date'] <= end_date)]
wanshouxigong_data = all_df[(all_df['station'] == 'Wanshouxigong') & (all_df['date'] >= start_date) & (all_df['date'] <= end_date)]

# Aggregate data by year
agg_df1 = guanyuan_data.groupby(by="year").agg({
    "No": "nunique",
    "CO": ["max", "min", "mean", "std"],
    "NO2": ["max", "min", "mean", "std"],
    "SO2": ["max", "min", "mean", "std"],
})

agg_df2 = wanshouxigong_data.groupby(by="year").agg({
    "No": "nunique",
    "CO": ["max", "min", "mean", "std"],
    "NO2": ["max", "min", "mean", "std"],
    "SO2": ["max", "min", "mean", "std"],
})

# Streamlit app
st.title('Dashboard Kualitas Udara ☁️')

# Create plot for each column
for column in agg_df1.columns:
    st.subheader(f'Perbandingan Kualitas Kadar {column} per Tahun')
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
    ax.plot(agg_df1.index, agg_df1[column], 'o-', label='guanyuan_df')  # Change line style and label
    ax.plot(agg_df2.index, agg_df2[column], 's-', label='wanshouxigong_df')  # Change line style and label
    ax.set_xlabel('Tahun')
    ax.set_ylabel(column)
    ax.grid(True)  # Add grid
    ax.legend()
    st.pyplot(fig)

st.header("Tingkat Polusi udara tertinggi yang disebabkan oleh Kendaraan")

# Filter data based on the selected date range
filtered_all_df = all_df[(all_df['date'] >= start_date) & (all_df['date'] <= end_date)]

# Changing the data into the format required by Isolation Forest
X = filtered_all_df['PM10'].values.reshape(-1, 1)

# Create model Isolation Forest
clf = IsolationForest(contamination=0.01)

# Training the model with data 
clf.fit(X)

# Predicting whether each point in the dataset is an anomaly
pred = clf.predict(X)

# Changing the predictions into a boolean mask for normal data (not anomalies)
mask = pred != -1

# Removing anomalies from the data
filtered_df = filtered_all_df[mask]

# Create a new boxplot
fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(x='year', y='PM10', hue='station', data=filtered_df, ax=ax)
ax.set_title('Perbandingan PM10 per Tahun ')
ax.set_xlabel('Tahun')
ax.set_ylabel('PM10')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Change direct position for legend
plt.xticks(rotation=45)  # Rotate x-axis label to easy for read 
st.pyplot(fig)

#copyright
st.caption("Copyright (c) Vinsensius Adrian Wijaya 2023")
