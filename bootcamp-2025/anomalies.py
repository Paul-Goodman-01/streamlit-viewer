import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore

def plot_isolation_forest_anomalies(df_in, contamination=0.01):

    df = df_in

    # Resample to hourly data
    df = df.resample('h', on='time').min()  # Use .sum() if necessary

    # Reset index for clean DataFrame
    df = df.reset_index()

    # Calculate consumption
    df["consumption"] = df['value'].diff(1)

    # Feature engineering
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.weekday  # 0 = Monday, 6 = Sunday
    df["month"] = df["time"].dt.month

    # Encode cyclic features (for seasonality)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Add in typical profiles to the dataset
    # Compute the median profile per (day-of-week, hour)
    #diurnal_profile = df.groupby(["day_of_week", "hour"])["consumption"].median().reset_index()
    diurnal_profile = df.groupby(["day_of_week", "hour"])["consumption"].mean().reset_index()

    # Merge diurnal profile back into the dataset
    #df = df.merge(diurnal_profile, on=["day_of_week", "hour"], suffixes=("", "_median"))
    df = df.merge(diurnal_profile, on=["day_of_week", "hour"], suffixes=("", "_mean"))

    # Features 
    features = ["consumption", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

    # Train Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[features])

    # Mark anomalies (-1 means anomaly, 1 means normal)
    df["is_anomaly"] = df["anomaly_score"] == -1

    # Get anomalous points as a dataframe
    anomaly_points = df[df["is_anomaly"]]
    anomaly_name = f"Anomalies, contam.={contamination}"

    # Create plot
    #fig = px.line(df, x="time", y="consumption", title="Water Consumption with Isolation Forest Anomalies")
    fig = go.Figure()

    title = f"Anomalies detected by Isolation Forest Method (Contamination={contamination})"

    # Add raw data as trace
    fig.add_trace(go.Scatter(x=df["time"], 
                             y=df["consumption"], 
                             mode='lines', 
                             line=dict(color="cyan"),
                             name='Value'))

    # Add anomalies as points
    fig.add_trace(go.Scatter(
        x=anomaly_points["time"],
        y=anomaly_points["consumption"],
        mode="markers",
        marker=dict(color="red", size=8, symbol="circle-open"),
        name=anomaly_name
    ))

    # Add typical diurnal profile (varies by day-of-week)
    fig.add_trace(go.Scatter(
        x=df["time"],
        #y=df["consumption_median"],
        y=df["consumption_mean"],
        mode="lines",
        line=dict(color="green", dash="dot"),
        name="Mean Diurnal Profile (by Day)"
    ))

    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation='h',  # Horizontal legend
            y=-0.25,  # Move it below the plot
            x=0.5,   # Center it horizontally
            xanchor='center',
            yanchor='top'
        ),
        title=title,
        xaxis_title="Date and Time",
        yaxis_title="Water Consumption"
    )

    return fig

def plot_statistical_anomalies(df_in, x=2.0):
    
    # Setup
    df = df_in

    # Resample to hourly values
    df = df.resample('h', on='time').min().reset_index()

    # Add consumption
    df["consumption"] = df['value'].diff(1)
    df["consumption"] = df["consumption"].fillna(0)

    # Extract hour and day of the week
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.weekday  # 0 = Monday, 6 = Sunday

    # Compute mean and standard deviation per (day_of_week, hour)
    diurnal_stats = df.groupby(["day_of_week", "hour"])["consumption"].agg(["mean", "std"]).reset_index()

    # Merge diurnal stats back into the dataset
    df = df.merge(diurnal_stats, on=["day_of_week", "hour"], suffixes=("", "_expected"))

    # Compute anomaly threshold (values > mean + x * std)
    df["threshold"] = df["mean"] + x * df["std"]
    df["is_anomaly"] = df["consumption"] > df["threshold"]

    # Extract anomaly points
    anomaly_points = df[df["is_anomaly"]]

    # Create line plot for actual consumption
    title = f"Anomalies detected by 'classical' statistical method (Sigma > {x})"
    anomaly_name = f"Anomalies (>{x}σ)"

    # Create plot
    #fig = px.line(df, x="time", y="consumption", title="Water Consumption with Isolation Forest Anomalies")
    fig = go.Figure()

    # Add raw data as trace
    fig.add_trace(go.Scatter(x=df["time"], 
                             y=df["consumption"], 
                             mode='lines', 
                             line=dict(color="cyan"),
                             name='Value'))

    # Add mean diurnal profile as a reference line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["mean"],
        mode="lines",
        line=dict(color="green", dash="dot"),
        name="Mean Diurnal Profile (by Day)"
    ))

    # Add red scatter points for anomalies
    fig.add_trace(go.Scatter(
        x=anomaly_points["time"],
        y=anomaly_points["consumption"],
        mode="markers",
        marker=dict(color="red", size=8, symbol="circle-open"),
        name=anomaly_name
    ))

    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation='h',  # Horizontal legend
            y=-0.25,  # Move it below the plot
            x=0.5,   # Center it horizontally
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title="Date and Time",
        yaxis_title="Water Consumption",
        title=title
    )

    # Return figure
    return fig

def plot_lgbm_anomalies(df_in, x=2.0):
    # Setup
    df = df_in

    # Resample to hourly values
    df = df.resample('h', on='time').min().reset_index()

    # Add consumption
    df["consumption"] = df['value'].diff(1)
    df["consumption"] = df["consumption"].fillna(0)

    # Feature Engineering
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.weekday  # 0 = Monday, 6 = Sunday
    df["month"] = df["time"].dt.month

    # Add previous hour's consumption as a feature (for temporal patterns)
    df["consumption_lag1"] = df["consumption"].shift(1)

    # Drop first row (due to NaN in lag feature)
    df = df.dropna()

    # Train/Test Split
    features = ["hour", "day_of_week", "month", "consumption_lag1"]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df["consumption"], test_size=0.2, shuffle=False)

    # Train LightGBM Model
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    df["predicted"] = model.predict(df[features])

    # Compute residuals (actual - predicted)
    df["residual"] = df["consumption"] - df["predicted"]

    # Identify anomalies using 3-sigma rule
    df["residual_std"] = df["residual"].std()
    df["threshold"] = x * df["residual_std"]
    df["is_anomaly"] = abs(df["residual"]) > df["threshold"]

    # Extract anomaly points
    anomaly_points = df[df["is_anomaly"]]

    # Create plot
    title = f"Anomalies detected by LightGBM method (Sigma > {x})"
    anomaly_name = f"Anomalies (>{x}σ)"

    # Create plot
    #fig = px.line(df, x="time", y="consumption", title="Water Consumption with Isolation Forest Anomalies")
    fig = go.Figure()

    # Add raw data as trace
    fig.add_trace(go.Scatter(x=df["time"], 
                             y=df["consumption"], 
                             mode='lines', 
                             line=dict(color="cyan"),
                             name='Value'))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["predicted"],
        mode="lines",
        line=dict(color="green", dash="dot"),
        name="LGBM Predicted Consumption"
    ))

    # Add red scatter points for anomalies
    fig.add_trace(go.Scatter(
        x=anomaly_points["time"],
        y=anomaly_points["consumption"],
        mode="markers",
        marker=dict(color="red", size=8, symbol="circle-open"),
        name=anomaly_name
    ))

    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation='h',  # Horizontal legend
            y=-0.25,  # Move it below the plot
            x=0.5,   # Center it horizontally
            xanchor='center',
            yanchor='top'
        ),
        title=title,
        xaxis_title="Date and Time",
        yaxis_title="Water Consumption"
    )

    # Return figure
    return fig

def plot_stl_anomalies(df_in, x=2.0):
    # Setup
    df = df_in   
    
    # Resample to hourly values
    df = df.resample('h', on='time').min().reset_index()

    # Add consumption
    df["consumption"] = df['value'].diff(1)
    df["consumption"] = df["consumption"].fillna(0)

    # Apply STL decomposition
    stl = STL(df["consumption"], period=24*7, robust=True)  # 7-day period for weekly seasonality
    result = stl.fit()

    # Extract components
    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["residual"] = result.resid

    # Identify anomalies using Z-score on residuals
    df["z_score"] = zscore(df["residual"])
    df["is_anomaly"] = abs(df["z_score"]) > x  # Threshold: |Z| > x

    # Extract anomaly points
    anomaly_points = df[df["is_anomaly"]]

    # Create time-series plot
    title = f"Anomalies detected by STL-Decomposition (Sigma > {x})"
    anomaly_name = f"Anomalies (>{x}σ)"

    # Create plot
    #fig = px.line(df, x="time", y="consumption", title="Water Consumption with Isolation Forest Anomalies")
    fig = go.Figure()

    # Add raw data as trace
    fig.add_trace(go.Scatter(x=df["time"], 
                             y=df["consumption"], 
                             mode='lines', 
                             line=dict(color="cyan"),
                             name='Value'))

    # Add seasonal component (expected consumption)
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["trend"] + df["seasonal"],  # Expected consumption
        mode="lines",
        line=dict(color="green", dash="dot"),
        name="Expected Consumption (Trend + Seasonality)"
    ))

    # Add red scatter points for anomalies
    fig.add_trace(go.Scatter(
        x=anomaly_points["time"],
        y=anomaly_points["consumption"],
        mode="markers",
        marker=dict(color="red", size=8, symbol="circle-open"),
        name=anomaly_name
    ))

    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation='h',  # Horizontal legend
            y=-0.25,  # Move it below the plot
            x=0.5,   # Center it horizontally
            xanchor='center',
            yanchor='top'
        ),
        title=title,
        xaxis_title="Date and Time",
        yaxis_title="Water Consumption"
    )

    # Show figure
    return fig
