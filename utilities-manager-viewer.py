import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from timer import Timer
from plot_utils import time_variation, plot_time_series
from thresholding import thresholding_algo
from anomalies import plot_isolation_forest_anomalies, plot_statistical_anomalies, plot_lgbm_anomalies, plot_stl_anomalies

# Initialise session state variables
if "initialised" not in st.session_state:
    st.session_state.file_list = []
    st.session_state.file_index = -1
    st.session_state.last_directory = "" 
    st.session_state.directory = ""
    st.session_state.current_file = None
    st.session_state.x_sigma = 2.0
    st.session_state.contamination = 0.01
    st.session_state.initialised = True
    st.session_state.lag = 3
    st.session_state.base = 2.0
    st.session_state.base_multi = 100.0 
    st.session_state.multi = 1.0
    st.session_state.influence = 0.1

# Functions to update file list and reset file state if a problem occurs
def reset_file_state():
    print("Resetting file state...")
    st.session_state.file_list = []
    st.session_state.file_index = -1
    st.session_state.current_file = None
    
def update_file_list(directory):
    if st.session_state.last_directory.lower() != st.session_state.directory.lower():
        print(f"Change {st.session_state.directory}")
        if os.path.isdir(st.session_state.directory):
            st.session_state.file_list = [f for f in os.listdir(directory) if f.endswith(".csv")]
            if st.session_state.file_list:
                st.session_state.file_index = 0
                st.session_state.current_file = st.session_state.file_list[st.session_state.file_index]
            else:
                reset_file_state()
        else:
            reset_file_state()
    st.session_state.last_directory = st.session_state.directory
   
# Function to catch the pressing of enter on the directory text box
def on_enter():
    update_file_list(st.session_state.directory)

# Function for numeric input validation
def is_valid_number(value, min, max):
    try:
        num = float(value)
        return min <= num <= max  # Ensure the number is positive
    except ValueError:
        return False

# Cast value to an int  
def force_int(value, default= 0):
    try:
        return int(value)
    except ValueError:
        print(f"Invalid input! Defaulting to {default}.")
        return int(default) 

# Check if a numeric value entry has changed
def check_value_changed(value, key, min, max):
    if is_valid_number(value, min, max):
        value = float(value)
        if st.session_state[key] != value:
            st.session_state[key] = value
            st.rerun()
    else:
        st.error(f"Invalid input! Please enter a number in range {min} to {max}!")
                  
# Function to load a data file
def load_file():
    # Set the path from current session state
    file_path = os.path.join(st.session_state.last_directory, st.session_state.current_file)
    print(f"Load file: {st.session_state.last_directory}, {st.session_state.current_file}")

    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found!")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty!")
    except pd.errors.ParserError:
        print(f"Error: The file '{file_path}' could not be parsed!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# Function to process a raw data dataframe adding dates, times and derivatives. 
def process_file(df):
    print(f"Processing file...")

    # Keep a copy of the raw data.
    df_raw = df

    # Convert 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # Add additional time and date data 
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['month_name'] = df['time'].dt.month_name()
    df.rename(columns={'day': 'day_name'}, inplace=True)
    df['day_of_week'] = df['time'].dt.day_of_week
    df['hour'] = df['time'].dt.hour

    # Calculate the first and second derivatives
    df['first_derivative'] = df['value'].diff()  # First derivative (difference of successive values)
    df['second_derivative'] = df['first_derivative'].diff()  # Second derivative (difference of first derivative)

    # Set base to be the average increase over time...
    first_value = df['value'].iloc[0]  # First value in the column
    last_value = df['value'].iloc[-1]  # Last value in the column
    num_entries = len(df['value'])  # Number of entries in the column
    
    # Set the gradient of increase to be the base value for the thresholding algorithm
    base = st.session_state.base + st.session_state.base_multi * ((last_value - first_value) / num_entries)

    # Force lag to be an int
    lag = force_int(st.session_state.lag)

    # Apply thresholding
    result = thresholding_algo(df["value"], 
                               lag, 
                               base, 
                               st.session_state.multi, 
                               st.session_state.influence)

    # Apply results
    df['signals'] = result['signals']
    df['avgFilter'] = result['avgFilter']
    df['thresholds'] = result['thresholds']
    df['upper_bound'] = np.array(result["avgFilter"]) + np.array(result["thresholds"])
    df['lower_bound'] = np.array(result["avgFilter"]) - np.array(result["thresholds"])

    return df

# Setup timer
timer = Timer()

###############################################################################################
## BEGIN UI 

# Title
st.title("Utilities Manager Viewer")

# UI for directory selection
col1, col2 = st.columns([4, 1])

with col1:
    st.text_input("Current directory path:", st.session_state.last_directory, key="directory", on_change=on_enter)

with col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
    if st.button("Load Directory"):
        update_file_list(st.session_state.directory)

st.markdown("---")

# Navigation buttons
if st.session_state.file_list:

    # Layout: Buttons + Filename
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
        if st.button("◀ Prev") and st.session_state.file_index > 0:
            st.session_state.file_index -= 1
            st.session_state.current_file = st.session_state.file_list[st.session_state.file_index]
            st.rerun()
            
    with col2:
        text = f'{st.session_state.file_index} : {st.session_state.current_file}'        
        st.text_input("Currently Viewing:", text, disabled=True, key="filename")
        
    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
        if st.button("Next ▶") and len(st.session_state.file_list) - 1:
            st.session_state.file_index += 1
            st.session_state.current_file = st.session_state.file_list[st.session_state.file_index]
            st.rerun()

    st.markdown("---")

    if st.session_state.file_index > -1:    
        
        st.session_state.current_file = st.session_state.file_list[st.session_state.file_index]
        # Get the current file and create dataframe
        df_raw = load_file()
        df = process_file(df_raw)

        # Ensure required columns exist
        required_columns = {"time", "value", "avgFilter", "upper_bound", "lower_bound", "signals"}
        if not required_columns.issubset(df.columns):
            st.error(f"CSV must contain columns: {required_columns}")
        else:
            # Display the raw dataframe as a table
                        
            st.markdown("### Raw Dataframe:")
            st.markdown("This is the raw data loaded from the .csv file for the site/water meter.")
            st.dataframe(df_raw)

            st.markdown("---")        

             # Header for Plotting UI
            st.markdown("### Raw meter readings with Threshold-Based Anomaly Detection:")

            st.markdown("""This section presents the raw data read from the .csv file for a particular site/water meter. An anomaly detection algorithm based on thresholding is applied to the data.""")
            st.markdown("""The thresholding algorithm calculates a moving average, based on a window of `lag` previous values. An anomaly is defined if the actual value received in a period exceeds the moving average value +/- the threshold.""")
            st.markdown("""The threshold is calculated by |(`base` x `base multiplier`) + (`sigma multiplier` x `sigma`)|. `Sigma` here is defined as the standard deviation of the values in the current window.""")
            st.markdown("""An `influence` parameter is also added to control how much of the last calculated value (i.e. at 't-1') should be taken into account in assessing the current period (i.e. time='t')""")
            st.markdown("""The default parameter are set to identify large discontinuities in the datasets.""")

            # Layout anomaly detection
            col1, col2 = st.columns([1, 1])

            with col1:    
                lag_temp = st.text_input("Lag:", st.session_state.lag, key="lag_edit")

            with col2:
                influence_temp = st.text_input("Influence:", st.session_state.influence, key="influence_edit")             

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:    
                base_temp = st.text_input("Base:", st.session_state.base, key="base_edit")

            with col2:
                base_multi_temp = st.text_input("Base Multiplier:", st.session_state.base_multi, key="base_multi_edit")

            with col3:
                sigma_multi_temp = st.text_input("Sigma Multiplier:", st.session_state.multi, key="sigma_multi_edit")

            check_value_changed(lag_temp, "lag", 0, 99999.9)
            check_value_changed(influence_temp, "influence", 0.0, 1.0)
            check_value_changed(base_temp, "base", 0.0, 99999.9)
            check_value_changed(base_multi_temp, "base_multi", 0.0, 99999.9)
            check_value_changed(sigma_multi_temp, "multi", 0.0, 20.0)

           
            # Generate and display the plot
            timer.start()
            fig = plot_time_series(df, "Meter reading, m<sup>3</sup>")
            st.plotly_chart(fig, use_container_width=True)
            elapsed = timer.stop()
            text = f"Parameters used: Lag: {st.session_state.lag}, Influence: {st.session_state.influence}, Base: {st.session_state.base}, Base Multi.: {st.session_state.base_multi}, Sigma Multi.: {st.session_state.multi}"
            st.text(text)
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)

            st.markdown("---")

            # Layout: 1st Derivatives
            st.markdown("### Water Consumption:")
            st.markdown("This plot gives the first derivative of the water meter reading (i.e. water consumption rate in a 15-min. period).")

            timer.start()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['time'], y=df['first_derivative'], mode='lines', name='First Derivative'))
            fig1.update_layout(title="First Derivative (i.e. Water consumption rate)", xaxis_title="Date and Time", yaxis_title="Water consumption, m<sup>3</sup>/15-min.")
            st.plotly_chart(fig1, use_container_width=True)
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)            
            
            st.markdown("---")

            # Layout: 2nd Derivatives
            st.markdown("### Rate of Change of Water Consumption:")
            st.markdown("This plot gives the second derivative of the water meter reading (i.e. rate of change of water consumption in a 15-min. period).")

            timer.start()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['time'], y=df['second_derivative'], mode='lines', name='Second Derivative'))
            fig2.update_layout(title="Second Derivative (i.e. Rate of change of consumption)", xaxis_title="Date and Time", yaxis_title="Change in consumption rate, m<sup>3</sup>/15-min<sup2</sup>")
            st.plotly_chart(fig2, use_container_width=True)
            elapsed = timer.stop()
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)  
         
            st.markdown("---")

            # Layout: Derivatives
            st.markdown("### Hourly, Daily and Seasonal Variations in Water Consumption:")

            st.markdown("""The following plots break down hourly water consumption (first derivitive) data by day-of-week, month and weekday to reveal patterns and trends in the data. Shaded areas represent the [95% confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) of the mean.""")

            # Layout time variation plot
            timer.start()
            fig3 ,fig4 = time_variation(df, 'time', 'first_derivative', 'Water consumption, m$^3$', hue='year')
            st.write("Time Variation: Diurnal, Monthly and Daily Consumption Profiles:")
            st.pyplot(fig3, use_container_width=True)
            st.write("Time Variation: Diurnal Consumption Profiles by Day-of-Week:")
            st.pyplot(fig4, use_container_width=True)
            elapsed = timer.stop()
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)             

            st.markdown("---")
            st.markdown("### Anomaly Detection - Based on Hourly Consumption Data:")

            st.markdown("""This section presents a variety of anomaly detection algorithms based on using hourly consumption rate (i.e. first derivative) data.""")
            st.markdown("""The following two edit boxes set values used by the various anomaly detection algorithms.""")
            st.markdown(""" - `Contamination` is used by the Isolation Forest method and is the proportion of data *expected* to be anomalous. """)
            st.markdown(""" - `Sigma multiplier` is used by all other algorithms to set the number of standard deviations away from the expected/predicted value that an anomaly may occur.""")
            st.markdown("""**NB: All of the methods below are currently based on post-processing all available data, and would need modifying for real-time operation.**""")
            st.markdown("""**There are also numerous improvements in data handling, pre-processing and feature extraction that could be made.**""")

            # Layout anomaly detection
            col1, col2 = st.columns([1, 1])

            with col1:    
                contam_temp = st.text_input("Contamination:", st.session_state.contamination, key="contamination_edit")

            with col2:
                sigma_temp = st.text_input("Sigma multiplier:", st.session_state.x_sigma, key="sigma_mult_edit")              
            
            check_value_changed(contam_temp, "contamination", 0.0, 1.0)
            check_value_changed(sigma_temp, "x_sigma", 0.0, 20.0)                

            st.markdown("#### Using classical statistics to detect anomalies:")
            st.markdown("""In this method we assume a distribution of values for each hour in a particular day (e.g. a normal distribution), then flag anomalies as those points that are more than a certain number of standard deviations away from the mean.""")

            timer.start()     
            fig5 = plot_statistical_anomalies(df, x=st.session_state.x_sigma)
            st.plotly_chart(fig5, use_container_width=True)  
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)

            st.markdown("#### Using an Isolation Forest:")
            st.markdown("""In this method we build an [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) model, and then extract anomalous points based on their depth within the forest structure.""")
            st.markdown("""An Isolation Forest model is a form of [ensemble-based](https://en.wikipedia.org/wiki/Ensemble_learning) predictor, using [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning).""")

            timer.start() 
            fig6 = plot_isolation_forest_anomalies(df, contamination=st.session_state.contamination)
            st.plotly_chart(fig6, use_container_width=True)
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."  
            st.text(text)

            st.markdown("#### Using STL (Seasonal, Trend and LOESS) Decomposition:")
            st.markdown("""In this method we build a [time-series STL](https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html) model to predict expected hourly values, and then use the [residual values](https://en.wikipedia.org/wiki/Errors_and_residuals) to flag points that are more than a set number of standard deviations away from the mean residual value.""")
            st.markdown("""If you've not met it before, LOESS stands for [Locally-weighted scatterplot smoothing](https://en.wikipedia.org/wiki/Local_regression).""")

            timer.start() 
            fig7 = plot_stl_anomalies(df, x=st.session_state.x_sigma)
            st.plotly_chart(fig7, use_container_width=True)
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)  
            
            st.markdown("#### Using LightGBM (Light Gradient-Boosting Model) Regression:")
            st.markdown("""In this method we build a [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) model to predict expected hourly values, and then use the [residual values](https://en.wikipedia.org/wiki/Errors_and_residuals) to flag points that are more than a set number of standard deviations away from the mean residual value.""")
            st.markdown("""Wikipedia provides a (mathematics-heavy) description of [gradient-boosting](https://en.wikipedia.org/wiki/Gradient_boosting).""")

            timer.start() 
            fig8 = plot_lgbm_anomalies(df, x=st.session_state.x_sigma)
            st.plotly_chart(fig8, use_container_width=True)
            text = f"Elapsed time: {round(timer.stop(), 4)} seconds."
            st.text(text)            

            st.markdown("**NB: There are many other ways anomalies could be calculated!**")

            st.markdown("---")
    else:
        st.error(f"No current file - hit 'prev' or 'next'!")
else:
    st.error(f"No .csv files found or directory not set!")
    reset_file_state()

