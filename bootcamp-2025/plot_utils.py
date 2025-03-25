#Import the packages we need for the time variation plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# This is a bit of a lengthy function that sets up the plotting of two figures:
# 1. Summary time variation plots as the first figure, giving us diurnal, monthly and day-of-week hourly summaries
# 2. Diurnal profiles for each day-of-week

def time_variation(df_in, date_column, pollutant, ylabel, hue=None):
    week = ['mon','tue','wed','thu','fri','sat','sun']
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    palette = 'pastel'
    top_y_mult = 1.1
    bottom_y_mult = 1.8
    
    # Kludge - need to resample for onehour and recalculate derivatives.
    df = df_in
    df = df.resample('h', on='time').min().reset_index()
    df['first_derivative'] = df['value'].diff()  # First derivative (difference of successive values)
    df['second_derivative'] = df['first_derivative'].diff()  # Second derivative (difference of first derivative)

    # converting feature date to datetime and dropping Nan values
    if hue!=None:
        df = df[[date_column, pollutant, hue]].dropna() # dropping rows of nan vals of the chosen pollutant
    else:
        df = df[[date_column, pollutant]].dropna()

    #Plots
    sns.set_style('whitegrid') 

    if hue!=None:
        pollutant_max=max(df.groupby([df[date_column].dt.hour, hue])[pollutant].mean())*top_y_mult #setting the lim of y due to max mean
    else:
        pollutant_max=max(df.groupby(df[date_column].dt.hour)[pollutant].mean())*top_y_mult 

    # Creating graphs of pollutant concentrations by hour, month and day of week
    fig,axes = plt.subplots(1, 3, sharex=False, figsize=(16,4)) #creating subplots, side by side
    fig.tight_layout(pad=-2) # making plots get closer
    fig.patch.set_facecolor('black')
        
    # value vs hour
    axes[0] = sns.lineplot(ax=axes[0],data=df,
                            x=df[date_column].dt.hour,
                            y=pollutant,
                            color='red',
                            linewidth=1.5,
                            hue=hue,
                            palette=palette,
                            err_kws={"alpha": 0.5, "linewidth": 0})
    axes[0].set_xticks(np.arange(0, 23, 4))
    axes[0].set_xticklabels(axes[0].get_xticks(), fontsize=13)
    #axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=13)
    axes[0].set_xlabel('hour', fontsize=15)
    axes[0].set_ylabel(ylabel, fontsize=15)
    axes[0].set_ylim(0, pollutant_max)
    axes[0].legend().set_title('')

    #fig.patch.set_facecolor('black')
    axes[0].set_facecolor('black')
    axes[0].xaxis.label.set_color("white")
    axes[0].yaxis.label.set_color("white")
    axes[0].title.set_color("white")
    axes[0].tick_params(colors="white")
    
    # value vs month
    axes[1] = sns.lineplot(ax=axes[1],
                           data=df,
                           x=df[date_column].dt.month,
                           y=pollutant,
                           color='red',
                           linewidth=1.5,
                           hue=hue,
                           palette=palette,
                           err_kws={"alpha": 0.5, "linewidth": 0})
    axes[1].set_xticks(np.arange(1, 13, 1))
    axes[1].set_xticklabels(months, fontsize=13)
    axes[1].set_yticklabels('')
    axes[1].set_xlabel('month', fontsize=15)
    axes[1].set_ylabel('')
    axes[1].set_ylim(0, pollutant_max)
    axes[1].legend().set_title('')
    axes[1].set_facecolor('black')
    axes[1].xaxis.label.set_color("white")
    axes[1].yaxis.label.set_color("white")
    axes[1].title.set_color("white")
    axes[1].tick_params(colors="white")

    # value vs day of week
    axes[2] = sns.lineplot(ax=axes[2],
                           data=df,
                           x=df[date_column].dt.dayofweek,
                           y=pollutant,
                           color='red',
                           linewidth=1.5,
                           hue=hue,
                           palette=palette,
                           err_kws={"alpha": 0.5, "linewidth": 0})
    axes[2].set_xticks(np.arange(0, 7, 1))
    axes[2].set_xticklabels(week, fontsize=13)
    axes[2].set_yticklabels('')
    axes[2].set_xlabel('day of week', fontsize=15)
    axes[2].set_ylabel('')
    axes[2].set_ylim(0, pollutant_max)
    axes[2].legend().set_title('')
    axes[2].set_facecolor('black')
    axes[2].xaxis.label.set_color("white")
    axes[2].yaxis.label.set_color("white")
    axes[2].title.set_color("white")
    axes[2].tick_params(colors="white")

    # creating graphs of concentration by hour by specific day of week
    fig2,axes2 = plt.subplots(1, 7, sharex=True, figsize=(16,2)) #subplots for each day of week
    fig2.tight_layout(pad=-2)
    fig2.patch.set_facecolor('black')

    if hue!=None:
        pollutant_max=max(df.groupby([df[date_column].dt.hour, hue])[pollutant].mean())*bottom_y_mult #setting the lim of y due to max mean
    else:
        pollutant_max=max(df.groupby(df[date_column].dt.hour)[pollutant].mean())*bottom_y_mult 

    for i in range(7):
        axes2[i] = sns.lineplot(ax=axes2[i],data=df,
                                x=df[df[date_column].dt.dayofweek==i][date_column].dt.hour,
                                y=pollutant,
                                color='red',
                                linewidth=1,
                                hue=hue,
                                palette=palette,
                                err_kws={"alpha": 0.5, "linewidth": 0})
        axes2[i].set_xticks(np.arange(0, 23, 4))
        axes2[i].set_xlabel('hour', fontsize=12)
        if i == 0:
            axes2[i].set_ylabel(ylabel, fontsize=12)
        else:
            axes2[i].set_ylabel('')
            axes2[i].set_yticklabels('')
        axes2[i].set_ylim(0, pollutant_max)
        axes2[i].set_title(week[i])
        axes2[i].legend().set_title('')
        axes2[i].set_facecolor('black')
        axes2[i].xaxis.label.set_color("white")
        axes2[i].yaxis.label.set_color("white")
        axes2[i].title.set_color("white")
        axes2[i].tick_params(colors="white")
    
    return fig, fig2

# Plotting function 
def plot_time_series(df, yaxis_title="Value"):

    # Convert DataFrame to dictionary
    data_dict = df.to_dict(orient="list")
    
    # Convert time column to datetime if it's not already
    time_series = pd.to_datetime(data_dict["time"])

    # Identify indices where 'signals' switches from 0 to non-zero
    signal_changes = [
        i for i in range(1, len(data_dict["signals"])) 
        if data_dict["signals"][i-1] == 0 and data_dict["signals"][i] != 0
    ]

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for the data
    fig.add_trace(go.Scatter(x=time_series, y=data_dict["value"], mode='lines', name='Value'))
    fig.add_trace(go.Scatter(x=time_series, y=data_dict["avgFilter"], mode='lines', name='Avg Filter'))
    fig.add_trace(go.Scatter(x=time_series, y=data_dict["upper_bound"], mode='lines', name='Avg + Thr.'))
    fig.add_trace(go.Scatter(x=time_series, y=data_dict["lower_bound"], mode='lines', name='Avg - Thr.'))

    # Add vertical lines at signal change points
    for i in signal_changes:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=time_series[i], x1=time_series[i],
                y0=min(min(data_dict["value"]), min(data_dict["lower_bound"])),
                y1=max(max(data_dict["value"]), max(data_dict["upper_bound"])),
                line=dict(color="yellow", width=1, dash="dash")
            )
        )

    # Customize layout
    fig.update_layout(
        title="Raw Data + Threshold Filter Anomalies",
        xaxis_title="Date and Time",
        yaxis_title=yaxis_title,
        legend_title="Legend",
        xaxis=dict(type='date')  # Ensure x-axis is treated as a datetime
    )

    return fig

