import numpy as np

# Thresholding function
# 'y' is an array of values
# 'lag' is the size of window to use to calculate a moving average (i.e. smoothed values)

# The threshold for 'events' is calculated from +/- abs('base' + 'multi' * current window std. dev.)  
# 'base' is the size of initial threshold between the current value and the smoothed value over which an 'event' may be considered to occur.
# 'multi' is the multiplier applied to the current standard deviation. 
# 'influence' is the relative importance of the new raw value to the last filtered value.
def thresholding_algo(y, lag, base, multi, influence):
    signals = np.zeros(len(y))
    filtered_y = np.array(y[:lag])
    avg_filter = np.zeros(len(y))
    std_filter = np.zeros(len(y))
    thresholds = np.zeros(len(y))
    
    avg_filter[lag-1] = np.nanmean(y[:lag])
    std_filter[lag-1] = np.nanstd(y[:lag])
    
    for i in range(lag, len(y)):
        threshold = (base + (multi * std_filter[i-1]))
        if abs(y[i] - avg_filter[i-1]) > threshold:
            signals[i] = 1 if y[i] > avg_filter[i-1] else -1
            filtered_y = np.append(filtered_y, influence * y[i] + (1 - influence) * filtered_y[-1])
        else:
            signals[i] = 0
            filtered_y = np.append(filtered_y, y[i])
        
        avg_filter[i] = np.nanmean(filtered_y[i-lag+1:i+1])
        std_filter[i] = np.nanstd(filtered_y[i-lag+1:i+1])
        thresholds[i] = threshold
    
    # Backfill pre-lag index values
    filtered_y[0:lag-1] = filtered_y[lag]
    avg_filter[0:lag-1] = avg_filter[lag]
    std_filter[0:lag-1] = std_filter[lag]
    thresholds[0:lag-1] = thresholds[lag]

    return {"raw": filtered_y, "signals": signals, "avgFilter": avg_filter, "thresholds": thresholds}
