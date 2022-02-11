import numpy as np


def show_random_forest_stats(result, expected):
    """ Showing stats related to the predicted and expected results"""
    error = abs(result - expected)  # Calculating the error
    avgAbsSrr = np.mean(error)      # Calculating average absolute error
    print('Average absolute error: {}%'.format(avgAbsSrr.values[0]*100))
    
