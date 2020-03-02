from datetime import datetime, timedelta
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def signal_correction_exp(beam, depth_z):
    """

    :param beam: Amplitude data in a XXXX x 28 format. (The script is made for input data that is 30 s averaged values)
    :param depth_z: The depth vector, where the depth is a negative value. [m]
    :return: Three values
            1) beam_corr_exp: The beam data matrix corrected for the water's dampening of the signal strenght.
            2) mean_beam_corr_exp: The mean corrected value for each depth in the dataset. Format 1 x 28.
    """

    # beam_corr_log = np.zeros([int(len(beam)), 28])
    beam_corr_exp = np.zeros([int(len(beam)), 28])

    count = 0
    for item in beam:
        for d in range(0, 28):
            depth = depth_z[d]

            # The correction value is subtracted from the original value to give a the data without the depth effect
            # Version that Lars had in an old script (no reference)
            # beam_corr_log[count][d] = item[d] - (-20 * np.log10(depth + 31) + 2 * (depth + 31) * 0.08)
            # Exponential that Lars made up
            beam_corr_exp[count][d] = item[d] - 20 * m.exp(-(depth + 31) / 5)

        count += 1

    # Calculating the mean echo amplitude for the entire dataset
    # mean_beam_corr_log = np.zeros([1, 28])
    mean_beam_corr_exp = np.zeros([1, 28])
    mean_beam_raw = np.zeros([1, 28])

    for d in range(0, 28):
        # mean_beam_corr_log[0][d] = np.mean(a5m_corr_log[:, d])
        mean_beam_corr_exp[0][d] = np.mean(beam_corr_exp[:, d])
        mean_beam_raw[0][d] = np.mean(beam[:, d])

    # Saves the raw mean, the exponentially corrected mean, filename, day (true or false), night (true or false)
    return beam_corr_exp, mean_beam_corr_exp


def halocline_depth(mean_signal_strength, depth, raw_signal_data, time_vector):
    """
    Finds the local maxima on the depth curve where the uppermost corresponds to the halocline depth. Returns the
    stratification depth and the coordinates for plotting the stratification depth on the signal strenght plot.
    equal the halocline depth (both as a value and as a point for plotting).
    :param mean_signal_strength: The mean signal strength over depth. Vector [1 x 28].
    :param depth: A vector with the depths. [1 x 28]
    :param raw_signal_data: The raw data from which the mean has been calculated (used for plotting only)
    :param time_vector: The time vector (used for plotting only)
    :return: strat_depth (the stratification depth) and strat_depth_point (point that can be used for plotting on the
    signal strenght "depth curve" [1 x 1] [value x depth].
    """

    # Finds the local mamxima points by finding the values which are larger than both adjacent values. Should maybe be
    # changed to having a positive value at one side and a negative on the other, but the outcome is the same.

    idx = 0
    signal_derivative = np.squeeze(mean_signal_strength)
    local_maxima = []
    for item in signal_derivative:
        if item == signal_derivative[0]:
            idx += 1
        elif item == signal_derivative[-1]:
            pass
        elif signal_derivative[idx - 1] < item > signal_derivative[idx + 1]:
            local_maxima.append([item, idx])
            idx += 1
        else:
            idx += 1

    # Plotting both figures in the same graph
    #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    #ax1.plot(np.transpose(mean_signal_strength), depth)
    strat_depth = -31

    for entry in local_maxima:
        if local_maxima is []:
            strat_depth = -31
            strat_depth_point = [0, -30]
        elif depth[entry[1]] > strat_depth:
            strat_depth = depth[entry[1]]
            strat_depth_point = [entry[0], depth[entry[1]]]
        #ax1.plot(entry[0], depth[entry[1]], 'kx', markersize=7)
    """
    ax1.set_title('{0} ({1})'.format('Estimated stratification depth', time_vector[0].strftime(' %d %B %Y ')))
    ax1.set_xlabel('Mean signal strength')
    ax1.set_ylabel('Depth [m]')
    ax1.set_ylim([-30.5, -3.5])

    cs = ax2.contourf(time_vector, depth, np.transpose(raw_signal_data), np.arange(35, 90, 0.5), cmap='jet')

    plt.gcf().autofmt_xdate()
    ax2.set_xlabel('Time and date (HH:MM yyyy-mm-dd)')
    ax2.set_ylabel('Depth [m]')

    ax2.set_title('{0} ({1})'.format('Echo amplitude signal strength', time_vector[0].strftime(' %d %B %Y ')))
    formatter = DateFormatter('%H:%M %Y-%m-%d')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.colorbar(cs)
    cs.changed()

    plt.show()
    """
    # Plotting the halocline depth estimation separately
    """
    fig1 = plt.plot(np.transpose(mean_signal_strength), depth)

    # As the maximum depth is 31, that will be used as the first value, as if there are no haloclines that will be the
    # Given depth. TODO: Should I change this to None instead? Would it be better to point out that?

    strat_depth = -31
    for entry in local_maxima:
        if local_maxima is []:
            strat_depth = -31
            strat_depth_point = [0, -30]
        elif depth[entry[1]] > strat_depth:
            strat_depth = depth[entry[1]]
            strat_depth_point = [entry[0], depth[entry[1]]]
        plt.plot(entry[0], depth[entry[1]], 'kx', markersize=7)
    plt.show()
    """

    # PLOTTING RAW DATA FOR COMPARISON SEPARATELY
    """
    cs = plt.contourf(time_vector, depth, np.transpose(raw_signal_data), np.arange(35, 90, 0.5), cmap='jet')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.ylabel('Depth [m]')

    plt.title('{0} ({1})'.format('Echo amplitude signal strength', time_vector[0].strftime(' %d %B %Y ')))
    formatter = DateFormatter('%H:%M %Y-%m-%d')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

    cs.changed()
    plt.colorbar()
    plt.show()
    """

    return strat_depth, strat_depth_point



