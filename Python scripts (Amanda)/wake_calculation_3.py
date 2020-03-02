from datetime import datetime, timedelta
from matplotlib import ticker
from csv import writer
from operator import itemgetter
import time
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.dates as mdates
from windrose import WindroseAxes, WindAxes
import matplotlib.cm as cm


def wake_depth(raw_amplitude_data, mean_amplitude_data, start_time, raw_time, mean_time, d_range, factor, duration):
    """
    Calculates the maximum depth and longevity of the wakes for the time periods (wake occurrences) in the start_times
    list. Returns a list of list, where each list contains information about the wake (time, duration, maximum depth).
    :param raw_amplitude_data: The raw data of the mean echo amplitude that is calculated in the Matlab script.
    :param mean_amplitude_data: The hourly (?) mean amplitude of the time series, which will be used to identify the
    wake signal, as the wake signal strenght will be diverging strongly from the mean value for the time period.
    :param start_time: The time and date to start the wake-calculation. The list is given by the closest shop passages
    in the AIS data and/or manually verified. The format is datetime objects.
    :param raw_time: The time vector that goes with the raw_amplitude_data (high resolution), in datetime format.
    :param mean_time: The time vector that goes with the mean_amplitude_data (lower resolution), in datetime format.
    :param d_range: The max depth for which the wake is calculated. An integer representing a depth cell.
    :param factor: The percentage that the wake signal must be compared to the average to be considered a wake.
    :param duration: The number of time-steps to be added to the starting time. Defines the duration of the calculation.
    :return: A list of lists of list where each "final" list contains the following information:
            0) current_date:
            1) current_depth:
            2) mean_date: The date
            3) mean_amplitude:
            4) wake: Indicates if the measured value is part of a wake. Can be either True or False.
    """
    start_times = [start_time]
    wake_info = []

    # Item is one of the datetime objects in the start_times list, which represents a ship passage
    for item in start_times:
        current_list = []
        # The index() function will give the position (row, column) of the item that is indexed. ie. the position of the
        # start time for the wake. The column of the start value, will be the start row for the amplitude raw dataset.
        # start = raw_time.index(item) (it does not work as the time must be the exact time, and that will not happen)

        count = 0
        for r_date in raw_time:
            if r_date is item:
                start = count
                count += 1
            elif r_date is raw_time[-1]:
                if item > r_date:
                    print('Error in finding start date in wake_depth, value not in range')
                elif raw_time[-2] < item < r_date:
                    start = count
                else:
                    pass
                    #print('Error in finding start date in wake_depth, end of list')
            elif r_date > item < raw_time[count + 1]:
                count += 1
            elif r_date < item < raw_time[count + 1]:
                start = count
                count += 1
            elif r_date < item > raw_time[count + 1]:
                count += 1
            else:
                print('Error in finding start date in wake_depth')

        # There are 120 averaged measurements per hour --> adding 60 to the index would give measurement for 30 min
        # adding 90 would be 45 min, and 120 would give measurement for 1 hour
        for value in range(start, (start + duration)):

            current_date = raw_time[value]
            # Find a mean_time within 15 min range of the current raw_time. Each mean time corresponds to a 30 min block
            # The date_indx will be used to find the corresponding mean value in the raw_amplitude_data.
            date_indx = 0

            small = (current_date - timedelta(minutes=15))
            large = (current_date + timedelta(minutes=15))

            for date in mean_time:
                if date == current_date:
                    mean_date = current_date
                    mean_indx = date_indx
                    date_indx += 1
                elif date < small:
                    date_indx += 1
                elif large > date >= small:
                    mean_date = date
                    mean_indx = date_indx
                    date_indx += 1
                elif date > large:
                    date_indx += 1
                else:
                    print('Error in date calculation in "wake_depth"')

            # Go through the depths from 28 to 0, as 28 is the bin at the surface and 0 closest to the instrument.
            for depth in range(27, d_range, -1):
                current_depth = depth
                mean_amplitude = mean_amplitude_data[mean_indx, depth]

                # mean_amplitude_data is a XX x 28 matrix,  were XX is the number of hours the dataset spans
                # Raw data is compared to the average value and if the raw data is 15% higher, it is considered a peak
                # the date_indx gives the mean amplitude value that is "closes" to the raw_amplitude value.
                # I chose the value of 1.15 by trying which value best captured the wake I detect.
                if raw_amplitude_data[value, depth] > (factor * mean_amplitude):
                    wake = True

                elif raw_amplitude_data[value, depth] < (factor * mean_amplitude):
                    wake = False
                else:
                    print('Error in wake comparison in "wake_depth')

                current_list.append([current_date, current_depth, mean_date, mean_amplitude, wake])

        wake_info.append(current_list)

    return wake_info


def wake_depth_total_mean(raw_amplitude_data, total_mean_amp_corr, start_time, raw_time, d_range, factor, duration):
    """
    Calculates the maximum depth and longevity of the wakes for the time periods (wake occurrences) in the start_times
    list. Returns a list of list, where each list contains information about the wake (time, duration, maximum depth).
    :param raw_amplitude_data: The raw data of the mean echo amplitude that is calculated in the Matlab script.
    :param total_mean_amp_corr: The over all mean amplitude of the time series, which will be used to identify the wake
    signal, as the wake signal strength will be diverging strongly from the mean value for the time period. [1 x 28]
    :param start_time: The time and date to start the wake-calculation. The format is datetime objects.
    :param raw_time: The time vector that goes with the raw_amplitude_data (high resolution), in datetime format.
    :param d_range: The max depth for which the wake is calculated. An integer representing a depth cell.
    :param factor: The percentage that the wake signal must be compared to the average to be considered a wake.
    :param duration: The number of time-steps to be added to the starting time. Defines the duration of the calculation.
    :return: A list containing the following information:
            0) current_date:
            1) current_depth:
            2) mean_date: The date
            3) mean_amplitude:
            4) wake: Indicates if the measured value is part of a wake. Can be either True or False.
            5) raw_amplitude_data: The raw data (signal strength, eps or uVar) related to each depth. Used for finding
               the max value in a later calculation.
            6) A list with the indices corresponding to the current raw data (used for plotting the wake later)
    """

    wake_info = []

    # The index() function will give the position (row, column) of the item that is indexed. ie. the position of the
    # start time for the wake. The column of the start value, will be the start row for the amplitude raw dataset.
    # start = raw_time.index(item) (it does not work as the time must be the exact time, and that will not happen)

    count = 0
    for r_date in raw_time:
        if r_date == raw_time[0] and r_date > start_time:
            if r_date < start_time + timedelta(minutes=5):
                start = count
                count += 1
            else:
                print('Error in finding start date, not in datset')

        elif r_date is start_time:
            start = count
            count += 1
        elif r_date is raw_time[-1]:
            if start_time > r_date:
                print('Error in finding start date in wake_depth, value not in range')
            elif raw_time[-2] < start_time < r_date:
                start = count
            else:
                pass
                # print('Error in finding start date in wake_depth, end of list')
        elif r_date > start_time < raw_time[count + 1]:
            count += 1
        elif r_date < start_time < raw_time[count + 1]:
            start = count
            count += 1
        elif r_date < start_time > raw_time[count + 1]:
            count += 1
        else:
            print('Error in finding start date in wake_depth')

    # There are 120 averaged measurements per hour --> adding 60 to the index would give measurement for 30 min
    # adding 90 would be 45 min, and 120 would give measurement for 1 hour
    for value in range(start, (start + duration)):
        current_date = raw_time[value]

        # Go through the depths from 28 to 0, as 28 is the bin at the surface and 0 closest to the instrument.
        for depth in range(27, d_range, -1):
            current_depth = depth

            # mean_amplitude_data is a XX x 28 matrix,  were XX is the number of hours the dataset spans
            # Raw data is compared to the average value and if the raw data is 15% higher, it is considered a peak
            # the date_indx gives the mean amplitude value that is "closes" to the raw_amplitude value.
            # I chose the value of 1.15 by trying which value best captured the wake I detect.
            if raw_amplitude_data[value, depth] > (factor * total_mean_amp_corr[0, depth]):
                wake = True

            elif raw_amplitude_data[value, depth] < (factor * total_mean_amp_corr[0, depth]):
                wake = False
            else:
                print('Error in wake comparison in "wake_depth')

            wake_info.append([current_date, current_depth, total_mean_amp_corr[0, depth], wake,
                              raw_amplitude_data[value, current_depth], [current_date, current_depth]])

    return wake_info


def wake_matrix(raw_amplitude_data, total_mean_amp_corr, start_time, raw_time, d_range, factor, duration,
                parameter, ship_passage, depth_vector, ship_name):
    """
    Finds the wake area and creates a new matrix containing only the measurements in the wake area and the rest of the
    matrix is padded with zeroes. The matrix time values are normalised with respect to the time when the ship passes
    closest to the ADCP instrument. A normalised time vector and a depth vector is also made for plotting. When creating
    the normalised time vector, the time between measurements is assumed to be 30 s.
    :param raw_amplitude_data: The raw data of the mean echo amplitude that is calculated in the Matlab script.
    :param total_mean_amp_corr: The over all mean amplitude of the time series, which will be used to identify the wake
    signal, as the wake signal strength will be diverging strongly from the mean value for the time period. [1 x 28]
    :param start_time: The time and date to start the wake-calculation. The format is datetime objects.
    :param raw_time: The time vector that goes with the raw_amplitude_data (high resolution), in datetime format.
    :param d_range: The max depth for which the wake is calculated. An integer representing a depth cell.
    :param factor: The percentage that the wake signal must be compared to the average to be considered a wake.
    :param duration: The number of time-steps to be added to the starting time. Defines the duration of the calculation.
    :param parameter: A string indicating which parameter that is being analysed, as the datasets have different
                      temporal resolution.
    :param ship_passage: The time (datetime object) of the ship passage. Used to normalise the time in the wake matrix.
    :param depth_vector: The depth vector used to give the dimensions of the wake_matrix
    :param ship_name: The name of the ship [string].
    :return: Plots the wake (optional) and a list containing the following information:
            0) wake_matrix: A matrix containing the wake area with padded zeroes around the area. The time is normalised
               to the time when the ship passes closest to the instrument.
            1) normalised_time_vector: The normalized time vector that goes together with the matrix (for plotting)
            2) depth_vector: The depth vector that goes with the matrix (for plotting)
    """

    wake_info = []

    # The index() function will give the position (row, column) of the item that is indexed. ie. the position of the
    # start time for the wake. The column of the start value, will be the start row for the amplitude raw dataset.
    # start = raw_time.index(item) (it does not work as the time must be the exact time, and that will not happen)

    count = 0
    for r_date in raw_time:
        if r_date == raw_time[0] and r_date > start_time:
            if r_date < start_time + timedelta(minutes=5):
                start = count
                count += 1
            else:
                print('Error in finding start date, not in datset')

        elif r_date is start_time:
            start = count
            count += 1
        elif r_date is raw_time[-1]:
            if start_time > r_date:
                print('Error in finding start date in wake_depth, value not in range')
            elif raw_time[-2] < start_time < r_date:
                start = count
            else:
                pass
                # print('Error in finding start date in wake_depth, end of list')
        elif r_date > start_time < raw_time[count + 1]:
            count += 1
        elif r_date < start_time < raw_time[count + 1]:
            start = count
            count += 1
        elif r_date < start_time > raw_time[count + 1]:
            count += 1
        else:
            print('Error in finding start date in wake_depth')

    # There are 120 averaged measurements per hour --> adding 60 to the index would give measurement for 30 min
    # adding 90 would be 45 min, and 120 would give measurement for 1 hour

    start_wake = 0
    for value in range(start, (start + duration)):
        current_date = raw_time[value]
        #if current_date < ship_passage:
            # As there are a few cases where the ship wake is indicated as staring before the passage (witin
        #    normalised_date = current_date - current_date

        # The current date is normalised
        normalised_date = current_date - ship_passage

        # Go through the depths from 28 to 0, as 28 is the bin at the surface and 0 closest to the instrument.
        for depth in range(27, d_range, -1):
            current_depth = depth

            # mean_amplitude_data is a XX x 28 matrix,  were XX is the number of hours the dataset spans
            # Raw data is compared to the average value and if the raw data is 15% higher, it is considered a peak
            # the date_indx gives the mean amplitude value that is "closes" to the raw_amplitude value.
            # I chose the value of 1.15 by trying which value best captured the wake I detect.
            if raw_amplitude_data[value, depth] > (factor * total_mean_amp_corr[0, depth]):
                wake = True
                if start_wake is 0:
                    start_wake = normalised_date

            elif raw_amplitude_data[value, depth] < (factor * total_mean_amp_corr[0, depth]):
                wake = False
            else:
                print('Error in wake comparison in "wake_depth')

            wake_info.append([current_date, current_depth, total_mean_amp_corr[0, depth], wake,
                              raw_amplitude_data[value, current_depth], [current_date, current_depth], normalised_date])

    # The maximum duration is 75 for the bubble wake and 40 in the turbulent wakes. The water depth is 31.5 m.
    # The max lag-time between ship passage and wake is 00:15:24 [HH:MM:SS], so that will be added to the max duration.
    # As duration = 30 corresponds to 15 minutes, 31 will cover the entire lag-time period. --> 106 (bubble)

    # The first and last normalised date in the wake_info list are found
    first_normalised_wake_date = start_wake
    first_normalised_entry = wake_info[0][6]

    # The time steps to pad at the start of the wake is given by the even number of times the duration can be divided by
    # 30 s
    # I'm adding ten to compensate for early start of dataset and to keep the indices positive (all get an added 10
    index_start_column_wake = ((first_normalised_wake_date - (first_normalised_wake_date % timedelta(seconds=30))) /
                               timedelta(seconds=30)) + 10
    index_start_wake_list = ((first_normalised_entry - (first_normalised_entry % timedelta(seconds=30))) /
                             timedelta(seconds=30)) + 10

    matrix_depth = len(depth_vector)                # There are 28 depth bins
    matrix_duration = 75 + 31 + 10                  # I'm adding ten to compensate for early start of dataset

    wake_matrix = np.zeros((matrix_depth, matrix_duration))
    normalised_time_vector = [None] * matrix_duration

    current_column_index = index_start_wake_list
    last_depth = 32
    for item in wake_info:
        current_depth = item[1]
        if last_depth > current_depth:
            pass
        elif last_depth < current_depth:
            current_column_index += 1
        last_depth = current_depth

        # For each item in the list that is a wake, the value is added to the wake matrix
        if item[3] is True:
            row_indx = item[5][1]
            # The measured value of the wake is added to the matrix at the correct indices
            wake_matrix[(row_indx, int(current_column_index))] = item[4]
            normalised_time_vector[int(current_column_index)] = item[6]
            end_time_wake = item[6]
        else:
            normalised_time_vector[int(current_column_index)] = item[6]
            pass

    # When the data is in place, the 0-valued timedelta objects for the normalised time vector needs to be replaced
    # The entries before the wake start
    count = 0
    for entry in range(int(index_start_column_wake), -1, -1):
        normalised_time_vector[entry] = first_normalised_wake_date - (timedelta(seconds=30) * count)
        count += 1

    # Entries after the wake ends
    count = 0
    for entry in range(int(current_column_index), len(normalised_time_vector), 1):
        normalised_time_vector[entry] = end_time_wake + (timedelta(seconds=30) * count)
        count += 1

    # Check if the start time of the matrix is close to 00:00:00 and that the end is close to 50 min.

    count = 0
    for item in normalised_time_vector:
        normalised_time_vector[count] = item.total_seconds() / 60           # Will give duration in minutes
        count += 1

    # TODO: Plotting! Uncomment to plot the wake-matrix
    """
    if parameter == 'amplitude':
        fig, cs = plt.subplots(figsize=(19, 9))
        cs = plt.contourf(normalised_time_vector, depth_vector, wake_matrix)
        plt.colorbar()

        plt.axvline(x=0, color='w')
        plt.title('Ship wake for PCA analysis with padded zeroes (' + ship_name + ') ' + ship_passage.strftime('%d %B %Y %X'))
        plt.xlabel('Minutes passed since ship passage (white line)')
        plt.ylabel('Depth [m]')
        fig_name = '{0}_{1}_bubble'.format(ship_passage.strftime('%m_%d_%H_%M'), ship_name)

    elif parameter == 'uVar' or 'eps':
        if parameter == 'uVar':
            levels = np.logspace(-3.5, 0, 70)
            title = 'Velocity variance [m/s] (log scale)'
            tick_scale = [10**-4, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**0]
            fig_name = '{0}_{1}_uVar'.format(ship_passage.strftime('%m_%d_%H_%M'), ship_name)
        elif parameter == 'eps':
            levels = np.logspace(-6, -1, 70)
            title = 'Turbulent kinetic energy dissipation rate (\u03B5) [m\u00b2/s\u00b3] (log scale)'
            tick_scale = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3,  10 ** -2, 10 ** -1]
            # fig_name = 'eps_2018_{0}'.format(time_vector[0].strftime('%m_%d_%H_%M'))
            fig_name = '{0}_{1}_eps'.format(ship_passage.strftime('%m_%d_%H_%M'), ship_name)

        fig, cs = plt.subplots(figsize=(19, 9))
        cs = plt.contourf(normalised_time_vector, depth_vector, wake_matrix, levels, cmap='jet', locator=ticker.LogLocator(),
                          figsize=(20, 30))

        plt.gcf().autofmt_xdate()
        plt.title('Ship wake for PCA analysis with padded zeroes ' + title + ' (' + ship_name + ') ' + ship_passage.strftime('%d %B %Y %X'))
        plt.xlabel('Minutes passed since ship passage (black line)')
        plt.ylabel('Depth [m]')
        ax = plt.gca()
        plt.axvline(x=0, color='k')

        cs.changed()
        plt.colorbar(cs, ax=ax, ticks=tick_scale)

    plt.show()
    #plt.savefig(fig_name + '_beam5.png', bbox_inches='tight')
    #plt.close(fig)
    """
    return [wake_matrix, normalised_time_vector, depth_vector]


def plotting_wake_calculation_data(raw_data, time_vector, depth, max_depth_calculation_list, title):
    """
    Plots the raw data used for the wake calculation and white markers induícating the start, end and max-depth of wake.
    :param raw_data: The matrix with the raw data used for the contourf-plot, used as the z-parameter.
    :param time_vector: The time vector belonging to the raw_data, used as the x-parameter.
    :param depth: The depth vector used as the y-variable.
    :param title: A string with the name of the parameter the plot is for. [string]
    :param max_depth_calculation_list: An output list from the maxdepth_longevity_calculation.
    :return:
    """

    # Plotting the corrected data as a contour plot, where z is the echo amplitude signal strength
    cs = plt.contourf(time_vector, depth, np.transpose(raw_data), np.arange(35, 90, 0.5), cmap='jet')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.ylabel('Depth [m]')
    ax = plt.gca()

    plt.title('{0} ({1})'.format(title, time_vector[0].strftime('%B %Y')))
    formatter = DateFormatter('%H:%M %Y-%m-%d')
    # formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    # ax.frmt_xdata = mdate.DateFormatter('%m-%d')

    cs.changed()
    plt.colorbar()

    # item = [max_val, max_depth, max_list, start_time, end_time, longevity]
    # item[2] = [current_date, current_depth, mean_date, mean_amplitude, wake]
    for item in max_depth_calculation_list:
        # timedelta objects only have days, seconds and microseconds as units, thus minutes must be calculated
        # time_min = str(item[5].seconds / 60)
        # plt.scatter(item[2][0], -item[1], c='w', marker='_')
        # plt.scatter(item[3], -4, c='w', marker=8)
        # plt.scatter(item[4], -4, c='w', marker=9)

        plt.scatter(item[2][0], -item[1], c='w', marker='_',
                    label='Max wake depth = {0} m \nWake longevity = {1}'
                          ' [h:mm:ss]'.format(str(-item[1]), str(item[5]).split('.', 2)[0]))
        plt.scatter(item[3], -4, c='w', marker=8)
        plt.scatter(item[4], -4, c='w', marker=9)

    plt.legend()
    plt.show()


def plotting_zoom_bubble_wake(raw_data, time_vector, depth, max_depth_calculation_list, title, ship_name):
    """
    Plots the raw data used for the wake calculation and white markers induícating the start, end and max-depth of wake.
    :param raw_data: The matrix with the raw data used for the contourf-plot, used as the z-parameter.
    :param time_vector: The time vector belonging to the raw_data, used as the x-parameter.
    :param depth: The depth vector used as the y-variable.
    :param title: A string with the name of the parameter the plot is for. [string]
    :param max_depth_calculation_list: An output list from the maxdepth_longevity_calculation. A list of one list.
    :param ship_name: The name of the ship [string].
    :return:
    """

    # Plotting the corrected data as a contour plot, where z is the echo amplitude signal strength
    start_wake = max_depth_calculation_list[0][3] - timedelta(minutes=30)
    end_wake = max_depth_calculation_list[0][4] + timedelta(minutes=30)
    fig_name = 'wake_2018_{0}_{1}'.format(max_depth_calculation_list[0][3].strftime('%m_%d_%H_%M'), ship_name)

    fig, cs = plt.subplots(figsize=(19, 5))     #(19, 9)
    cs = plt.contourf(time_vector, depth, np.transpose(raw_data), np.arange(35, 90, 0.5), cmap='jet')
    plt.xlim((start_wake, end_wake))

    plt.gcf().autofmt_xdate()
    #plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.xlabel('Time [HH:MM]', color='w', fontsize=20)
    plt.ylabel('Depth [m]', color='w', fontsize=20)
    ax = plt.gca()

    ax.spines["top"].set_color('k')
    ax.spines["right"].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.tick_params(axis='x', colors='w', labelsize=15)
    ax.tick_params(axis='y', colors='w', labelsize=15)

    #plt.title('{0} ({1}) {2}'.format(title, max_depth_calculation_list[0][3].strftime('%B %Y %X'), ship_name))
    plt.title('{0} ({1})'.format(title, max_depth_calculation_list[0][3].strftime('%d %B %Y')), color='w', fontsize=25)
    #formatter = DateFormatter('%H:%M %Y-%m-%d')
    formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    # ax.frmt_xdata = mdate.DateFormatter('%m-%d')

    cs.changed()
    cbar = plt.colorbar()
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=15, color='w')
    cbar.ax.yaxis.set_tick_params(color='w')
    cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='w')
    cs.changed()

    # item = [max_val, max_depth, max_list, start_time, end_time, longevity]
    # item[2] = [current_date, current_depth, mean_date, mean_amplitude, wake]
    #for item in max_depth_calculation_list:
        # timedelta objects only have days, seconds and microseconds as units, thus minutes must be calculated
        # time_min = str(item[5].seconds / 60)
        # plt.scatter(item[2][0], -item[1], c='w', marker='_')
        # plt.scatter(item[3], -4, c='w', marker=8)
        # plt.scatter(item[4], -4, c='w', marker=9)

        #plt.scatter(item[2][0], -item[1], c='w', marker='_',
         #           label='Max wake depth = {0} m \nWake longevity = {1}'
          #                ' [h:mm:ss]'.format(str(-item[1]), str(item[5]).split('.', 2)[0]))
        #plt.scatter(item[3], -4, c='w', marker=8)
        #plt.scatter(item[4], -4, c='w', marker=9)

    #plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(fig_name + '_beam5.png', bbox_inches='tight', pad_inches=0, frameon=False, transparent=True)
    #plt.show()
    plt.close(fig)


def maxdepth_longevity_calculation(wake_info_list):
    """
    Calculates the maximum depth of the bubble wake and the longevity of the wake.
    :param wake_info_list: Has the same format as the wake_info returned from wake_depth_total_mean function.
    [current_date, current_depth, mean_amplitude, wake: Can be either True or False, .]
    :return: A list with the following parameters:
            0) max_depth_bin: The depth value at the deepest point of the bubble wake. As the values are in the opposite order
             (from instrument to surface, the real depth needs to be calculated). [m]
            1) max_depth: The calculated maximum depth of the wake. Negative value [m]
            2) max_list: This is the same as the max_depth_bin parmeter and should be removed.
            3) start_time: The start time of the wake. [Datetime]
            4) end_time: The end time of the wake. [Datetime]
            5) longevity: The longevity of the wake in minutes. [Datetime]
            6) max_intensity: The maximum intensity of the parameter in the wake (ie. echo amplitude, eps or uVar).
            7) max_intensity_coords: The [time, depth] coordinates for the maximum intensity value (used for plotting).
            8) intensity_max_depth: The value at the max wake depth (used for plotting)
    """

    # The list that will be returned in the end, containing relevant wake info.
    new_list = []
    wake_measurements = []

    # Each item in the wake_info_list is a list with entries from one ship passage
    count = 0           # If count == 0, there are no wake entries and the calculation cannot be performed
    for wake_list_entry in wake_info_list:

        # This list will contain all measurements where wake = True
        if wake_list_entry[3] is True:
            wake_measurements.append(wake_list_entry)
            count += 1
        elif wake_list_entry[3] is False:
            pass
        else:
            print('Error in maxdepth_logevity_calculation when dividing the True/False entries')

    # FINDING THE MAXIMUM WAKE DEPTH
    max_depth_bin = 27
    intensity_max_depth = 0
    for value in wake_measurements:
        if value[1] <= max_depth_bin:
            max_depth_bin = value[1]
            intensity_max_depth = value[4]
            max_list = value

        elif value[1] > max_depth_bin:
            pass
        else:
            print('Error in maxdepth_logevity_calculation when finding max depht')

        instrument_dept = 32            # [m]
        bin_size = 1                    # [m]
        cell_number = 1                 # The first cell is the one closest to the instrument, thus the deepest one
        max_cell_depth = instrument_dept - (0.5 + cell_number * bin_size)
        # As bin 0 = closes to the instrument, the difference between the max_cell_depth and the number of the cell
        # (max_depth_bin) will correspond to the depth of the centre of the deepest cell where the wake is visible
        max_depth = max_cell_depth - max_depth_bin

    # CALCULATE THE MAXIMUM VALUE/INTENSITY IN THE WAKE (PRIMARILY INTERESTING FOR eps AND uVar).
    max_intensity = 0
    for value in wake_measurements:
        if value[4] >= max_intensity:
            max_intensity = value[4]
            depth_intensity = 30.5 - value[1]        # max_cell_depth = instrument_dept - (0.5 + cell_number * bin_size)
            max_intensity_coords = [value[0], depth_intensity]

        elif value[4] < max_intensity:
            pass
        else:
            print('Error in maxdepth_logevity_calculation when finding max value')

    # CALCULATING THE START, END, AND LONGEVITY OF THE WAKE

    start_time = wake_measurements[0][0]
    end_time = wake_measurements[-1][0]
    longevity = end_time - start_time

    # print(max_depth_bin, max_depth, max_list, start_time, end_time, longevity)
    new_list.append([max_depth_bin, max_depth, max_list, start_time, end_time, longevity, max_intensity,
                     max_intensity_coords, intensity_max_depth])

    # TODO: Chane this and the following code so the output is a list, not a list of lists. The plotting in
    #  wake_depth_calculation does not work if I change it now

    return new_list


def save_csv_file_wake_info(filename, wake_list):
    """
    Takes the wake_list and writes a csv-file with information about each ship passage/track.
    :param filename: The name of the csv-file that is created and saved.
    :param wake_list: A list containing wake objects.
    :return: A saved cvs-file of the same name as the filename, containing the following information for each wake
    (header included), as well as a list of list where each list contains the info written to the csv file:
            0) The date the ship wake starts, string with format [YYYY-MM-DD HH:MM:SS]
            1) Quality markers indicating the uncertainty in the data used in the calculations
            2) The name of the ship(s) that created the wake [str]
            3) Distance to the vertical beam of the instrument in meters.
            4) Speed over ground in knots
            5) Ship type [str]
            6) Static draught of the ship [m]
            7) Current draught of the ship [m]
            8) Length of the ship [m]
            9) Width of the ship [m]
            10) Course over ground (cog) of the ship [degrees]
            11) Maximum depth of the bubble wake [m]
            12) The longevity of the bubble wake [minutes]
            13) The longevity of the bubble wake [m]
            14) Estimated stratification depth [m]
            15) Maximum turbulence intensity/dissipation rate [speed]
            16) Maximum depth of the turbulent wake [m]
            17) Longevity of eps wake [m]
            18) Max uVar intensity
            19) Max depth uVar wake
            20) Longevity uVar wake [min]
            21) Longevity uVar wake [m]
            22) Mean current speed ship dir (top 5m) [m/s]
            23) Mean current speed instrument dir (top 5m) [m/s]
            24) Mean current speed ship dir (5-10m) [m/s]
            25) Mean current speed instrument dir (5-10m) [m/s]
            26) Mean current speed ship dir (10-15m) [m/s]
            27) Mean current speed instrument dir (10-15m) [m/s]
            28) Mean current speed ship dir (15-20m) [m/s]
            29) Mean current speed instrument dir (15-20m) [m/s]
            30) Mean current speed ship dir (20-25m) [m/s]
            31) Mean current speed instrument dir (20-25m) [m/s]
            32) Indicating if it is arrival or departure [can be None]
            33) The deepest depth included in the wake calculation [m]
            34) The percentage used in the wake calculation to indicate what deviation from the average value that
                indicates a wake. It is kept as high as possible, but varies.
            35) The filename of the file used in the wake and stratification calculation.
            36) Which case that has been used in the calculation (A or B)
            37) List with average ship and instrument current directions for different depths [0-5, 5-10, 10-15, 15-30, 20-25] (depths are all + ca 3 m due to blanking distance.
    """
    final_master_list = []

    with open(filename, 'w', newline='') as outfile:  # The newline='' removes the blank line after every row

        csvWriter = writer(outfile)
        csvWriter.writerow(['Wake start date [YYYY-MM-DD HH:MM:SS]', 'Quality markers', 'Ship(s) name',
                            'Distance to instrument [m] (vertical beam)', 'Speed over ground [knots]',
                            'Ship type', 'Static draught [m]', 'Current draught [m]', 'Length', 'Width',
                            'Course over ground [Degrees]', 'Max depth bubble wake', 'Wake longevity bubble wake [min]',
                            'Wake longevity bubble wake [m]', 'Estimated stratification depth',
                            'Max eps intensity', 'Max depth eps wake', 'Longevity eps wake [min]',
                            'Longevity eps wake [m]', 'Max uVar intensity', 'Max depth uVar wake',
                            'Longevity uVar wake [min]', 'Longevity uVar wake [m]',
                            'MCS ship (top 5m) [m/s]', 'MCS instrument (top 5m) [m/s]', 'MCS ship (5-10m) [m/s]',
                            'MCS instrument (5-10m) [m/s]', 'MCS ship (10-15m) [m/s]',  'MCS instrument (10-15m) [m/s]',
                            'MCS ship (15-20m) [m/s]', 'MCS instrument (15-20m) [m/s]', 'MCS ship (20-25m) [m/s]',
                            'MCS instrument (20-25m) [m/s]', 'Arrival/Departure', 'Min depth wake calculation',
                            'Percentage', 'Filename wake and stratification calculation',
                            'Case [A or B]'])

        final_master_list.append(['Wake start date [YYYY-MM-DD HH:MM:SS]', 'Quality markers', 'Ship(s) name',
                                  'Distance to instrument [m] (vertical beam)', 'Speed over ground [knots]',
                                  'Ship type', 'Static draught [m]', 'Current draught [m]', 'Length', 'Width',
                                  'Course over ground [Degrees]', 'Max depth bubble wake',
                                  'Wake longevity bubble wake [min]', 'Wake longevity bubble wake [m]',
                                  'Estimated stratification depth', 'Max eps intensity', 'Max depth eps wake',
                                  'Longevity eps wake [min]', 'Longevity eps wake [m]', 'Max uVar intensity',
                                  'Max depth uVar wake', 'Longevity uVar wake [min]', 'Longevity uVar wake [m]',
                                  'Current speed [m/s] ship dir', 'Current speed [m/s] instrument dir',
                                  'Arrival/Departure', 'Min depth wake calculation', 'Percentage',
                                  'Filename wake and stratification calculation', 'Case [A or B]',
                                  'Mean current speed (ship, instrument) different depths [m/s]'])

        for item in wake_list:

            # TODO: At some point the beam can be given in the input file, so that different beams can be used.
            # The input for the wake depth calculation is the beam you want to analyze, as a string
            item.wake_depth_calculation('a5m')

            ships = []
            if 'two ships' in item.quality_markers:
                if 'both' in item.quality_markers:
                    ships.append([(item.ship_info[0].name, item.ship_info[1].name)])
                else:
                    ships.append([item.ship_info[0].name])
            else:
                ships.append([item.ship_info[0].name])

            # As some wakes have two ships that possibly could have created the wake, there are two possible cases. The
            # A-case writes a file with the data from the first ship for ALL wakes. The B-case writes a file with the
            # data for the second ship for  all cases that have the quality marker "both" (as that indicates that both
            # ships could have created the wake).

            if 'both' in item.quality_markers:
                ships = ships[0][0]
            else:
                ships = ships[0]

            str_full_list_a = [str(item.wake_start_time.strftime('%Y-%m-%d  %H:%M:%S')), str(item.quality_markers),
                               str(ships[0]), str(item.ship_info[0].distance[0][0]), str(item.ship_info[0].sog),
                               item.ship_info[0].ship_type_t, str(item.ship_info[0].static_draught),
                               str(item.ship_info[0].current_draught), str(item.ship_info[0].length),
                               str(item.ship_info[0].width), str(item.ship_info[0].cog), str(item.max_wake_depth),
                               str(item.longevity_time).split('.', 2)[0], str(round(item.longevity_distance[0])),
                               str(item.stratification_depth), str(round(item.max_eps, 5)), str(item.max_depth_eps),
                               str(item.eps_longevity_time).split('.', 2)[0],
                               str(round(item.eps_longevity_distance[0])),
                               str(round(item.max_uVar, 3)), str(item.max_depth_uVar),
                               str(item.uVar_longevity_time).split('.', 2)[0],
                               str(round(item.uVar_longevity_distance[0])), str(round(item.ship_current_vector[0][0], 4)),
                               str(round(item.ship_current_vector[0][1], 4)), str(round(item.ship_current_vector[1][0], 4)),
                               str(round(item.ship_current_vector[1][1], 4)), str(round(item.ship_current_vector[2][0], 4)),
                               str(round(item.ship_current_vector[2][1], 4)), str(round(item.ship_current_vector[3][0], 4)),
                               str(round(item.ship_current_vector[3][1], 4)), str(round(item.ship_current_vector[4][0], 4)),
                               str(round(item.ship_current_vector[4][1], 4)), str(item.ship_info[0].arr_dep), str(item.min_depth),
                               str(item.percentage), item.filename, 'Case A']

            full_list_a = [item.wake_start_time.strftime('%Y-%m-%d  %H:%M:%S'), item.quality_markers,
                           ships[0], item.ship_info[0].distance[0][0], item.ship_info[0].sog,
                           item.ship_info[0].ship_type_t, item.ship_info[0].static_draught,
                           item.ship_info[0].current_draught, item.ship_info[0].length, item.ship_info[0].width,
                           item.ship_info[0].cog, item.max_wake_depth, item.longevity_time, item.longevity_distance[0],
                           item.stratification_depth, item.max_eps, item.max_depth_eps, item.eps_longevity_time,
                           item.eps_longevity_distance[0], item.max_uVar, item.max_depth_uVar, item.uVar_longevity_time,
                           item.uVar_longevity_distance[0], item.current_ship_dir, item.current_instrument_dir,
                           item.ship_info[0].arr_dep, item.min_depth, item.percentage, item.filename, 'Case A',
                           item.ship_current_vector]

            # [0-5, 5-10, 10-15, 15-30, 20-25]

            if 'two ships' in item.quality_markers:
                if 'both' in item.quality_markers:
                    full_list_b = [item.wake_start_time.strftime('%Y-%m-%d  %H:%M:%S'), item.quality_markers,
                                   ships[1], item.ship_info[1].distance[0][0], item.ship_info[1].sog,
                                   item.ship_info[1].ship_type_t, item.ship_info[1].static_draught,
                                   item.ship_info[1].current_draught, item.ship_info[1].length,
                                   item.ship_info[1].width, item.ship_info[1].cog, item.max_wake_depth,
                                   item.longevity_time, item.longevity_distance[1], item.stratification_depth,
                                   item.max_eps, item.max_depth_eps, item.eps_longevity_time,
                                   item.eps_longevity_distance[1], item.max_uVar, item.max_depth_uVar,
                                   item.uVar_longevity_time, item.uVar_longevity_distance[1], item.current_ship_dir,
                                   item.current_instrument_dir, item.ship_info[1].arr_dep, item.min_depth,
                                   item.percentage, item.filename, 'Case B', item.ship_current_vector]

                    str_full_list_b = [str(item.wake_start_time.strftime('%Y-%m-%d  %H:%M:%S')), str(item.quality_markers),
                                       str(ships[1]), str(item.ship_info[1].distance[0][0]), str(item.ship_info[1].sog),
                                       item.ship_info[1].ship_type_t, str(item.ship_info[1].static_draught),
                                       str(item.ship_info[1].current_draught), str(item.ship_info[1].length),
                                       str(item.ship_info[1].width), str(item.ship_info[1].cog),
                                       str(item.max_wake_depth), str(item.longevity_time).split('.', 2)[0],
                                       str(round(item.longevity_distance[1])), str(item.stratification_depth),
                                       str(round(item.max_eps, 5)), str(item.max_depth_eps),
                                       str(item.eps_longevity_time).split('.', 2)[0],
                                       str(round(item.eps_longevity_distance[1])), str(round(item.max_uVar, 3)),
                                       str(item.max_depth_uVar), str(item.uVar_longevity_time).split('.', 2)[0],
                                       str(round(item.uVar_longevity_distance[1])), str(round(item.ship_current_vector[0][0], 4)),
                                       str(round(item.ship_current_vector[0][1], 4)), str(round(item.ship_current_vector[1][0], 4)),
                                       str(round(item.ship_current_vector[1][1], 4)), str(round(item.ship_current_vector[2][0], 4)),
                                       str(round(item.ship_current_vector[2][1], 4)), str(round(item.ship_current_vector[3][0], 4)),
                                       str(round(item.ship_current_vector[3][1], 4)), str(round(item.ship_current_vector[4][0], 4)),
                                       str(round(item.ship_current_vector[4][1], 4)), str(item.ship_info[1].arr_dep),
                                       str(item.min_depth), str(item.percentage), item.filename, 'Case B']

                    csvWriter.writerow(str_full_list_a)
                    final_master_list.append(full_list_a)
                    csvWriter.writerow(str_full_list_b)
                    final_master_list.append(full_list_b)
                else:
                    csvWriter.writerow(str_full_list_a)
                    final_master_list.append(full_list_a)

            else:
                csvWriter.writerow(str_full_list_a)
                final_master_list.append(full_list_a)

    return final_master_list


def plotting_wake_calculation_data_log_single(raw_data, time_vector, depth, max_list, parameter, ship_name):
    """
    Plots the raw data used for the wake calculation and white markers induícating the start, end and max-depth of wake.
    :param raw_data: The matrix with the raw  data used for the contourf-plot, used as the z-parameter.
    :param time_vector: The time vector belonging to the raw_data, used as the x-parameter.
    :param depth: The depth vector used as the y-variable.
    :param max_list: An output list from the maxdepth_longevity_calculation
    :param parameter: Indicates which of the parameters esp and uVar that is plotted.
    :param ship_name: A string indicating the name of the ship that made the wake.
    :return:
    """

    if parameter == 'uVar':
        levels = np.logspace(-3.5, -1, 70)
        raw_data = np.transpose(raw_data)
        title = 'Velocity variance [m/s]' # (log scale)'
        tick_scale = [10**-4, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1]
        colour = 'k'
        unit = 'm/s'
        fig_name = 'uVar_2018_{0}'.format(time_vector[0].strftime('%m_%d_%H_%M'))
    elif parameter == 'eps':
        levels = np.logspace(-6, -3, 70)
        title = 'Turbulent kinetic energy dissipation rate (\u03B5) [m\u00b2/s\u00b3]' # (log scale)'
        tick_scale = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]
        colour = 'k'
        unit = '[m\u00b2/s\u00b3]'
        fig_name = 'eps_2018_{0}'.format(time_vector[0].strftime('%m_%d_%H_%M'))

    fig, cs = plt.subplots(figsize=(19, 5))        # (18, 10)
    cs = plt.contourf(time_vector, depth, raw_data, levels, cmap='jet', locator=ticker.LogLocator())     # (20, 30)

    plt.gcf().autofmt_xdate()
    #plt.xlabel('Time and date (HH:MM yyyy-mm-dd)', color='w', fontsize=15)
    plt.xlabel('Time [HH:MM]', color='w', fontsize=20)
    plt.ylabel('Depth [m]', color='w', fontsize=20)
    ax = plt.gca()

    ax.spines["top"].set_color('k')
    ax.spines["right"].set_color('k')
    ax.spines['bottom'].set_color('k')
    #plt.setp(ax.spines.values(), linewidth=2)
    ax.spines['left'].set_color('k')
    ax.tick_params(axis='x', colors='w', labelsize=15)
    ax.tick_params(axis='y', colors='w', labelsize=15)

    #plt.title('{0} ({1}) {2}'.format(title, time_vector[0].strftime('%d %B %Y %X'), ship_name), color='k', fontsize=20)
    plt.title('{0} ({1})'.format(title, time_vector[0].strftime('%d %B %Y')), color='w', fontsize=25)
    #formatter = DateFormatter('%H:%M %Y-%m-%d')
    formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

    cs.changed()
    cbar = plt.colorbar(cs, ax=ax, ticks=tick_scale)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=15, color='w')
    cbar.ax.yaxis.set_tick_params(color='w')
    cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='w')
    cs.changed()

    """
    # item = [max_val, max_depth, max_list, start_time, end_time, longevity, max_intensity, max_intensity_coords]
    # item[2] = [current_date, current_depth, mean_amplitude, wake]
    for item in max_list:
        # timedelta objects only have days, seconds and microseconds as units, thus minutes must be calculated
        # time_min = str(item[5].seconds / 60)
        plt.scatter(item[2][0], -item[1], c=colour, marker='_',
                    label='Value max depth = {0} {1} \nMax wake depth = {2} m \nWake longevity = {3}'
                          ' [h:mm:ss]'.format(str(round(item[8], 5)), unit, str(-item[1]), str(item[5]).split('.', 2)[0]))
        plt.scatter(item[3], -4, c='k', marker=8)
        plt.scatter(item[4], -4, c=colour, marker=9)
        plt.scatter(item[7][0], -item[7][1], c='k', marker='+',
                    label='Max value = {0} {1} \nDepth max value = {2} m'.format(str(round(item[6], 5)),
                                                                                unit, str(-item[7][1])))
    """
    #plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)
    #plt.show()
    # print(fig_name + '_corr.png')
    plt.savefig(fig_name + '_corr.png',  bbox_inches='tight',  pad_inches=0, frameon=False, transparent=True)
    #plt.show()
    plt.close(fig)


def plotting_wake_area(wake_depth_list, python_time, depth_z, raw_data, parameter, ship_name):
    """
    Plots the raw_data in a countourf plot and on top of that it plots the wake area as black circles (wake) and the
    non-wake area as white circles. Can handle the parameters amplitude, epsilon (eps) and uVar.
    :param wake_depth_list: Output list from the function "wake_depth_total_mean".
    :param python_time: Time vector in python datetime format.
    :param depth_z: The depth vector
    :param raw_data: The raw data used for the contourf plot.
    :param parameter: A string giving the parameter to plot. Will effect the scale on the figure axes.
    :param ship_name: A string indicating the name of the ship that made the wake.
    :return: A contourf plot indicating the wake area.
    """

    # PLOTTING THE WAKE AREA TO SEE IF IT IS A GOOD MATCH
    wake_measurements = []
    non_wake = []

    # Plotting the wake to check how well it is defined
    # Each item in the wake_info_list is a list with entries from one ship passage
    for wake_list_entry in wake_depth_list:

        # This list will contain all measurements where wake = True
        if wake_list_entry[3] is True:
            wake_measurements.append(wake_list_entry)
        elif wake_list_entry[3] is False:
            non_wake.append(wake_list_entry)

    if parameter is 'amplitude':
        cs = plt.contourf(python_time, depth_z, np.transpose(raw_data), np.arange(35, 90, 0.5), cmap='jet')

        plt.gcf().autofmt_xdate()
        plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
        plt.ylabel('Depth [m]')
        ax = plt.gca()

        plt.title('{0} ({1}) {2}'.format('Echo amplitude signal strength', python_time[0].strftime('%B %Y'), ship_name))
        formatter = DateFormatter('%H:%M %Y-%m-%d')

        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

        cs.changed()
        plt.colorbar()

        # item = [max_val, max_depth, max_list, start_time, end_time, longevity]
        # item[2] = [current_date, current_depth, mean_date, mean_amplitude, wake]
        for item in wake_measurements:
            plt.scatter(item[5][0], -(30.5 - item[5][1]), c='k', alpha=0.2, marker='.')

        for item in non_wake:
            plt.scatter(item[5][0], -(30.5 - item[5][1]), c='w', alpha=0.3, marker='.')

    else:

        if parameter == 'uVar':
            levels = np.logspace(-3.5, -1, 70)
            raw_data = np.transpose(raw_data)
            title = 'Velocity variance [m/s] (log scale)'
            tick_scale = [10 ** -4, 10 ** -3, 10 ** -2.5, 10 ** -2, 10 ** -1.5, 10 ** -1]
        elif parameter == 'eps':
            levels = np.logspace(-6, -3, 70)
            title = 'Turbulent kinetic energy dissipation rate (\u03B5) [m\u00b2/s\u00b3] (log scale)'
            tick_scale = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]

        cs = plt.contourf(python_time, depth_z, raw_data, levels, cmap='jet', locator=ticker.LogLocator())

        plt.gcf().autofmt_xdate()
        plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
        plt.ylabel('Depth [m]')
        ax = plt.gca()

        plt.title('{0} ({1}) {2}'.format(title, python_time[0].strftime('%B %Y'), ship_name))
        formatter = DateFormatter('%H:%M %Y-%m-%d')

        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

        cs.changed()
        plt.colorbar(cs, ax=ax, ticks=tick_scale)

        for item in wake_measurements:
            plt.scatter(item[5][0], -(30.5 - item[5][1]), c='k', alpha=0.2, marker='.')

        for item in non_wake:
            plt.scatter(item[5][0], -(30.5 - item[5][1]), c='w', alpha=0.3, marker='.')

    plt.show()


def plotting_windrose(north_current, east_current, ship_current, instrument_current, ship_name, title):
    """
    Plots the windrose data, both the N,E wind and the ship,instrument direction. In the second N = the direction away
    from the instrument, E = oposite direction of the ship's traveling direction
    :param north_current:
    :param east_current:
    :param ship_current:
    :param instrument_current:
    :param ship_name: A string indicating the name of the ship that made the wake.
    :param title: String indicating what data that is plotted.
    :return:
    """

    current_direction_NE = np.zeros((1, len(east_current[0])))
    current_speed_NE = np.zeros((1, len(east_current[0])))
    current_direction_ship = np.zeros((1, len(east_current[0])))
    current_speed_ship = np.zeros((1, len(east_current[0])))

    for item in range(0, len(east_current[0])):
        # The wind speed is the magnitude/norm of the vector defined by the north and east vector.
        # wind_speed[0][item] = np.sqrt(north_current[0][item]**2 + east_current[0][item]**2)
        current_speed_NE[0][item] = np.linalg.norm([north_current[0][item], east_current[0][item]])
        current_speed_ship[0][item] = np.linalg.norm([instrument_current[0][item], ship_current[0][item]])

        alpha = np.degrees(np.arctan2(north_current[0][item], east_current[0][item]))
        alpha_ship = np.degrees(np.arctan2(instrument_current[0][item], ship_current[0][item]))

        # Calculating the speed and direction for the NE data
        if alpha <= 90:
            # Quadrant I: alpha is positive (alpha < 90) and theta is given by 90 - theta
            # Quadrant II and III: alpha is negative (alpha < 0) and theta is given by 90 + positive alpha = 90 - -alpha
            # Therefore, the same calculation works for these three quadrants
            current_direction_NE[0][item] = 90 - alpha
        elif 90 < alpha <= 180:
            # Quadrant IV (theta = 360 + (90 - alpha) = 450 - alpha)
            current_direction_NE[0][item] = 450 - alpha
        else:
            print('Error in calculating angle for plotting windrose in NE coordinates')

        # Calculating the speed and direction for the ship-oriented data
        if alpha_ship <= 90:
            # Quadrant I: alpha is positive (alpha < 90) and theta is given by 90 - theta
            # Quadrant II and III: alpha is negative (alpha < 0) and theta is given by 90 + positive alpha = 90 - -alpha
            # Therefore, the same calculation works for these three quadrants
            current_direction_ship[0][item] = 90 - alpha_ship
        elif 90 < alpha_ship <= 180:
            # Quadrant IV (theta = 360 + (90 - alpha) = 450 - alpha)
            current_direction_ship[0][item] = 450 - alpha_ship
        else:
            print('Error in calculating angle for plotting windrose in ship-oriented coordinates')

    ax = WindroseAxes.from_ax()
    ax.bar(current_direction_NE[0], current_speed_NE[0], normed=True, bins=np.arange(0.0, 0.16, 0.02), cmap=cm.YlGnBu)
    # ax.bar(current_direction_NE[0], current_speed_NE[0], normed=True, opening=0.8, edgecolor='white')
    # ax.contourf(current_direction_NE[0], current_speed_NE[0], bins=np.arange(0.01, 0.5, 0.05), cmap=cm.hot)
    ax.set_legend()
    plt.title('N-E current speed {1} ({0})'.format(ship_name, title))
    plt.show()

    ax2 = WindroseAxes.from_ax()
    ax2.bar(current_direction_ship[0], current_speed_ship[0], normed=True, bins=np.arange(0.0, 0.16, 0.02),
            cmap=cm.YlGnBu)
    ax2.set_legend()
    plt.title('Current speed {1} (N=away from instrument, W = ship traveling direction) ({0})'.format(ship_name, title))
    plt.show()

    """
    ax = WindAxes.from_ax()
    ax, params = ax.pdf(current_speed_NE[0], bins=np.arange(0.0, 0.16, 0.02))
    plt.legend()
    plt.show()
    """


def current_calculation(east_current, north_current, ship_cog, ship_name, title):
    """
    Calculates the current speed in the traveling direction of the ship (x-plane) and the current speed towards/away
    from the ADCP instrument (y-plane). It is done by defining a rotated coordinate system and then transforming the
    east,north current vectors to x,y vectors. The angle of rotation (theta) is calculated using the ship'scourse over
    ground (ship-cog), which defines the new x-plane.
    :param east_current: The current vector in easterly direction (E,N coordinate system). Measured values [m/s].
    :param north_current: The current vector in northern direction (E,N coordinate system). Measured values [m/s].
    :param ship_cog: The ship's course over ground, which indicates the heading of the ship. [float, degrees, 0-360]
    :param ship_name: A string indicating the name of the ship that made the wake.
    :param title: String indicating what data that is plotted.
    :return: 1) current_ship_dir: The current speed in the traveling direction of the ship. Negative values indicates
                that the ship is traveling with the current and positive speeds indicates traveling against the current.
             2) current_instrument_dir: The current speed away from/towards the ADCP instrument. A negative speed
             indicates a current towards the instrument and a positive speeds indicates a current away from it.
    """

    # CALCULATE ANGLE OF ROTATION THETA
    # Define he angle between the 0/360 degrees and the ship_cog. It will be used to determine the angle of rotation
    alpha = 360 - round(ship_cog)

    # Theta is the angle of rotation. As the x-axis is in the plane of the ship_cog, and the North axis is the y-axis,
    # alpha represents the angle of rotation of the x-axis with respect to the North/y axis. To get the angle of
    # rotation of the y-axis 90 degrees have to be removed from the alpha angle.
    # theta = alpha - 90 = 360 - ship_cog - 90 = 270 - alpha

    # Short version
    theta = 270 - alpha

    # CALCULATING THE CURRENT SPEED IN THE SHIP'S TRAVELING DIRECTION (X) AND TOWARDS/FROM THE INSTRUMENT (Y)
    # As the new coordinate system is defined, the ship will be traveling against the current if the speed is positive
    # and with the current if the speed is negative. The same goes for the current speed towards/away from the
    # instrument. A positive speed indicates that the wake is being pushed away from the instrument by the currents, and
    # a negative value indicates a movement towards the instrument.
    current_ship_dir = np.zeros((1, len(east_current[0])))
    current_instrument_dir = np.zeros((1, len(east_current[0])))

    if theta is 0:
        # If there is no angle of rotation the E,N and X,Y axis are the same and so are the speed vectors.
        current_ship_dir = east_current
        current_instrument_dir = north_current
    else:
        # ROTATING/TRANSFORMING THE CURRENT VECTORS FROM THE E,N COORDINATE SYSTEM TO THE ROTATED X,Y COORDINATE SYSTEM
        for item in range(0, len(east_current[0])):
            current_ship_dir[0][item] = (east_current[0][item] * np.cos(np.radians(theta)) + north_current[0][item] *
                                         np.sin(np.radians(theta)))
            current_instrument_dir[0][item] = (-east_current[0][item] * np.sin(np.radians(theta)) +
                                               north_current[0][item] * np.cos(np.radians(theta)))

    # TODO: Uncomment this to plot the windroses
    # plotting_windrose(north_current, east_current, current_ship_dir, current_instrument_dir, ship_name, title)

    # Calculating the mean value
    current_ship_dir_mean = np.mean(current_ship_dir)
    current_instrument_dir_mean = np.mean(current_instrument_dir)

    if title is 'all depths':
        return current_ship_dir, current_instrument_dir
    else:
        return current_ship_dir_mean, current_instrument_dir_mean


