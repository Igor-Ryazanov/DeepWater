from datetime import datetime, timedelta
import csv
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from wake_calculation_3 import wake_depth, maxdepth_longevity_calculation, wake_depth_total_mean
from signal_correction_water import signal_correction_exp, halocline_depth

# Loading the matlabfile and saving the variables
# test_file = sio.loadmat('0829_day_pythonfile.mat')  # File with clear wakes. Test file which it works for!
# nr_hours = 11                                   # File with clear wakes

# The number of parts the hour is to be divided into. 2 = 30 min slots, which will give hourly mean
time_factor = 2


wake_list = []
with open('wake_calculation_for_csv.csv', 'r') as f:     # 'Ship_wakes_passages.csv'
    wake_file = csv.reader(f, delimiter=';')
    for line in wake_file:
        test = line[0]
        if line[0] == 'ï»¿Filename':
            pass
        elif line == '':
            pass
        else:
            wake_list.append([line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5]),
                              int(line[6]), int(line[7]), float(line[8]), int(line[9])])


# wake_list = [['day_08_29_pythonfile', 2018, 8, 29, 10, 0, 14, -1, 1.15, 65], ['day_08_29_pythonfile', 2018, 8, 29, 14, 50, 14, -1, 1.15, 90]]
# wake_list = [['day_09_09_pythonfile', 2018, 9, 9, 12, 50, 8, -1, 1.15, 50], ['day_09_09_pythonfile', 2018, 9, 9, 15, 50, 8, 10, 1.17, 40]]

for wake in wake_list:
    mat_file = wake[0] + '_pythonfile'
    start_date = datetime(wake[1], wake[2], wake[3], wake[4], wake[5])
    nr_hours = wake[6]
    min_depth_range = wake[7]
    percentage = wake[8]
    duration = wake[9]

    # Loading the data from the matlab file
    test_file = sio.loadmat(mat_file)
    a1m = test_file['a1m']
    a3m = test_file['a3m']
    a4m = test_file['a4m']
    a5m = test_file['a5m']
    tm = test_file['tm'][0]
    depth_z = test_file['z'][0]

    # To plot, the time vector needs to be transposed
    a1m_t = np.transpose(a1m)
    a5m_t = np.transpose(a5m)

    python_datetime = tm
    python_tm = []

    # Converting matlab-time to datetime-time.
    for item in python_datetime:
        python_tm.append(datetime.fromordinal(int(item)) + timedelta(days=item % 1) - timedelta(days=366))

    # TODO: Here you change the beam you want to study (first parameter)
    a5m_corr_exp, mean_a5m_corr_exp = signal_correction_exp(a5m, python_tm, depth_z)

    # Reshape the matrix to have 120 measurements in each section (the number of averages per hour = averaging
    # measurements for 0.5 minutes). 28 x 1320

    #h_a5m = np.reshape(a5m_corr_exp, (round(nr_hours * time_factor), round(120 / time_factor), 28))
    #h_python_tm = np.reshape(python_tm, (round(nr_hours * time_factor), round(120 / time_factor)))

    # wake_calculation = wake_depth(a1m, h_mean_a5m, start_date, python_tm, h_mean_python_tm, min_depth_range,
    # percentage, duration)

    wake_calculation_total_mean = wake_depth_total_mean(a5m_corr_exp, mean_a5m_corr_exp, start_date, python_tm,
                                                        min_depth_range, percentage, duration)

    max_depth_list = maxdepth_longevity_calculation(wake_calculation_total_mean)

    # Calculate the halocline depth using the derivative of the mean amplitude over depth

    # mean_signal_corr_derivative, strat_depth, strat_depth_point = halocline_depth(mean_a5m_corr_exp, depth_z)
    strat_depth, strat_depth_point = halocline_depth(mean_a5m_corr_exp, depth_z)

    """
    fig1 = plt.plot(np.transpose(mean_a5m_corr_exp), depth_z)
    plt.plot(strat_depth_point[0], strat_depth_point[1], 'kx', markersize=7, label='Est. stratification depth')
    plt.title('Signal strength')
    plt.legend()
    plt.show()


    # Plotting the raw data as a contour plot, where z is the echo amplitude signal strength
    cs = plt.contourf(python_tm, depth_z, np.transpose(a5m), np.arange(35, 90, 0.5), cmap='jet')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.ylabel('Depth [m]')
    ax = plt.gca()
    #plt.title('Echo Amplitude during the period ' + python_tm[0].strftime('%m/%d/%Y') + ' - ' + python_tm[-1].strftime('%m/%d/%Y'))

    plt.title('Raw Echo Amplitude (' + python_tm[0].strftime('%B %Y') + ')')
    formatter = DateFormatter('%H:%M %Y-%m-%d')
    #formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    #ax.frmt_xdata = mdate.DateFormatter('%m-%d')

    cs.changed()
    plt.colorbar()
    test = 4
    test2 = 4 + test
    legend = []
    # item = [max_val, max_depth, max_list, start_time, end_time, longevity]
    # item[2] = [current_date, current_depth, mean_date, mean_amplitude, wake]
    for item in max_depth_list:
        # timedelta objects only have days, seconds and microseconds as units, thus minutes must be calculated
        time_min = str(item[5].seconds / 60)
        plt.scatter(item[2][0], -item[1], c='w', marker='_')
        plt.scatter(item[3], -4, c='w', marker=8)
        plt.scatter(item[4], -4, c='w', marker=9)
    """

    plt.show()

    # Plotting the corrected data as a contour plot, where z is the echo amplitude signal strength
    cs = plt.contourf(python_tm, depth_z, np.transpose(a5m_corr_exp), np.arange(35, 90, 0.5), cmap='jet')

    plt.gcf().autofmt_xdate()
    plt.xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.ylabel('Depth [m]')
    ax = plt.gca()
    #plt.title('Echo Amplitude during the period ' + python_tm[0].strftime('%m/%d/%Y') + ' - ' + python_tm[-1].strftime('%m/%d/%Y'))

    plt.title('Corrected Echo Amplitude (' + python_tm[0].strftime('%B %Y') + ')')
    formatter = DateFormatter('%H:%M %Y-%m-%d')
    #formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    #ax.frmt_xdata = mdate.DateFormatter('%m-%d')

    cs.changed()
    plt.colorbar()
    test = 4
    test2 = 4 + test
    legend = []
    # item = [max_val, max_depth, max_list, start_time, end_time, longevity]
    # item[2] = [current_date, current_depth, mean_date, mean_amplitude, wake]
    for item in max_depth_list:
        # timedelta objects only have days, seconds and microseconds as units, thus minutes must be calculated
        time_min = str(item[5].seconds / 60)
        plt.scatter(item[2][0], -item[1], c='w', marker='_')
        plt.scatter(item[3], -4, c='w', marker=8)
        plt.scatter(item[4], -4, c='w', marker=9)

    plt.show()

