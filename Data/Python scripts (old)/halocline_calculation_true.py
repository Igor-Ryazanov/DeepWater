from datetime import datetime, timedelta
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math as m
from signal_correction_water import signal_correction_exp, halocline_depth
import csv
import pandas as pd
# from wake_calculation import wake_depth, maxdepth_longevity_calculation


# Loading the matlabfile and saving the variables

load_file = 'Ship_wakes_passages_day_night.csv'

filename_list = []
with open(load_file, 'r') as f:     # 'Ship_wakes_passages_day_night.csv'
    name_file = csv.reader(f, delimiter=';')
    for line in name_file:
        test = line[0]
        if line[0] == 'ï»¿filename':
            pass
        else:
            # Saving the filename [0], day [1], night [2] in the filename_list.
            filename_list.append([line[0], line[1], line[2]])

depth = []
mean_values = []
for name in filename_list:
    filename = name[0] + '_pythonfile'
    test_file = sio.loadmat(filename)

    a1m = test_file['a1m']
    a3m = test_file['a3m']
    a4m = test_file['a4m']
    a5m = test_file['a5m']
    tm = test_file['tm'][0]
    depth_z = test_file['z'][0]
    if name == filename_list[0]:
        depth = depth_z

    # To plot, the time vector needs to be transposed
    a1m_t = np.transpose(a1m)
    a5m_t = np.transpose(a5m)

    python_datetime = tm
    python_tm = []
    beam = a5m

    # Converting matlab-time to datetime-time.
    for item in python_datetime:
        python_tm.append(datetime.fromordinal(int(item)) + timedelta(days=item % 1) - timedelta(days=366))

    # Correct/normalize for the increased signal strength with increased depth
    # d = depth (so it should be the depth bin)
    # EA = the raw amplitude (at the certain depth)
    # Format of a5m 1760 x 28

    a5m_corr_log = np.zeros([int(len(beam)), 28])
    a5m_corr_exp = np.zeros([int(len(beam)), 28])

    count = 0
    for item in beam:
        for d in range(0, 28):
            depth = depth_z[d]

            # The correction value is subtracted from the original value to give a the data without the depth effect
            a5m_corr_log[count][d] = item[d] - (-20 * np.log10(depth + 31) + 2 * (depth + 31) * 0.08)       # Version that Lars had in an old script (no reference)
            a5m_corr_exp[count][d] = item[d] - 20 * m.exp(-(depth + 31) / 5)             # Exponential that Lars made up

        count += 1

    # Calculating the mean echo amplitude for the entire dataset
    mean_a5m_corr_log = np.zeros([1, 28])
    mean_a5m_corr_exp = np.zeros([1, 28])
    mean_beam_raw = np.zeros([1, 28])
    h_mean_python_tm = []

    for d in range(0, 28):
        # mean_a5m_corr_log[0][d] = np.mean(a5m_corr_log[:, d])
        mean_a5m_corr_exp[0][d] = np.mean(a5m_corr_exp[:, d])
        mean_beam_raw[0][d] = np.mean(beam[:, d])

    # Saves the raw mean, the exponentially corrected mean, filename, day (true or false), night (true or false)
    mean_values.append([mean_beam_raw, mean_a5m_corr_exp, name[0], name[1], name[2]])

count = 0

halocline_depth_list = []
for entry in mean_values:
    if count == 2:
        count = 0
        #plt.legend()
        #plt.title('Mean signal strength')
        #plt.ylabel('Depth')
        #plt.xlabel('Signal strength (echo sounder)')
        #plt.show()
    if count == 0:
        strat_depth, strat_depth_point = halocline_depth(entry[1], depth_z)
        #fig1 = plt.plot(np.transpose(entry[0]), depth_z, color='k', linewidth=1, label='Raw mean day')
        #plt.plot(np.transpose(entry[1]), depth_z, color='orange', linewidth=3, label='Corrected mean exp ' + entry[2])
        #plt.plot(strat_depth_point[0], strat_depth_point[1], 'kx', markersize=7, label='Est. stratification depth')

        if entry[3] == 'TRUE':
            entry_string = entry[2].split('_')
            entry_day = int(entry_string[2])
            entry_month = int(entry_string[1])
            day_time = datetime(2018, entry_month, entry_day, 12, 0)
            halocline_depth_list.append([strat_depth, day_time])
        elif entry[3] == 'FALSE':
            entry_string = entry[2].split('_')
            entry_day = int(entry_string[2])
            entry_month = int(entry_string[1])
            night_time = datetime(2018, entry_month, entry_day, 23, 59)
            #halocline_depth_list.append([strat_depth, night_time])

    else:
        strat_depth, strat_depth_point = halocline_depth(entry[1], depth_z)
       # plt.plot(np.transpose(entry[0]), depth_z, color='grey', linewidth=1, label='Raw mean night')
        #plt.plot(np.transpose(entry[1]), depth_z, color='steelblue', linewidth=3, label='Corrected mean exp ' + entry[2])
        #plt.plot(strat_depth_point[0], strat_depth_point[1], 'kx', markersize=7)

        if entry[3] == 'TRUE':
            entry_string = entry[2].split('_')
            entry_day = int(entry_string[2])
            entry_month = int(entry_string[1])
            day_time = datetime(2018, entry_month, entry_day, 12, 0)
            halocline_depth_list.append([strat_depth, day_time])
        elif entry[3] == 'FALSE':
            entry_string = entry[2].split('_')
            entry_day = int(entry_string[2])
            entry_month = int(entry_string[1])
            night_time = datetime(2018, entry_month, entry_day, 23, 59)
           # halocline_depth_list.append([strat_depth, night_time])

    count += 1

xs = []
ys = []
for item in halocline_depth_list:
    # item[1] is the depth (-value) and item[0] is the datetime object
    xs.append(item[1])
    ys.append(item[0])

ctd_xs = [-10, -12.5, -5]
ctd_ys = [datetime(2018, 8, 28, 11, 30), datetime(2018, 9, 11, 9, 30), datetime(2018, 9, 25, 9, 0)]

filename = "smhi-opendata_test.csv"
wind_speed = []
w_timestamp =[]
w_direction = []

with open(filename, 'r') as f:
    AIS_file = csv.reader(f, delimiter=';')
    for line in AIS_file:
        if line[0] == 'Datum':
            pass
        else:
            wind_speed.append(float(line[4]))
            w_direction.append(int(line[2]))
            date_time = line[0] + ' ' + line[1]
            w_timestamp.append(datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S'))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(w_timestamp, wind_speed, '-')
ax1.set_xlim(datetime(2018, 8, 23, 0, 0), datetime(2018, 9, 30, 0, 0))
ax1.set_title('Wind speed at Vinga')
ax3 = ax1.twinx()
ax3.plot(xs, ys, 'k-', markersize=7)
plt.plot(ctd_ys, ctd_xs, 'rx', markersize=7)
#ax3.plot(ctd_xs, ctd_ys, 'rx', markersize=7)

ax2.plot(w_timestamp, w_direction, '-')
ax4 = ax2.twinx()
ax4.plot(xs, ys, 'k-', markersize=7)
plt.plot(ctd_ys, ctd_xs, 'rx', markersize=7)
#ax4.plot(ctd_xs, ctd_ys, 'rx', markersize=7)
ax2.set_title('Wind direction at Vinga')
ax2.set_xlim(datetime(2018, 8, 23, 0, 0), datetime(2018, 9, 30, 0, 0))

#plt.legend('Wind speed')

plt.show()


"""
plt.plot(xs, ys, 'k-', markersize=7)

plt.show()

strat_depth, strat_depth_point = halocline_depth(mean_a5m_corr_exp, depth_z)

fig1 = plt.plot(np.transpose(mean_a5m_corr_exp), depth_z)
plt.plot(strat_depth_point[0], strat_depth_point[1], 'kx', markersize=7, label='Est. stratification depth')
plt.title('Signal strength')
plt.legend()
plt.show()


# Used for plotting more than 1 day and 1 night
if count == 2:
    count = 0
    plt.legend()
    plt.title('Mean signal strength')
    plt.ylabel('Depth')
    plt.xlabel('Signal strength (echo sounder)')
    plt.show()
if count == 0:
    fig1 = plt.plot(np.transpose(entry[0]), depth_z, color='k', linewidth=1, label='Raw mean')
    ax1 = plt.gca()
    ax1.set_prop_cycle(cycler('color', ['orange', 'steelblue']) + cycler('lw', [3, 3]))
    # ax1.set_prop_cycle(cycler('color', ['orange', 'steelblue', 'r', 'lightskyblue', 'maroon', 'blue']) + cycler('lw', [1, 1, 1, 1, 1, 1]))

    plt.plot(np.transpose(entry[1]), depth_z, label='Corrected mean exp ' + entry[2])

else:
    plt.plot(np.transpose(entry[1]), depth_z, label='Corrected mean exp ' + entry[2])

count += 1
"""

