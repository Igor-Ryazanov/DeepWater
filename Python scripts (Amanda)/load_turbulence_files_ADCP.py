import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as dates
import csv
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.io as sio


def load_turbulence_file(matlabfile_name, beam, start_point):
    """
    Takes a string with the matlab filename and the beam to be analyzed and returns
    :param matlabfile_name: A string with the name of the matlab file with the turbulence data.
    :param beam: The beam to do the analysis for ('one' or 'five')
    :param start_point: A string giving which start point the turbulence analysis starts at: the beginning of the bubble
     wake or when the ship passes.
    :return: A list with the following turbulence data:
            0)python_turb_datetime
            1) depth_z
            2) uVar1
            3) uVar2
            4) eps1
            5) eps2
            6) err1
            7) err2
            8) current_u_east
            9) current_u_north

    """

    # Loading the name of the datafile containing the wake
    if start_point is 'bubble':
        mat_file = matlabfile_name[0: -4] + '_pythonfile_bubble' + '.mat'
    elif start_point is 'ship':
        mat_file = matlabfile_name[0: -4] + '_pythonfile_ship' + '.mat'

    # mat_file = matlabfile_name[0: -4] + '_pythonfile' + '.mat'    # Use this for the old version

    # Loading the data from the matlab file
    turbulence_data_file = sio.loadmat(mat_file)

    analyzed_beam = beam
    turb_time = turbulence_data_file['t'][0]
    depth_z = turbulence_data_file['z'][0]

    if analyzed_beam == 'five':
        uVar1 = turbulence_data_file['u5Var1']
        uVar2 = turbulence_data_file['u5Var2']
        eps1 = turbulence_data_file['eps51']
        eps2 = turbulence_data_file['eps52']
        err1 = turbulence_data_file['err51']
        err2 = turbulence_data_file['err52']

    elif analyzed_beam == 'one':
        uVar1 = turbulence_data_file['u1Var1']
        uVar2 = turbulence_data_file['u1Var2']
        eps1 = turbulence_data_file['eps1']
        eps2 = turbulence_data_file['eps2']
        err1 = turbulence_data_file['err1']
        err2 = turbulence_data_file['err2']

    elif analyzed_beam == 'three':
        uVar1 = turbulence_data_file['u3Var1']
        uVar2 = turbulence_data_file['u3Var2']

    elif analyzed_beam == 'four':
        uVar1 = turbulence_data_file['u4Var1']
        uVar2 = turbulence_data_file['u4Var2']

    current_u_east = turbulence_data_file['uem']
    current_u_north = turbulence_data_file['unm']
    python_datetime = turb_time
    python_turb_datetime = []

    # Converting matlab-time to datetime-time.
    for item in python_datetime:
        python_turb_datetime.append(datetime.fromordinal(int(item)) + timedelta(days=item % 1) - timedelta(days=366))

    turbulence_data = [python_turb_datetime, depth_z, uVar1, uVar2, eps1, eps2, err1, err2, current_u_east,
                       current_u_north]

    return turbulence_data


def mean_calculation_turbulence(mean_start, mean_end, turb_filename, beam, start_point):
    """
    Loads the datset used for the mean calculation and returns the mean values at each depth for each parametre.
    :param mean_start: The start date and time of the section of the raw dataset that will be used to calculate the mean
    :param mean_end: The end date and time of the section of the raw dataset that will be used to calculate the mean
    :param turb_filename: The filename of the dataset to use in the mean calculation. Manually decided.
    :param beam: The name of the beam to analyze ('one' or 'five')
    :param start_point: A string giving which start point the turbulence analysis starts at: the beginning of the bubble
     wake or when the ship passes.
    :return: A list with the following parameters:
            0) mean_uVar1: A vector with the mean velocity variance, NOT corrected for waves, for each depth [1 x 28].
            1) mean_uVar2: A vector with the mean velocity variance, corrected for waves, for each depth [1 x 28].
            2) mean_eps1: A vector with the mean TKE dissipation rate, NOT corrected for waves for each depth [1 x 28].
            3) mean_eps2: A vector with the mean TKE dissipation rate, corrected for waves for each depth [1 x 28].
            4) mean_err1: A vector with the mean error (?), NOT corrected for waves for each depth [1 x 28].
            5) mean_err2: A vector with the mean error (?), corrected for waves for each depth [1 x 28].
            6) mean_current_u_east: A vector with the mean velocity in the eastern direction [1 x 28].
            7) mean_current_u_north: A vector with the mean velocity in the northern direction [1 x 28].
    """

    # Loading the name of the datafile containing the wake
    if start_point is 'bubble':
        mat_file = turb_filename[0: -4] + '_pythonfile_bubble' + '.mat'
    elif start_point is 'ship':
        mat_file = turb_filename[0: -4] + '_pythonfile_ship' + '.mat'

    # mat_file = turb_filename[0: -4] + '_pythonfile' + '.mat'          # For old version, without current

    # Loading the data from the matlab file
    mean_data_file = sio.loadmat(mat_file)

    analyzed_beam = beam
    mean_turb_time = mean_data_file['t'][0]
    m_depth_z = mean_data_file['z'][0]

    if analyzed_beam == 'five':
        uVar1 = mean_data_file['u5Var1']
        uVar2 = mean_data_file['u5Var2']
        eps1 = mean_data_file['eps51']
        eps2 = mean_data_file['eps52']
        err1 = mean_data_file['err51']
        err2 = mean_data_file['err52']

    elif analyzed_beam == 'one':
        uVar1 = mean_data_file['u1Var1']
        uVar2 = mean_data_file['u1Var2']
        eps1 = mean_data_file['eps1']
        eps2 = mean_data_file['eps2']
        err1 = mean_data_file['err1']
        err2 = mean_data_file['err2']

    current_u_east_m = mean_data_file['uem']
    current_u_north_m = mean_data_file['unm']
    python_datetime = mean_turb_time
    python_turb_datetime = []

    # Converting matlab-time to datetime-time.
    for item in python_datetime:
        python_turb_datetime.append(datetime.fromordinal(int(item)) + timedelta(days=item % 1) - timedelta(days=366))

    # FINDING THE START AND END OF THE DATASET USED TO CALCULATE THE MEAN
    # start = python_turb_datetime.index(mean_start) does not work as the time must be the exact time, which it isn't
    # The start_indx will give the row index from where the dataset used to calculate the mean should start, and the
    # end_index the end row.
    # Count is used to keep track of the row index, start_found is used for the cases where the end index must be found
    count = 0
    start_found = False

    for item in python_turb_datetime:
        # If mean_end == 'end' all data from mean_start to the end will be used and no end index is needed.
        if mean_end == 'end':
            if mean_start <= item:
                start_indx = count
                end_indx = len(python_turb_datetime)
                break
            else:
                count += 1
        else:
            # If both indices are needed, start is found first and then the loop is stopped when the end index is found.
            if start_found is True:
                if mean_end <= item:
                    end_indx = count
                    break
                else:
                    count += 1

            elif start_found is False:
                if mean_start <= item:
                    start_indx = count
                    start_found = True
                    count += 1
                else:
                    count += 1

    # Creating the matrices where the new data is stored
    section_uVar1 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_uVar2 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_eps1 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_eps2 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_err1 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_err2 = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_current_u_east_m = np.zeros([(end_indx - start_indx), len(m_depth_z)])
    section_current_u_north_m = np.zeros([(end_indx - start_indx), len(m_depth_z)])

    count = 0
    for idx in range(start_indx, end_indx):
        # Finding the values to be used in the mean calculation
        section_uVar1[count, :] = uVar1[idx, :]
        section_uVar2[count, :] = uVar2[idx, :]
        section_eps1[count, :] = eps1[:, idx]
        section_eps2[count, :] = eps2[:, idx]
        section_err1[count, :] = err1[:, idx]
        section_err2[count, :] = err2[:, idx]
        section_current_u_east_m[count, :] = current_u_east_m[idx, :]
        section_current_u_north_m[count, :] = current_u_north_m[idx, :]

        count += 1

    # Calculating the mean value for the section of the turbulence file that is not part of the wake
    mean_uVar1 = np.zeros([1, 28])
    mean_uVar2 = np.zeros([1, 28])
    mean_eps1 = np.zeros([1, 28])
    mean_eps2 = np.zeros([1, 28])
    mean_err1 = np.zeros([1, 28])
    mean_err2 = np.zeros([1, 28])
    mean_current_u_east = np.zeros([1, 28])
    mean_current_u_north = np.zeros([1, 28])

    for d in range(0, 28):
        mean_uVar1[0][d] = np.mean(section_uVar1[:, d])
        mean_uVar2[0][d] = np.mean(section_uVar2[:, d])
        mean_eps1[0][d] = np.mean(section_eps1[:, d])
        mean_eps2[0][d] = np.mean(section_eps2[:, d])
        mean_err1[0][d] = np.mean(section_err1[:, d])
        mean_err2[0][d] = np.mean(section_err2[:, d])
        mean_current_u_east[0][d] = np.mean(section_current_u_east_m[:, d])
        mean_current_u_north[0][d] = np.mean(section_current_u_north_m[:, d])

    return [mean_uVar1, mean_uVar2, mean_eps1, mean_eps2, mean_err1, mean_err2, mean_current_u_east,
            mean_current_u_north]


"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(sharex=True, nrows=4, ncols=1)

    # Plotting the raw data as a contour plot, where z is the echo amplitude signal strength
    levels_eps = np.logspace(-6, -3, 70)
    levels_var = np.logspace(-3, -1, 70)

    # The unit for epsilon (\u03B5) is m^2/s^3 or [J/(kg*s]
    fig1 = ax1.contourf(python_turb_datetime, depth_z, eps1, levels_eps, cmap='jet', locator=ticker.LogLocator())
    ax1.set_title('Turbulent kinetic energy dissipation rate (\u03B5) [m\u00b2/s\u00b3] (not corrected, log scale)')
    ax1.set_ylabel('Depth [m]')
    plt.colorbar(fig1, ax=ax1, ticks=[10**-6, 10**-5, 10**-4, 10**-3])

    fig2 = ax2.contourf(python_turb_datetime, depth_z, eps2,  levels_eps, cmap='jet', locator=ticker.LogLocator())
    ax2.set_title('Turbulent kinetic energy dissipation rate (\u03B5) [m\u00b2/s\u00b3] (corrected for waves, log scale)')
    ax2.set_ylabel('Depth [m]')
    plt.colorbar(fig2, ax=ax2, ticks=[10**-6, 10**-5, 10**-4, 10**-3])

    fig3 = ax3.contourf(python_turb_datetime, depth_z, np.transpose(uVar1), levels_var, cmap='jet',
                        locator=ticker.LogLocator())
    ax3.set_title('Velocity variance [m/s] (uVar not corrected)')
    ax3.set_ylabel('Depth [m]')
    plt.colorbar(fig3, ax=ax3, ticks=[10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1])

    fig4 = ax4.contourf(python_turb_datetime, depth_z, np.transpose(uVar2), levels_var, cmap='jet',
                        locator=ticker.LogLocator())
    ax4.set_title('Velocity variance [m/s] (uVar corrected for waves)')
    ax4.set_ylabel('Depth [m]')
    ax4.set_xlabel('Time and date (HH:MM yyyy-mm-dd)')
    plt.colorbar(fig4, ax=ax4,  ticks=[10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1])

    plt.gcf().autofmt_xdate()


    plt.suptitle('Velocity variance during the period ' + python_turb_datetime[0].strftime('%m/%d/%Y') +
                 ' - ' + python_turb_datetime[-1].strftime('%m/%d/%Y'))

    for ax in fig.get_axes():
        ax.label_outer()

    # plt.suptitle('Turbublence estimates (' + python_turb_datetime[0].strftime('%B %Y') + ')')
    formatter = DateFormatter('%H:%M %Y-%m-%d')
    # formatter = DateFormatter('%H:%M')

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    # ax.frmt_xdata = mdate.DateFormatter('%m-%d')

    plt.show()
    """


# load_turbulence_file('Passage_8_29_10_2_pythonfile.mat', 'five')


