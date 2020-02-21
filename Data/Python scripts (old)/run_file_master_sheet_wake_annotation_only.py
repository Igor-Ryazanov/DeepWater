from datetime import datetime, timedelta
import csv
from ShipWake import ShipWake
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from wake_calculation_3 import wake_depth, maxdepth_longevity_calculation, wake_depth_total_mean, save_csv_file_wake_info
from signal_correction_water import signal_correction_exp, halocline_depth
from AIS import ShipAIS
from load_turbulence_files_ADCP import load_turbulence_file, mean_calculation_turbulence
from PlottingHistograms import plot_histogram_wake_single, plot_histogram_wake_all


# Load the csv file with wake info, read the data and save in a list.
wake_list = []
with open('test_master_sheet_7.csv', 'r') as f:     # 'test_master_sheet_7.csv', 'plotting_for_SE_no_eps.csv'
    wake_file = csv.reader(f, delimiter=';')
    for line in wake_file:
        test = line[0]
        if line[0] == 'ï»¿Filename':
            pass
        elif line == '':
            pass
        else:
            # The quality markers will give information about which parameters that are available in each line and if
            # there are any data quality issues.
            raw_quality_markers = (line[1]).split(',')
            quality_markers = []
            for item in raw_quality_markers:
                quality_markers.append(item.strip())

            # Parsing info needed for the wake analysis
            start_date = datetime(int(line[2]), int(line[3]), int(line[4]), int(line[5]), int(line[6]))
            min_depth = int(line[8])
            percentage = float(line[9])
            duration = int(line[10])

            # Create a Ship object that I can use to save the information about the ship creating the wake
            ship = ShipAIS()
            # Saving the information for the first ship object
            ship.distance = [(float(line[16]), 'ADCP')]  # [tuple]. Distance to the ADCP in meter
            ship.date = datetime.strptime(line[17], '%Y-%m-%d  %H:%M')  # [yyyy-mm-dd HH:MM:SS] Datenum object.
            ship.sog = int(line[18])  # [Knots]. Speed over ground
            ship.name = line[19]  # [string]. Ship name
            ship.ship_type_t = line[20]  # [string]. Ship type category.
            ship.static_draught = float(line[21])  # [m]. Static draught of ship.
            ship.current_draught = float(line[22])  # [m]. Added from Port IT dataset.
            ship.arr_dep = line[23]  # [string]. Indicates arrival/departure
            ship.length = float(line[24])  # [m]
            ship.width = float(line[25])  # [m]
            ship.cog = float(line[26])    # [degrees]

            # TODO: Uncomment this section to calculate the wake area for the turbulence data as well.
            """
            turbulence_filename_bubble = line[38]      # The filename of the file with the early start of the wakes
            turbulence_filename_ship = line[39]  # The filename in the cases where there is a later start for the wake

            start_date_turb = datetime.strptime(line[40], '%Y-%m-%d  %H:%M')  # [yyyy-mm-dd HH:MM:SS] Datenum

            # If the end date is the string 'end', there is no end date, but the rest of the file should be used.
            if line[41] == 'end':
                end_date_turb = 'end'
            else:
                end_date_turb = datetime.strptime(line[41], '%Y-%m-%d  %H:%M')  # End date for the mean turb calculation

            turb_mean_filename = line[42] # Filename for mean calculation, not always same as for turbulence calculation

            alt_turb_start = None
            if 'turb start' in quality_markers:
                alt_turb_start = datetime.strptime(line[49], '%Y-%m-%d  %H:%M')

            # These are the parameters needed for the eps and uVar wake calculation:
            # [eps_min_depth, eps_percentage, eps_duration, uVar_min_depth, uVar_percentage,
            # uVar_duration, alternative start time if needed]
            wake_parameters = [int(line[43]), float(line[44]), int(line[45]), int(line[46]), float(line[47]),
                               int(line[48]), alt_turb_start]

            # TODO: Change the beam here! Options are 'one' or 'five'. Wave correction can be True or False.
            #  Bubble or ship can be True.
            turbulence_beam = 'five'
            wave_corrected_data = True
            bubble = True
            ship_start = False

            # The list returned from the turublence data contains the following parameters:
            # 0: python_turb_datetime, 1: depth_z, 2: uVar1, 3: uVar2, 4: eps1, 5: eps2, 6: err1, 7: err2,
            # 8: current_u_east, 9: current_u_north
            if bubble is True:
                turbulence_data = load_turbulence_file(turbulence_filename_bubble, turbulence_beam, 'bubble')
                # The list returned from mean_calculation_turbulence contains the following parameters:
                # 0: mean_uVar1, 1: mean_uVar2, 2: mean_eps1, 3: mean_eps2, 4: mean_err1, 5: mean_err2, 6: mean_u_east,
                # 7: mean_u_north
                turbulence_mean = mean_calculation_turbulence(start_date_turb, end_date_turb, turb_mean_filename,
                                                              'five', 'bubble')
            if ship_start is True:
                turbulence_data = load_turbulence_file(turbulence_filename_ship, turbulence_beam, 'ship')
                turbulence_mean = mean_calculation_turbulence(start_date_turb, end_date_turb, turb_mean_filename,
                                                              'five', 'ship')

            # turbulence_mean = mean_calculation_turbulence(start_date_turb, end_date_turb, turb_mean_filename, 'five')
            """

            if 'two ships' in quality_markers:
                if 'both' in quality_markers:
                    # Saving the information for the second ship object
                    ship_2 = ShipAIS()
                    ship_2.distance = [(float(line[27]), 'ADCP')]
                    ship_2.date = datetime.strptime(line[28], '%Y-%m-%d  %H:%M')
                    ship_2.sog = int(line[29])
                    ship_2.name = line[30]
                    ship_2.ship_type_t = line[31]
                    ship_2.static_draught = float(line[32])
                    ship_2.current_draught = float(line[33])
                    ship.arr_dep = line[34]
                    ship_2.length = float(line[35])
                    ship_2.width = float(line[36])
                    ship_2.cog = float(line[37])

                    # Create a ShipWake object containing all necessary information for analysis and the wake in a list.
                    # 0: filename, 1: quality markers, 2: start date for the wake analysis, 3: percentage, 4: duration,
                    # 5: ship_info info about the ship(s) creating the wake) [ship object(s)].
                    wake = ShipWake(line[0], quality_markers, start_date, min_depth, percentage, duration, [ship, ship_2])

                    # This function completes the calculation for the parameters of the bubble wake etc.
                    # If the aim is to print the csv-file, this calculation is performed inside that function
                    wake.wake_depth_calculation('a5m')
                    """
                    wake.turbulent_wake_depth_calculation(turbulence_data, turbulence_mean, wave_corrected_data,
                                                          wake_parameters)
                    """
                else:
                    wake = ShipWake(
                        line[0], quality_markers, start_date, min_depth, percentage, duration, [ship])

                    wake.wake_depth_calculation('a5m')
                    """
                    wake.turbulent_wake_depth_calculation(turbulence_data, turbulence_mean, wave_corrected_data,
                                                          wake_parameters)
                    """
                wake_list.append(wake)

            else:
                wake = ShipWake(line[0], quality_markers, start_date, min_depth, percentage, duration, [ship])

                wake.wake_depth_calculation('a5m')
                """
                wake.turbulent_wake_depth_calculation(turbulence_data, turbulence_mean, wave_corrected_data,
                                                      wake_parameters)
                """
                wake_list.append(wake)


# TODO: The wake_list contains the created ShipWake objects, where all the information from the different calculations
#  are saved. Each wake in the list represents one of the annotated ship wakes. I hope you will be able to get
#  some/most of the information you need form the ShipWake objects.


# Write and save a csv-file with all the data and return a list with all the data that can be used for further
# calculations

# final_master_list = save_csv_file_wake_info('master_file_for_figures_1.csv', wake_list)

print('Done')






