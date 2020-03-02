import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as dates
import csv
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import scipy.io as sio
from wake_calculation_3 import wake_depth, maxdepth_longevity_calculation, wake_depth_total_mean, \
    plotting_wake_calculation_data, plotting_wake_calculation_data_log_single, plotting_wake_area, \
    plotting_zoom_bubble_wake, wake_matrix, current_calculation
from signal_correction_water import signal_correction_exp, halocline_depth
from load_turbulence_files_ADCP import load_turbulence_file, mean_calculation_turbulence


class ShipWake:
    def __init__(self, filename, quality_markers, start_date, min_depth, percentage, duration, ship_info):
        """
        Creates a ship wake object containing all information needed for the ship wake analysis.
        0: filename, 1: quality markers, 2: start date for the wake analysis, 3: percentage, 4: duration,
        5: ship_info info about the ship(s) creating the wake) [ship object(s)].
        :param file_name: Name of the file where the echo sounder data is stored [string].
        :param quality_markers: A list of strings containing quality markers that gives information about which
        parameters that are present and the quality of the data.
        :param min_depth: The deepest depth that is considerd in the wake calculation (used to exclude noise at bottom).
        :param start_date: The start date needed for the wake depth analysis.
        :param percentage: The percentage difference in signal strength needed for the wake depth analysis.
        :param duration: The number of measurements included in the wake analysis, thereby giving the duration of the
        wake, needed for the wake depth analysis.
        :param ship_info: A list with of ship objects, where each ship object contains the relevant information for the
        wake analysis.
        """

        # Parameters needed to create the ShipWake object
        self.filename = filename                # [string]. The filename of the datafile used in the wake calculations.
        self.quality_markers = quality_markers  # [list of strings]. Key words used to modify the analysis
        self.start_date = start_date            # [List of integers]. Parameters used to perform the wake calculation.
        self.min_depth = min_depth              # [Integer]. The deepest depth considered in the wake calculation
        self.percentage = percentage            # [Float]. % difference in signal strength for the wake depth analysis.
        self.duration = duration                # [integer]. The duration of the wake.
        self.ship_info = ship_info              # [List of lists]. Information about the ship that created the wake.

        # Parameters that are defined later
        self.max_wake_depth = np.nan              # [Float]. Depth in meter.
        self.longevity_time = np.nan              # [Datetime]. Duration of the wake as time in datetime format.
        self.longevity_distance = []              # [List]. List of one or two possible distances in meter.
        self.wake_start_time = np.nan             # [Datetime]. Start date for the wake [dd mmm yyyy HH:MM:SS.000] [UTC]
        self.wake_end_time = np.nan               # [Datetime]. End date for the wake [dd mmm yyyy HH:MM:SS.000] [UTC]
        self.stratification_depth = np.nan        # [Float]. The estimated stratification dept in meter.
        self.direction = []                       # [List]. The direction(s) of the ship(s) [degrees].

        # Parameters derived from the turbulence_wake_depth_calculation
        self.max_eps = np.nan                     # [Float]. The maximum TKE dissipation rate (epsiolon).
        self.max_depth_eps = np.nan               # [Float]. The depth of the maximum epsilon in meter.
        self.eps_longevity_time = np.nan          # [Datetime]. Duration of the wake as time in datetime format.
        self.eps_longevity_distance = []          # [List]. List of one or two possible distances in meter.
        self.max_uVar = np.nan                    # [Float]. The maximum velocity variance [m/s]?
        self.max_depth_uVar = np.nan              # [Float]. The depth of the maximum velocity variance in meter.
        self.uVar_longevity_time = np.nan         # [Datetime]. Duration of the wake as time in datetime format.
        self.uVar_longevity_distance = []         # [List]. List of one or two possible distances in meter.

        # Datasets for PCA analysis
        self.bubble_matrix = None                 # Matrix over the bubble wake that is used in the PCA analysis.
        self.eps_matrix = None                    # Matrix over the epsilon wake that is used in the PCA analysis.
        self.uVar_matrix = None                   # Matrix over the uVar wake that is used in the PCA analysis.

        # Parameters currently missing
        self.av_current_speed = np.nan            # [Knots]. The average current speed in knots (or m/s?).
        self.av_current_direction = np.nan        # [Degrees]. The average current direction in degrees.
        self.mean_current_east = np.nan           # [m/s?]. The mean easterly current speed at the time of the passage.
        self.mean_current_north = np.nan          # [m/s?]. The mean northerly current speed at the time of the passage.
        self.current_ship_dir = np.nan            # [m/s]. The current speed in the ship traveling direction at the time of the passage at all depths [1 x 28].
        self.current_instrument_dir = np.nan      # [m/s]. The current speed towards/away from the instrument at the time of the passage at all depths [1 x 28].
        self.ship_current_vector = []             # List with the average ship and instrument current directions for different depths [0-5, 5-10, 10-15, 15-30, 20-25] (depths are all + ca 3 m due to blanking distance.

    def wake_depth_calculation(self, beam):
        """
        Calculates all the wake object parameters that are not defined when initiating the object.
        :param beam: The beam is a string indicating which of the beams in the ADCP that the calculation is made for.
        :return: All remaining parameters in the wake object.
        """

        # Loading the name of the datafile containing the wake
        mat_file = self.filename + '_pythonfile'

        # Loading the data from the matlab file
        wake_data_file = sio.loadmat(mat_file)
        analyzed_beam = wake_data_file[beam]
        tm = wake_data_file['tm'][0]
        depth_z = wake_data_file['z'][0]

        python_datetime = tm
        python_tm = []

        # Converting matlab-time to datetime-time.
        for item in python_datetime:
            python_tm.append(datetime.fromordinal(int(item)) + timedelta(days=item % 1) - timedelta(days=366))

        # Making a correction in the signal strength based on the distance from the instrument (exponential function)
        beam_corr_exp, mean_beam_corr_exp = signal_correction_exp(analyzed_beam, depth_z)

        # This calculation returns a list containing 0: current_date, 1: current_depth, 2: mean_amplitude,
        # 3) wake (indicates if the measured value is part of a wake, can be either True or False), 4: Raw data
        wake_calculation_total_mean = wake_depth_total_mean(beam_corr_exp, mean_beam_corr_exp, self.start_date,
                                                            python_tm, self.min_depth, self.percentage, self.duration)

        # Creates a normalised wake-matrix with padded zeroes for the bubble wake
        self.bubble_matrix = wake_matrix(beam_corr_exp, mean_beam_corr_exp, self.start_date, python_tm, self.min_depth,
                                       self.percentage, self.duration, 'amplitude', self.ship_info[0].date, depth_z,
                                       self.ship_info[0].name)

        # This calculation returns a list of a list, where each "final" list contains:
        # [max_val, max_depth, max_list, start_time, end_time, longevity, max_intensity, max_intensity_coords,
        # intensity_max_depth])
        max_depth_list = maxdepth_longevity_calculation(wake_calculation_total_mean)

        # plotting_wake_calculation_data(beam_corr_exp, python_tm, depth_z, max_depth_list,
        #                                        'Echo Amplitude signal strength')

        # PARAMETERS DERIVED FROM WAKE DEPTH CALCULATION
        self.max_wake_depth = max_depth_list[0][1]
        self.wake_start_time = max_depth_list[0][3]
        self.wake_end_time = max_depth_list[0][4]
        self.longevity_time = max_depth_list[0][5]   # [Datetime]. Duration of the wake as time in datetime format.

        # LONGEVITY CALCULATION
        ship = self.ship_info[0]

        # The longevity in meters = speed over ground * longevity (time)
        if "two ships" in self.quality_markers:
            # If both is indicated, it means that it is unclear which of the ships that have created the wake
            if "both" in self.quality_markers:
                ship_2 = self.ship_info[1]

                # Time delta objects have a class function total_seconds() that will give the number of seconds
                # The conversion factor from knot to m/s: 1 knot = 0.51444 m/s
                # longevity_m [m] = (ship speed [knots] * conversion factor [knots --> m/s]) * wake duration [s]
                longevity_m = (ship.sog * 0.51444) * self.longevity_time.total_seconds()
                longevity_m_2 = (ship_2.sog * 0.51444) * self.longevity_time.total_seconds()
                self.longevity_distance = [longevity_m, longevity_m_2]

            # If 'both' is not indicated, it means that the first ship is the most likely to have created the wake,
            # thus only that one is used in the calculation.
            else:
                self.longevity_distance = [(ship.sog * 0.51444) * self.longevity_time.total_seconds()]

        # This happen when there is only one ship
        else:
            self.longevity_distance = [(ship.sog * 0.51444) * self.longevity_time.total_seconds()]

        # STRATIFICATION DEPTH CALCULATION
        # Calculate the halocline depth using the derivative of the mean amplitude over depth
        # mean_signal_corr_derivative, strat_depth, strat_depth_point = halocline_depth(mean_a5m_corr_exp, depth_z)
        strat_depth, strat_depth_point = halocline_depth(mean_beam_corr_exp, depth_z, beam_corr_exp, python_tm)

        self.stratification_depth = strat_depth

        # TODO: PLOTTING
        #  1) Contourf plot of the amplitude data for the entire day/night period with wake start, end and max indicated
        #  2) Contourf plot (same as above) but with the wake area indicated by scattered circles
        #  3) Contourf plot zoomed in on the wake and start, min and max indicated

        # 1) plotting_wake_calculation_data(beam_corr_exp, python_tm, depth_z, max_depth_list, 'Echo amplitude signal strength')

        plotting_wake_area(wake_calculation_total_mean, python_tm, depth_z, beam_corr_exp, 'amplitude', self.ship_info[0].name)

        # 3) plotting_zoom_bubble_wake(beam_corr_exp, python_tm, depth_z, max_depth_list, 'Echo amplitude signal strength', self.ship_info[0].name)

    def turbulent_wake_depth_calculation(self, raw_turbulence_data_list, mean_turbulent_list,
                                         correction, wake_parameters):
        """
        Calculates the wake parameters that relate to the turbulent wake (max value, max depth, longevity)
        :param raw_turbulence_data_list: A list with the raw datasets for all parameters. The format is the output
        format of the load_turbulence_file function (0: python_turb_datetime, 1: depth_z, 2: uVar1, 3: uVar2, 4: eps1,
        5: eps2, 6: err1, 7: err2)
        :param mean_turbulent_list: A list with the mean values for each depth for each parameter in the
        raw_turbulence_data_list list. The format of each mean-vector is [1x28], one value for each depth.
        (0: mean_uVar1, 1: mean_uVar2, 2: mean_eps1, 3: mean_eps2, 4: mean_err1, 5: mean_err2)
        :param correction: Indicates which dataset to use, the one corrected for waves (correction = True) or
        the raw one (correction = False)
        :param wake_parameters: The parameters (min_depth, percentage, duration) needed for the wake depth calculation
        for eps and uVar. [uVar_min_depth, uVar_percentage, uVar_duration, eps_min_depth, eps_percentage, eps_duration,
        alternative start time (is either None or a datetime object)]
        :return: Calculates the turbulence parameters that are missing:
                                                                        self.max_eps
                                                                        self.max_depth_eps
                                                                        self.eps_longevity_time
                                                                        self.eps_longevity_distance
                                                                        self.max_uVar
                                                                        self.max_depth_uVar
                                                                        self.uVar_longevity_time
                                                                        self.uVar_longevity_distance
        """
        turb_time = raw_turbulence_data_list[0]
        depth_turb = raw_turbulence_data_list[1]
        uVar1 = raw_turbulence_data_list[2]
        uVar2 = raw_turbulence_data_list[3]
        eps1 = raw_turbulence_data_list[4]
        eps2 = raw_turbulence_data_list[5]

        mean_uVar1 = mean_turbulent_list[0]
        mean_uVar2 = mean_turbulent_list[1]
        mean_eps1 = mean_turbulent_list[2]
        mean_eps2 = mean_turbulent_list[3]

        mean_current_east = mean_turbulent_list[6]
        mean_current_north = mean_turbulent_list[7]

        """
        err1 = raw_turbulence_data_list[6]
        err2 = raw_turbulence_data_list[7]
        mean_err1 = mean_turbulent_list[4]
        mean_err2 = mean_turbulent_list[5]
        """

        # TURBULENT WAKE CALCULATION
        # This calculation returns a list containing 0: current_date, 1: current_depth, 2: mean_date, 3: mean_amplitude,
        # 4) wake (indicates if the measured value is part of a wake, can be either True or False), 5: The raw data for
        # the current depth and date.
        # wake_parameters[eps_min_depth, eps_percentage, eps_duration, uVar_min_depth, uVar_percentage, uVar_duration]
        # TODO: Add a parameter 'alt_start' than can equals None or not, if not None, use instead of self.start_date

        if wake_parameters[6] is not None:
            start_date_turb = wake_parameters[6]
        else:
            start_date_turb = self.start_date

        if correction is True:
            # print(wake_parameters[0], wake_parameters[1], wake_parameters[2])
            uVar_wake_calculation = wake_depth_total_mean(uVar2, mean_uVar2, start_date_turb, turb_time,
                                                          wake_parameters[3], wake_parameters[4], wake_parameters[5])
            # print(wake_parameters[3], wake_parameters[4], wake_parameters[5])
            eps_wake_calculation = wake_depth_total_mean(np.transpose(eps2), mean_eps2, start_date_turb, turb_time,
                                                         wake_parameters[0], wake_parameters[1], wake_parameters[2])

            # Creates a normalised wake-matrix with padded zeroes for the uVar and eps wake
            self.uVar_matrix = wake_matrix(uVar2, mean_uVar2, start_date_turb, turb_time, wake_parameters[3],
                                                wake_parameters[4], wake_parameters[5], 'uVar', self.ship_info[0].date,
                                                depth_turb, self.ship_info[0].name)
            self.eps_matrix = wake_matrix(np.transpose(eps2), mean_eps2, start_date_turb, turb_time,
                                               wake_parameters[0], wake_parameters[1], wake_parameters[2], 'eps',
                                               self.ship_info[0].date, depth_turb, self.ship_info[0].name)

            """
            # I am not sure what these parameters represent or how to use them, but they are included
            err_wake_calculation = wake_depth_total_mean(err1, mean_err2, self.start_date, turb_time, self.min_depth,
                                                                              self.percentage, self.duration)
            """

        elif correction is False:
            uVar_wake_calculation = wake_depth_total_mean(uVar1, mean_uVar1, start_date_turb, turb_time,
                                                          wake_parameters[3], wake_parameters[4], wake_parameters[5])
            eps_wake_calculation = wake_depth_total_mean(np.transpose(eps1), mean_eps1, start_date_turb, turb_time,
                                                         wake_parameters[0], wake_parameters[1], wake_parameters[2])

            self.uVar_matrix = wake_matrix(uVar1, mean_uVar1, start_date_turb, turb_time, wake_parameters[3],
                                                wake_parameters[4], wake_parameters[5], 'uVar', self.ship_info[0].date,
                                                depth_turb, self.ship_info[0].name)
            self.eps_matrix = wake_matrix(np.transpose(eps1), mean_eps1, start_date_turb, turb_time,
                                               wake_parameters[0], wake_parameters[1], wake_parameters[2], 'eps',
                                               self.ship_info[0].date, depth_turb, self.ship_info[0].name)
            """
            # I am not sure what these parameters represent or how to use them, but they are included
            err_wake_calculation = wake_depth_total_mean(err1, mean_err1, self.start_date, turb_time, self.min_depth,
                                                                  self.percentage, self.duration)
            """
        else:
            print('Error in turbulent wake depth calculation')

        # This calculation returns a list of a list, where each "final" list contains:
        # [max_val, max_depth, max_list, start_time, end_time, longevity, max_intensity, max_intensity_coords])
        uVar_depth_list = maxdepth_longevity_calculation(uVar_wake_calculation)
        eps_depth_list = maxdepth_longevity_calculation(eps_wake_calculation)
        """
        err_depth_list = maxdepth_longevity_calculation(err_wake_calculation)
        """

        # PARAMETERS DERIVED FROM WAKE DEPTH CALCULATION
        # As the output of the maxdepth_longevity_calculation is a list of list, an additional index [0] must be used
        self.max_eps = eps_depth_list[0][6]
        self.max_depth_eps = eps_depth_list[0][1]
        self.eps_longevity_time = eps_depth_list[0][5]

        self.max_uVar = uVar_depth_list[0][6]
        self.max_depth_uVar = uVar_depth_list[0][1]
        self.uVar_longevity_time = uVar_depth_list[0][5]

        # LONGEVITY CALCULATION (distance)
        ship = self.ship_info[0]

        # The longevity in meters = speed over ground * longevity (time)
        if "two ships" in self.quality_markers:
            # If both is indicated, it means that it is unclear which of the ships that have created the wake
            if "both" in self.quality_markers:
                ship_2 = self.ship_info[1]

                # Time delta objects have a class function total_seconds() that will give the number of seconds
                # The conversion factor from knot to m/s: 1 knot = 0.51444 m/s
                # longevity_m [m] = (ship speed [knots] * conversion factor [knots --> m/s]) * wake duration [s]

                longevity_m_eps = (ship.sog * 0.51444) * self.eps_longevity_time.total_seconds()
                longevity_m_2_eps = (ship_2.sog * 0.51444) * self.eps_longevity_time.total_seconds()
                self.eps_longevity_distance = [longevity_m_eps, longevity_m_2_eps]

                longevity_m_uVar = (ship.sog * 0.51444) * self.uVar_longevity_time.total_seconds()
                longevity_m_2_uVar = (ship_2.sog * 0.51444) * self.uVar_longevity_time.total_seconds()
                self.uVar_longevity_distance = [longevity_m_uVar, longevity_m_2_uVar]

            # If 'both' is not indicated, it means that the first ship is the most likely to have created the wake,
            # thus only that one is used in the calculation.
            else:
                self.eps_longevity_distance = [(ship.sog * 0.51444) * self.eps_longevity_time.total_seconds()]
                self.uVar_longevity_distance = [(ship.sog * 0.51444) * self.uVar_longevity_time.total_seconds()]

        # This happen when there is only one ship
        else:
            self.eps_longevity_distance = [(ship.sog * 0.51444) * self.eps_longevity_time.total_seconds()]
            self.uVar_longevity_distance = [(ship.sog * 0.51444) * self.uVar_longevity_time.total_seconds()]

        # CURRENT PARAMETERS
        # Here the parameter is vector with the mean velocity in east/north direction for each depth [1 x 28].
        self.mean_current_east = mean_current_east
        self.mean_current_north = mean_current_north
        # Should I have an average of the top 5 m instead?
        average_mean_current_east = np.mean(mean_current_east[0, 23:28])
        average_mean_current_north = np.mean(mean_current_north[0, 23:28])

        # Current parameters used in the PCA, current speed in the ship's traveling direction and in the direction
        # towards/away from the instrument. Positive values means traveling with the current and towards the instrument.
        self.current_ship_dir, self.current_instrument_dir = current_calculation(mean_current_east, mean_current_north,
                                                                                 ship.cog, ship.name, 'all depths')

        current_ship_dir_0_5, current_instrument_dir_0_5 = current_calculation([mean_current_east[0, 23:28]],
                                                                                 [mean_current_north[0, 23:28]],
                                                                                 ship.cog, ship.name, 'top 5m')

        current_ship_dir_5_10, current_instrument_dir_5_10 = current_calculation([mean_current_east[0, 18:23]],
                                                                                 [mean_current_north[0, 18:23]],
                                                                                 ship.cog, ship.name, '5-10m')

        current_ship_dir_10_15, current_instrument_dir_10_15 = current_calculation([mean_current_east[0, 13:18]],
                                                                                 [mean_current_north[0, 13:18]],
                                                                                 ship.cog, ship.name, '10-15m')

        current_ship_dir_15_20, current_instrument_dir_15_20 = current_calculation([mean_current_east[0, 8:13]],
                                                                                 [mean_current_north[0, 8:13]],
                                                                                 ship.cog, ship.name, '15-20m')

        current_ship_dir_20_25, current_instrument_dir_20_25 = current_calculation([mean_current_east[0, 3:8]],
                                                                                   [mean_current_north[0, 3:8]],
                                                                                   ship.cog, ship.name, '20-25m')

        self.ship_current_vector = \
            [[current_ship_dir_0_5, current_instrument_dir_0_5], [current_ship_dir_5_10, current_instrument_dir_5_10],
             [current_ship_dir_10_15, current_instrument_dir_10_15], [current_ship_dir_15_20,
                                                                      current_instrument_dir_15_20],
             [current_ship_dir_20_25, current_instrument_dir_20_25]]

        #plt.hist(self.current_ship_dir)
        #plt.plot(np.transpose(mean_current_east), depth_turb, label='Mean current east')
        #plt.plot(np.transpose(self.current_ship_dir), depth_turb, label='Current ship direction')
        #plt.legend()
        #plt.show()

        # PLOTTING THE RESULT TO SEE IF THE CALCULATIONS ARE OK
        # The first two plots are countourf plots whit the ends, start and max depth of wake indicated. The second two
        # plots are contourf plots with the wake area indicated by black dots.

        """
        if correction is True:
            plotting_wake_calculation_data_log_single(eps2, turb_time, depth_turb, eps_depth_list, 'eps', self.ship_info[0].name)
            plotting_wake_calculation_data_log_single(uVar2, turb_time, depth_turb, uVar_depth_list, 'uVar', self.ship_info[0].name)
            #plotting_wake_area(eps_wake_calculation, turb_time, depth_turb, eps2, 'eps', self.ship_info[0].name)
            #plotting_wake_area(uVar_wake_calculation, turb_time, depth_turb, uVar2, 'uVar', self.ship_info[0].name)

        elif correction is False:
            plotting_wake_calculation_data_log_single(eps1, turb_time, depth_turb, eps_depth_list, 'eps', self.ship_info[0].name)
            plotting_wake_calculation_data_log_single(uVar1, turb_time, depth_turb, uVar_depth_list, 'uVar', self.ship_info[0].name)
            #plotting_wake_area(eps_wake_calculation, turb_time, depth_turb, eps1, 'eps', self.ship_info[0].name)
            #plotting_wake_area(uVar_wake_calculation, turb_time, depth_turb, uVar1, 'uVar', self.ship_info[0].name)

        else:
            print('Error in plotting turbulent wake')
        """






