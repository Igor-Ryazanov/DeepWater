import numpy as np
from geopy import distance
import math as m
import pickle
from datetime import timedelta, datetime


def find_2_smallest(track_list, rig):
    """
    Finds the two ship objects in the track list that are the closest to the instruments and returns them.
    :param track_list: A list of ship objects in a track.
    :param rig: The name of the rig the distance is calculated to
    :return: The ship objects that are closest to the specified instrument (rig).
    """
    # There should be at least two elements
    arr_size = len(track_list)
    first_ship = []
    second_ship = []
    if rig == 'Lars':
        rig_value = 0
    if rig == 'Anders':
        rig_value = 1

    if arr_size < 2:
        print("Invalid Input, to few ships in track list, check for errors in find_2_smallest")
        return

    first = second = float('inf')
    for i in range(0, arr_size):
        # If current element is smaller than first then
        # update both first and second
        if track_list[i].distance[rig_value][0] < first:
            second = first
            second_ship = first_ship
            first = track_list[i].distance[rig_value][0]
            first_ship = track_list[i]

        # If arr[i] is in between first and second then
        # update second
        elif track_list[i].distance[rig_value][0] < second and track_list[i].distance[rig_value][0] != first:
            second = track_list[i].distance[rig_value][0]
            second_ship = track_list[i]

    if second == float('inf'):
        print("Only one valid value, check for errors in find_2_smallest")

    else:
        return first_ship, second_ship


def find_3_closest(track_list, rig):
    """
    Finds the closest ship object in the track list and the ship object just before and after it and returns them.
    :param track_list: A list of ship objects in a track.
    :param rig: The name of the rig the distance is calculated to
    :return: The three ship objects that are on the track closest to the specified instrument (rig).
    """
    # There should be at least two elements
    arr_size = len(track_list)
    closest_ship = []
    previous_ship = []
    next_ship = []
    first_ship = []
    second_ship = []

    if rig == 'Lars':
        rig_value = 0
    if rig == 'Anders':
        rig_value = 1

    if arr_size < 3:
        print("Invalid Input, to few ships in track list, check for errors in find_2_closest")
        return

    # Find the closest ship
    closest = float('inf')
    for i in range(0, arr_size):
        # If current element is smaller than closest then update closest_ship and closest
        if track_list[i].distance[rig_value][0] < closest:
            closest = track_list[i].distance[rig_value][0]
            closest_ship = track_list[i]

    if closest == float('inf'):
        print("No valid value, check for errors in find_3_closest")
        return

    # Find the ship object previous to and next after the closest ship object (in time)
    for i in range(0, arr_size):
        # If i = closest ship, pick i-1 and 1+1 as the previous and next ship objects.
        if track_list[i] < closest_ship:
            previous_ship = track_list[i-1]
            next_ship = track_list[i+1]

    else:
        return first_ship, second_ship


def distance_calculation(a, b, point, ship_a, ship_b):
    """
    Calculates the shortest distance between the track AB and the point (C) and the coordinates for the closest point on
    the track ab (x). If one of the endpoints (ship_a or ship_b) are closest, the closest of these points are returned.
    The distance is calculated in the following steps:
    1) The lat,lon coordinates for a, b, and point are converted into a local coordinate system, where the instrument
    (point) is the origo. The local coordinate system is a cartesian x,y system with a unit of meters. (should it be km)
    2) The angle (alpha), which is the angle created by the vector AC and AB, is calculated.
    3) The distance between point C and X is calculated using the angle alpha.
    3) The coordinates for X are calculated and then converted into lon, lat.
    4) The distance between point X and point B is calculated [km].
    :param a: A list containing the lat, lon coordinates (decimal degrees) of endpoint a.
    :param b: A list containing the lat, lon coordinates (decimal degrees) of endpoint b.
    :param point: A list containing the latitude and longitude (decimal degrees) of point c.
    :param ship_a: The distance from endpoint a to the point c (km). (used for comparison between spherical calculation
    and planar calculation, same for next parameter)
    :param ship_b: The distance from endpoint b to the point c (km).
    :return: The shortest distance between point C and X (in km as a floating point number), the lat, lon coordinates
    for the point X, the distance between point A and X [km], the distance between point B and X [km].
    """

    if a == b:
        print('ValueError, check function distance_calculation')
    else:

        # DEFINING THE COORDINATES OF THE POINTS IN LAT, LON
        local_origo = point  # The instrument coordinates are used as origo of the local coordinate system [lat, lon]
        point_a = a          # [lat, lon]
        point_b = b          # [lat, lon]
        point_c = point      # [lat, lon]

        # CONVERTING THE COORDINATES OF THE POINTS TO LOCAL COORDINATE SYSTEM IN [km]
        # x = (Lon - Lon_origo) * 1.852 * cos(Lat_origo) [km]
        # y = (Lat - Lat_origo) * 1.852 [km]

        ax = (point_a[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60        # [km]
        ay = (point_a[0] - local_origo[0]) * 1.852 * 60                              # [km]

        bx = (point_b[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60       # [km]
        by = (point_b[0] - local_origo[0]) * 1.852 * 60                               # [km]

        cx = (point_c[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60      # [km]
        cy = (point_c[0] - local_origo[0]) * 1.852 * 60                              # [km]

        # CALCULATING VECTORS AND NORMS USED TO FIND DISTANCE CX AND COORDINATES FOR X (all in local coordinate system)
        # Calculating vectors AC, BC, and AB [km]
        vector_ac = np.array([cx - ax, cy - ay])
        vector_bc = np.array([cx - bx, cy - by])
        vector_ab = np.array([bx - ax, by - ay])

        # Calculating the length/norm of the vectors AC, BC, and AB (in km)
        length_ac_test = ship_a         # (Used for comparison only, the legth in ship_a and b are calculated using
        length_bc_test = ship_b         # spherical coordinates)
        length_ac = np.linalg.norm(vector_ac)
        length_bc = np.linalg.norm(vector_bc)
        length_ab = np.linalg.norm(vector_ab)

        # A print test to see that the calculated distances for the two methods are similar (they should be) [km]
        print('NEW TRACK')
        #print('Spherical distance Ship a: ' + str(ship_a) + ', local distance Ship a: ' + str(length_ac))
        #print('Spherical distance Ship b: ' + str(ship_b) + ', local distance Ship b: ' + str(length_bc))

        # CALCULATING THE ANGLE ALPHA, THE DISTANCE CX AND THE COORDINATES FOR X (still in local coordinates)
        # Calculating the angle alpha in RADIANS
        # A * B = norm(A) * norm(B) * cos(alpha)
        alpha_r = m.acos(np.dot(vector_ac, vector_ab) / (length_ac * length_ab))
        alpha_d = m.degrees(alpha_r)  # Might be a parameter we want to use later, so better save it too

        print('alpha = ' + str(alpha_d))

        # Calculate the distance CX using the angle alpha in [m]
        # CX = norm(A) * sin(alpha)
        length_cx = length_ac * m.sin(alpha_r)  # The shortest distance between line AB and point C (cx)
        length_ax = length_ac * m.cos(alpha_r)  # The distance from point A to X. Needed to find the coordinates for X

        # Calculate the coordinates for X
        # X = a + (ax/ab) * (b - a),
        # Take the coordinates of point a and add a scaled vector representing the distance between point a and point x
        # B - A = point b - point a = vector_ab,
        # ax/ab is the ratio between the length/norm of ax and the entire ab vector, which gives the "scaling" of the
        # vector ab that is added to point/coordinates in a
        point_x_loc = np.array([ax, ay]) + (length_ax / length_ab) * vector_ab

        # Convert coordinates for point X into lat, lon coordinates
        # Lon = (x_coord_loc / (1852 * cos(Lat_origo))) + Lon_origo
        # Lat = ( y_coord_loc / 1852 ) + Lat_origo
        point_x_lat = (point_x_loc[1] / (1.852 * 60)) + local_origo[0]
        point_x_lon = (point_x_loc[0] / (1.852 * 60 * m.cos(local_origo[0])) + local_origo[1])
        point_x = [point_x_lat, point_x_lon]

        # Calculate the distance between the closest point and the instruments using geopy distance.distance
        shortest_dist = distance.distance([point_x_lat, point_x_lon], point).km     # Only used for comparison!

        # Calculate the length of bx (it will be needed in the next step), alternative using pythagoras theorem
        length_bx = length_ab - length_ax                                # m.sqrt(length_cx**2 - length_bc**2)

        # Comparison between the spherical and planar calculation for the shortest distance
        #print('Spherical shortest distance: ' + str(shortest_dist) + ', planar shortest distance: ' + str(length_cx))

        shortest_distance_loc = min(length_cx, length_ac, length_bc)
        shortest_distance = min(length_cx, ship_a, ship_b)

        return min(length_cx, ship_a, ship_b), point_x, length_ax, length_bx


def distance_to_instruments_planar(shiplist, rig_list):
    """
    Calculate the closest distance [km] between the instruments and all the ship tracks in the shiplist. The ship track
    is given by the coordinates in the shiplist, and the coordinates of the instruments are given by the rig_list.
    :param shiplist: A a list of shiplists where each list contains measurements from positions within a certain radius
     of the instruments listed in the rig_list. The coordinates in each list are from the the same ship passage/track.
    :param rig_list: A list of tuples, each tuple contains information about an instrument (longitude, latitude, name).
    :return: A list of tuples containing the following 11 parameters:
                0: A list of the lat/lon coordinates used to calculate the shortest distance, can be used for plotting
                1: The name of the ship
                2: Date and time for the closest point (tuple containing a datenum object for each rig in the rig list)
                3: MMSI
                4: Length
                5: Width
                6: Static draught
                7: Speed over ground (based on the speed at the point/measurement just before the closest point)
                8: IMO
                9: Distance to Lars instrument [km]
                10: Distance to Anders instrument [km]
                11: Ship type (str)
    """

    # Setting the parameters for the rigs
    L_rig = rig_list[0]
    Lars = (L_rig[0], L_rig[1])       # Longitude and latitude in decimal degrees
    A_rig = rig_list[1]
    Anders = (A_rig[0], A_rig[1])     # Longitude and latitude in decimal degrees
    new_list = []

    # A track is a list of ships (xx positions for the same ship corresponding to the closest track to the instruments)
    for track in shiplist:
        position_info = []
        segments = []

        # The following if-statements cleans the dataset so only valid tracks are kept + save coordinates for plotting.
        iteration = 0
        if len(track) < 2:              # If there is only one measurement the calculation cannot be performed
            pass
        elif len(track) < 3 and track[iteration] == track[iteration + 1]:
            pass
            # If there are only two measurements/ships and they are identical, then the track is skipped
        else:
            for ship in track:
                if iteration < (len(track) - 1) and track[iteration] == track[iteration + 1]:
                    iteration += 1
                    # Skip the ship-objects that are identical with the previous ship-object
                elif iteration < (len(track) - 1) and track[iteration].longitude == track[iteration + 1].longitude and \
                        track[iteration].latitude == track[iteration + 1].latitude:
                    iteration += 1
                    # Skip the ship-objects which have coordinates identical with the previous ship-object, as the
                    # distance between such points will be zero (and give Error in the calculations)
                else:
                    position_info.append(ship)

                    # Save coordinates for plotting the track
                    segments.append([ship.latitude, ship.longitude])
                    iteration += 1

        # This if-statement checks again so that there are no tacks with too few items
        if len(position_info) < 2 and len(segments) < 2:
            pass
        else:
            # Using the cleaned track list (position_info), the two ship objects closest to the two instruments are
            # identified, and they will be used to calculate the shortest distance to the instruments.
            # The distance used for finding the two closest points is calculated using the function distance.distance
            # which uses spherical coordinates. This is because that calculation has been done in a previous step,
            # which is convenient as the only thing needed is the lat,lon coordinates of the points of interest.

            # See the function find_2_smallest() for a description of the calculation
            lars_ship_a, lars_ship_b = find_2_smallest(position_info, 'Lars')
            anders_ship_a, anders_ship_b = find_2_smallest(position_info, 'Anders')

            # Calculate the point on the line AB that is closest to C (either of the riggs), using the function Lars
            # gave for the shortest distance from a straight line to a point. See the function distance_calculation()
            # for details about the calculation.

            lars_dist, lars_closest_point, lars_dist_ship_a, lars_dist_ship_b = distance_calculation([lars_ship_a.latitude, lars_ship_a.longitude], [lars_ship_b.latitude, lars_ship_b.longitude],
                                    Lars, (lars_ship_a.distance[0][0]) / 1000, (lars_ship_b.distance[0][0]) / 1000)

            anders_dist, anders_closest_point, anders_dist_ship_a, anders_dist_ship_b = distance_calculation([anders_ship_a.latitude, anders_ship_a.longitude], [anders_ship_b.latitude, anders_ship_b.longitude],
                                Anders, (anders_ship_a.distance[0][0]) / 1000, (anders_ship_b.distance[0][0]) / 1000)

            # Calculating the date and time for the closest position. See function time_closest_point() for details.
            lars_date_closest_point, lars_info_ship = time_closest_point(lars_ship_a, lars_ship_b, lars_closest_point, lars_dist_ship_a, lars_dist_ship_b)
            anders_date_closest_point, anders_info_ship = time_closest_point(anders_ship_a, anders_ship_b, anders_closest_point, anders_dist_ship_a, anders_dist_ship_b)

            # Append the information for the track to the new_list
            # 0: line segments, 1: ship name, 2: list of date of closest point for [Lars, Anders], 3: MMSI, 4: length,
            # 5: width, 6: static draught, 7: A list of the speed over ground at the first of the two closest
            # ship objects for each rig [Lars, Anders], 8: IMO number, 9: distance to Lars instrument,
            # 10: distance to Anders instrument, 11: ship type
            new_list.append([segments, lars_info_ship.name, [lars_date_closest_point, anders_date_closest_point],
                            lars_info_ship.MMSI, lars_info_ship.length, lars_info_ship.width,
                             lars_info_ship.static_draught, [lars_info_ship.sog, anders_info_ship.sog],
                             lars_info_ship.IMO, lars_dist, anders_dist, lars_info_ship.ship_type_t])

    return new_list


def time_closest_point(ship_a, ship_b, closest_point, dist_ship_a, dist_ship_b):
    """
    Calculates the time at which the ship is the closest point.
    :param ship_a: One of the two closest ship objects.
    :param ship_b: The other of the two closest ship objects.
    :param closest_point: The lat, lon coordinates of the closest point on the track. (The one we want the time for)
    :param dist_ship_a: The distance between ship object a and the closest point [km].
    :param dist_ship_b: The distance between the ship object b and the closest point [km].
    :return: The date for when the ship passes the closest point (datetime.datetime object) and the ship object that
    preceded the closest point (in time).
    """

    # If the closest point is the position of the ship_a object, the date for ship_a and the ship_a object is returned
    if closest_point[0] == ship_a.latitude and closest_point[1] == ship_a.longitude:
        return ship_a.date, ship_a

    # If the closest point is the position of the ship_b object, the date for ship_b aand the ship_b object is returned
    elif closest_point[0] == ship_b.latitude and closest_point[1] == ship_b.longitude:
        return ship_b.date, ship_b

    else:
        # Average speed for the track (should I use a weighted measure instead?)
        av_speed = (ship_a.sog * ship_b.sog) / 2                        # Average speed over ground in knots

        # Find ship-object with the earliest date. Starting date and distance to the instruments will be calculated
        # form this point
        if min(ship_a.date, ship_b.date) == ship_a.date:
            first_point = ship_a
            first_distance = dist_ship_a
        elif min(ship_a.date, ship_b.date) == ship_b.date:
            first_point = ship_b
            first_distance = dist_ship_b
        else:
            print('Error, check function time_closest_point')

        # The date is calculated as: date = first date + time taken to reach the closest point.
        # Create a timedelta object using the speed and distance traveled to get the hours traveled.
        # timedelta = distance from the first point to closest point / (speed * conversion factor (1 knot = 1.852 km/h))
        date_closest_point = first_point.date + timedelta(hours=(first_distance / (av_speed * 1.852)))

    return date_closest_point, first_point


# RUN CODE
# Loading data file
filename = 'Example_data_distance'
infile = open(filename, 'rb')
master_ship_list = pickle.load(infile)
infile.close()

# setting parameters
lars_rigg_lat, lars_rigg_lon = 57.61178299617746, 11.661020044367095
anders_rigg_lat, anders_rigg_lon = 57.61254996208881, 11.662700022640175
rigg_list = [(57.61178299617746, 11.661020044367095, 'Lars rigg'), (57.61254996208881, 11.662700022640175, 'Anders rigg')]

# Run the distance calculation
ship_list_with_distance_calculated = distance_to_instruments_planar(master_ship_list, rigg_list)


# After this a use a series of functions to plot the data, but I am not sure how you want to check the result.
# The code below plot the distance to your instrument in km, for each track in the ship_list_with_distance_calculated.
#for ship in ship_list_with_distance_calculated:
#    print(ship[9])




