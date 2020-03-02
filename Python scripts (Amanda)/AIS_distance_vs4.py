import numpy as np
from geopy import distance
import math as m
from operator import itemgetter
from datetime import timedelta, datetime
import os
os.environ['PROJ_LIB'] = r'\Users\nylunda\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.1.0-hfa6e2cd_1\Library\share'


def find_3_closest(track_list, rig):
    """
    Finds the closest ship object to the specified instrument, and the ship object just before and just after that
    point, and returns them. There can be three or two returned values.
    :param track_list: A list of ship objects in a track.
    :param rig: The name of the rig the distance is calculated to
    :return: The ship objects that are closest to the specified instrument (rig). It can be one, two or three objects.
    """

    # There should be at least two elements
    arr_size = len(track_list)
    closest_ship = None
    previous_ship = None
    next_ship = None

    if rig == 'Lars rigg':
        rig_value = 0
    if rig == 'Anders rigg':
        rig_value = 1

    if arr_size < 2:
        print("Invalid Input, to few ships in track list, check for errors in find_3_closest")
        return

    closest = float('inf')
    item = 0
    for i in track_list:
        # If current element is smaller than closest then update the closest value and the closest ship.
        # Also update the previous and next ship objects.

        if i.distance[rig_value][0] < closest:

            # If the closest element is the first element there will be no previous_ship
            if i == track_list[0]:
                closest = i.distance[rig_value][0]
                closest_ship = i

                if i.latitude == track_list[item + 1].latitude and i.longitude == track_list[item + 1].longitude:
                    items_left = len(track_list)-item
                    for value in range(1, items_left):
                        if i.latitude == track_list[item + value].latitude and \
                                i.longitude == track_list[item + value].longitude:
                            pass
                        else:
                            next_ship = track_list[item + value]
                            break
                else:
                    next_ship = track_list[item + 1]

            # If the closest element is the last element there will be no next_ship
            elif i == track_list[-1]:
                closest = i.distance[rig_value][0]
                closest_ship = i
                previous_ship = track_list[item - 1]

            else:
                closest = i.distance[rig_value][0]
                closest_ship = i
                previous_ship = track_list[item - 1]
                next_ship = track_list[item + 1]
        item += 1

    if closest == float('inf'):
        print("No  valid value, check for errors in find_2_smallest")

    # If there are only two values, the closest object will come first.
    elif previous_ship is None:
        return closest_ship, next_ship

    elif next_ship is None:
        return closest_ship, previous_ship

    else:
        return previous_ship, closest_ship, next_ship


def longlat_to_cartesian(lat, lon):
    """
    Converts coordinates in longitude and latitude into cartesian coordinates. Requires the math and geopy package.
    The calculation is assuming a spherical earth.
    :param lon: Longitude in degrees.
    :param lat: Latitude in degrees.
    :return: x, y, z cartesian coordinates (radians).
    """

    radius = distance.EARTH_RADIUS      # in km
    x = radius * m.cos(m.radians(lat)) * m.cos(m.radians(lon))
    y = radius * m.cos(m.radians(lat)) * m.sin(m.radians(lon))
    z = radius * m.sin(m.radians(lat))

    return x, y, z


def distance_cx_and_coord_x(magnitude_ac, alpha, magnitude_ab, vector_ab, ax, ay, local_origo):
    """
    Calculates the shortest distance CX between the line AB and the point C (instrument) in the triangle ABC, where X
    is the point on the line AB with the shortest distance to C. C is origo in the local coordinate system.
    Returns the shortest distance in km (CX), the coordinates för the point X (lat, lon), and the distance between
    point X and point B (BX) in km.
    :param magnitude_ac:    The magnitude of the vector AC [km] in the triangle ABC.
    :param alpha:           The angle between the vector AC and AB in the triangle ABC in radians.
    :param magnitude_ab:    The magnitude of the vector AB [km] in the triangle ABC.
    :param vector_ab:       The vector AB in the local coordinate system.
    :param ax:              X-coordinates for point A in the triangle ABC in the local coordinate system.
    :param ay:              Y-coordinates for point A in the triangle ABC in the local coordinate system.
    :param local_origo:     The coordinates of the origo in the local coordinate system in latitude and longitude.
    :return:                The shortest distance in km (length_cx), the coordinates för the point X (point_x) in lat
                            and lon, the distance between point X and point A in km (length_ax), and the distance
                            between point X and point B in km (length_bx).
    """
    # Calculate the distance CX using the angle alpha in [m]
    # CX = norm(A) * sin(alpha)

    length_cx = magnitude_ac * m.sin(alpha)  # The shortest distance between line AB and point C (cx) [m]
    length_ax = magnitude_ac * m.cos(alpha)  # The dist from point A to point X [m]. To find the coordinates for X

    # Calculate the coordinates for X
    # X = A + (AX/AB) * (B - A), --> Take the coordinates of point A and add a scaled vector representing
    # the distance between point A and point X
    # B - A = point b - point a = vector_ab,
    # ax/ab is the ratio between the length/norm of ax and the entire ab vector,
    # which gives the "scaling" of the vector ab that is added to point/coordinates in A.

    point_x_loc = np.array([ax, ay]) + (length_ax / magnitude_ab) * vector_ab

    # Convert coordinates for point X into lat, lon coordinates
    # Lon = (x_coord_loc / (1852 * cos(Lat_origo))) + Lon_origo
    # Lat = ( y_coord_loc / 1852 ) + Lat_origo

    point_x_lat = ((point_x_loc[1] / (1.852 * 60)) + local_origo[0])
    point_x_lon = (point_x_loc[0] / (1.852 * 60 * m.cos(local_origo[0])) + local_origo[1])
    point_x = [point_x_lat, point_x_lon]

    # Calculate the length of bx (it will be needed in the next step), alternative using pythagoras theorem
    # length_bx = abs(magnitude_ab - length_ax)
    length_bx = magnitude_ab - length_ax  # m.sqrt(length_cx**2 - magnitude_bc**2)

    # print('alpha: ' + str(m.degrees(alpha)) + ', CX: ' + str(length_cx) + ', AX: ' + str(length_ax) + ', AC: ' +
    #      str(magnitude_ac) + ', AB: ' + str(magnitude_ab) + ', BX: ' + str(length_bx))

    return length_cx, point_x, length_ax, length_bx


def distance_calculation(a, b, d, point):
    """
    Calculates the shortest distance between the point (C) and the line/track between the points a and b (AB) and/or
    the line between point d and b (DB). If one of the endpoints (a, b, or d) is the closest point, the closest of these
    points are returned. The distance is calculated in the following steps:

    1) The lon, lat coordinates are converted into local planar coordinates [km], using the point C as the origo.
    2) Vectors and norms are calculated for all sides in the triangles ABC (and BCD if there is a point D).
    3) The angle alpha is calculated for the angles CÂB and C^BA using the formula: AC*AB = norm(AC)*norm(AB)*cos(alpha)
       (for CÂB AC is switched to BC). If there is a point d, the alpha is also calculated for the angles C^BD and C^DB.
    4) Calculate the shortest distance (CX) between the line formed by AB (or BD) and the point C, and find the
       coordinates for point X. If one of the endpoints (A, B, or D) is the closest point, the distance to C from that
       point and its coordinates are returned. The distance and coordinates are calculated using the formula
       distance_cx_and_coord_x. If any of the alphas in each of the two triangles (ABC or BCD) are > 90 degrees, one of
       the points (A, B, or D) is the closest point, as the line created by the two points has its closest point
       "outside" the triangle. This is considered in the calculation.
    5) The shortest distance in the two triangles are compared, and the distance and coordinates of the shortest are
       returned. The distance is in km and the coordinates in lon, lat.

    Requires the math and numpy package.
    :param a: A list containing the lat, lon coordinates (decimal degrees) of endpoint A.
    :param b: A list containing the lat, lon coordinates (decimal degrees) of endpoint B.
    :param d: A list containing the lat, lon coordinates (decimal degrees) of endpoint D.
    :param point: A list containing the latitude and longitude (degrees) of point C (the instrument).
    :return: The shortest distance to the point C in km as a floating point number (closest_distance_xxx), the lat, lon
    coordinates for the point on the track closest to the instrument X (closest_point_xxx), the distance between the
    point A and point X in km (length_ax), the distance between point B and point X in km (length_bx), the distance
    between point D and point X in km (length_dx), and the triangle where the closest distance is found (ABC or BCD).
    """

    if a == b:                  # or a == point or b == point:
        print('ValueError: a == b')

    else:
        if d[0] is not None and b == d:
            d[0] = None
        # DEFINING THE COORDINATES OF THE POINTS IN LAT, LON
        local_origo = point  # The coords. of the instrument are used as the origo of the local coord. system [lat, lon]
        point_a = a          # [lat, lon]
        point_b = b          # [lat, lon]
        point_c = point      # [lat, lon]

        # CONVERTING THE COORDINATES OF THE POINTS TO LOCAL COORDINATE SYSTEM IN [km]
        # x = (Lon - Lon_origo) * 1.852 * cos(Lat_origo) [km]
        # y = (Lat - Lat_origo) * 1.852 [km]

        ax = (point_a[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60         # [km], test to remove * 1852
        ay = (point_a[0] - local_origo[0]) * 1.852 * 60                                 # [km]

        bx = (point_b[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60         # [km]
        by = (point_b[0] - local_origo[0]) * 1.852 * 60                                 # [km]

        cx = (point_c[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60         # [km]
        cy = (point_c[0] - local_origo[0]) * 1.852 * 60                                 # [km]

        if d[0] is not None:
            point_d = d                                                                 # [lat, lon]
            dx = (point_d[1] - local_origo[1]) * m.cos(local_origo[0]) * 1.852 * 60     # [km]
            dy = (point_d[0] - local_origo[0]) * 1.852 * 60                             # [km]

        # CALCULATING VECTORS AND NORMS USED TO FIND THE DISTANCE CX AND COORDINATES FOR X (all in local coord. system)
        # Calculating vectors AC, BC, and AB [km]
        vector_ac = np.array([cx - ax, cy - ay])
        vector_bc = np.array([cx - bx, cy - by])
        vector_ab = np.array([bx - ax, by - ay])

        if d[0] is not None:
            # As the b object is the closest point when there are 3 objects, the distance DB is the equivalent to AB.
            vector_dc = np.array([cx - dx, cy - dy])
            vector_db = np.array([bx - dx, by - dy])

        # Calculating the magnitude/norm/length of the vectors AC, BC, and AB (in km) (always positive)
        magnitude_ac = np.linalg.norm(vector_ac)
        magnitude_bc = np.linalg.norm(vector_bc)
        magnitude_ab = np.linalg.norm(vector_ab)

        if d[0] is not None:
            # Calculating the magnitude/norm/length of the vectors DC and DB (in km)
            magnitude_dc = np.linalg.norm(vector_dc)
            magnitude_db = np.linalg.norm(vector_db)

        # CALCULATING THE ANGLE ALPHA
        # Calculating the angle alpha in RADIANS
        # A * B = norm(A) * norm(B) * cos(alpha)        (norm(A) = magnitude_a)

        # As A*B != B*A, two alphas must be calculated for each pair of ship objects (A and B, and B and D)
        alpha_r_ab = m.acos(np.dot(vector_ac, vector_ab) / (magnitude_ac * magnitude_ab))
        alpha_r_ba = m.acos(np.dot(vector_bc, -vector_ab) / (magnitude_bc * magnitude_ab))

        if d[0] is not None:
            alpha_r_db = m.acos(np.dot(vector_dc, vector_db) / (magnitude_dc * magnitude_db))
            alpha_r_bd = m.acos(np.dot(vector_bc, -vector_db) / (magnitude_bc * magnitude_db))

        # If any of the alphas in each of the two triangles (ABC or BCD) are > 90 degrees, one of the points (A or B, or
        # B or D) is the closest point, as the line created by the two points has its closest point "outside" the
        # triangle. Thus the angles are converted to degrees to check the triangles.
        alpha_d_ab = m.degrees(alpha_r_ab)
        alpha_d_ba = m.degrees(alpha_r_ba)

        if d[0] is not None:
            alpha_d_db = m.degrees(alpha_r_db)
            alpha_d_bd = m.degrees(alpha_r_bd)

        # Test print to see if the angles varies (they should)
        # print('alpha ab = ' + str(alpha_d_ab) + ', alpha ba = ' + str(alpha_d_ba))
        # if d is not None:
        #   print('alpha db = ' + str(alpha_d_db) + ', alpha bd = ' + str(alpha_d_bd))

        # CALCULATING/FINDING THE SHORTEST DISTANCE CX (OR ONE OF THE OBJECTS) AND THE COORDINATES FOR THE
        # CLOSEST POINT X (OR ONE OF THE OBJECTS) (still in local coordinates)

        # Each triangle is checked in turn and the closest point found (either one of the objects or the point X)

        # First the shortest distance and closes point in the ABC triangle is found
        if alpha_d_ab > 90 or alpha_d_ba > 90:
            # print('More than 90 in ABC')
            if magnitude_ac < magnitude_bc:
                # print('ABC: point a = closest distance abc')
                # print(magnitude_ac)
                closest_distance_abc = magnitude_ac
                # print(magnitude_ac, magnitude_bc)
                closest_point_abc = point_a
                length_ax = None             # As the point A = X
                length_bx_abc = magnitude_ab     # As the point A = X
                length_dx = None
            elif magnitude_bc < magnitude_ac:
                # print('ABC: point b = closest distance abc')
                # print(magnitude_bc)
                closest_distance_abc = magnitude_bc
                # print(magnitude_bc, magnitude_ac)
                closest_point_abc = point_b
                length_bx_abc = None             # As the point B = X
                length_ax = magnitude_ab     # As the point B = X
                length_dx = None
            else:
                print('Error in distance_calculation comparing angles in ABC')

        # If both angles are < 90 degrees, the closest distance CX is calculated, and the coordinates for the closest
        # point X are also calculated and converted to from the local coordinate system to long, lat
        else:
            # print('abc calculation')
            length_cx_ab, point_x_ab, length_ax_ab, length_bx_ab = distance_cx_and_coord_x(magnitude_ac, alpha_r_ab,
                                                                        magnitude_ab, vector_ab, ax, ay, local_origo)
            # NOTE, here the length bx and ba are switched, as the function returns the length to X from the point given
            # in the first magnitude vector (the one that is not C, in this case B, as the vector is BC) as the vector a
            length_cx_ba, point_x_ba, length_bx_ba, length_ax_ba = distance_cx_and_coord_x(magnitude_bc, alpha_r_ba,
                                                                        magnitude_ab, -vector_ab, bx, by, local_origo)

            # print('alphaAB: CX: ' + str(length_cx_ab) + ' AX: ' + str(length_ax_ab) + ' BX: ' + str(length_bx_ab))
            # print('alphaBA: CX: ' + str(length_cx_ba) + ' AX: ' + str(length_ax_ba) + ' BX: ' + str(length_bx_ba))

            if length_cx_ab < length_cx_ba:
                # print(length_cx_ab)
                closest_distance_abc = length_cx_ab
                closest_point_abc = point_x_ab
                length_ax = length_ax_ab
                length_bx_abc = length_bx_ab
                length_dx = None

            else:
                # print(length_cx_ba)
                closest_distance_abc = length_cx_ba
                closest_point_abc = point_x_ba
                length_ax = length_ax_ba
                length_bx_abc = length_bx_ba
                length_dx = None

        # print('ABC: length_ax = ' + str(length_ax) + ', length_bx: ' + str(length_bx_abc))

        if d[0] is not None:
            # If there are three points, the shortest distance is calculated for the BCD triangle as well, and that
            # distance is then compared to the ABC triangle, returning the shortest distance of the two.

            if alpha_d_db > 90 or alpha_d_bd > 90:
                # print('More than 90 in BCD')
                if magnitude_bc > magnitude_dc:
                    # print('BCD: point d = closest distance bcd')
                    # print(magnitude_dc)
                    closest_distance_bcd = magnitude_dc
                    # print(magnitude_dc, magnitude_bc)
                    closest_point_bcd = point_d
                    length_dx = None             # As the point D = X
                    length_bx_bcd = magnitude_db     # As the point D = X
                elif magnitude_bc < magnitude_dc:
                    # print('BCD: point b = closest distance bcd')
                    # print(magnitude_bc)
                    closest_distance_bcd = magnitude_bc
                    # print(magnitude_bc, magnitude_dc)
                    closest_point_bcd = point_b
                    length_bx_bcd = None             # As the point B = X
                    length_dx = magnitude_db     # As the point B = X
                else:
                    print('Error in distance_calculation in comparing angles BCD')

            else:
                # print('bcd calculation')
                length_cx_bd, point_x_bd, length_bx_bd, length_dx_bd = distance_cx_and_coord_x(magnitude_bc, alpha_r_bd,
                                                                                               magnitude_db, -vector_db,
                                                                                               bx, by, local_origo)
                # NOTE, here the length dx and bx are switched, as the function returns the length to X from the point
                # given in the first magnitude vector (the one that is not C, in this case D, as the vector is DC)
                # as the vector a.
                length_cx_db, point_x_db, length_dx_db, length_bx_db = distance_cx_and_coord_x(magnitude_dc, alpha_r_db,
                                                                                               magnitude_db, vector_db,
                                                                                               dx, dy, local_origo)
                # print(length_cx_bd, length_cx_db)

                if length_cx_bd < length_cx_db:
                    # print(length_cx_bd)
                    closest_distance_bcd = length_cx_bd
                    closest_point_bcd = point_x_bd
                    length_dx = length_dx_bd
                    length_bx_bcd = length_bx_bd

                else:
                    # print(length_cx_db)
                    closest_distance_bcd = length_cx_db
                    closest_point_bcd = point_x_db
                    length_dx = length_dx_db
                    length_bx_bcd = length_bx_db

            # print('Closest distance abc and bcd')
            # print(closest_distance_abc, closest_distance_bcd)

            # print('BCD: length_ax = ' + str(length_ax) + ', length_bx: ' + str(length_bx_bcd))

            if closest_distance_abc <= closest_distance_bcd:
                # print('final closest distance abc:')
                # print(closest_distance_abc)
                # print('Triangle = ABC, length_ax: ' + str(length_ax) + ', length_bx: ' + str(length_bx_abc) +
                #      ', length_dx: ' + str(length_dx))
                return closest_distance_abc, closest_point_abc, length_ax, length_bx_abc, length_dx, 'ABC'
            elif closest_distance_abc > closest_distance_bcd:
                # print('final closest distance bcd:')
                # print(closest_distance_bcd)
                # print('Triangle = BCD, length_ax: ' + str(length_ax) + ', length_bx: ' + str(length_bx_bcd) +
                #      ', length_dx: ' + str(length_dx))
                return closest_distance_bcd, closest_point_bcd, length_ax, length_bx_bcd, length_dx, 'BCD'
            else:
                print('Error in distance_calculation when comparing the two triangles closest distance')

        else:
            # print('final closest distance abc:')
            # print(closest_distance_abc)
            # print('Triangle = ABC, length_ax: ' + str(length_ax) + ', length_bx: ' + str(length_bx_abc) +
            #      ', length_dx: ' + str(length_dx))
            return closest_distance_abc, closest_point_abc, length_ax, length_bx_abc, length_dx, 'ABC'


def track_calculation(position_info, rig_name, rig_coords):
    """

    :param position_info: A list of the ship objects belonging to the same track.
    :param rig_name: The name of the rig (either 'Lars' or 'Anders').
    :param rig_coords: The coordinates of the rig in lat, lon (decimal degrees).
    :return: A list containing the following 11 parameters:
                0: The name of the ship [string]
                1: Date and time for the closest point on the "track" [datenum]
                2: MMSI (based on the last position before the closest point)
                3: Length (based on the last position before the closest point)
                4: Width (based on the last position before the closest point)
                5: Static draught (based on the last position before the closest point)
                6: Speed over ground (average between the points used to calculate the closest point, or if the closest
                   point is one of the ship objects, that points speed is given)
                7: IMO (number based on the last position before the closest point)
                8: Distance to the instrument [km]
                9: Ship type (str)
                10: Coordinated for the closest point [lat, lon]
                11: The two ship objects used to calculate the closest point
                12: Course over ground (cog), the direction of the ship [degrees]
    """
    ship_a = None
    ship_b = None
    ship_d = None
    ships = find_3_closest(position_info, rig_name)

    # As the output from find_3_closest can be 2 OR 3 ship objects, two if-statements are needed.
    if len(ships) == 2:
        ship_a, ship_b = ships[0], ships[1]
        ship_d = None
        d_lat = None
        d_lon = None

    elif len(ships) == 3:
        ship_a, ship_b, ship_d = ships[0], ships[1], ships[2]
        d_lat = ship_d.latitude
        d_lon = ship_d.longitude

    else:
        print('Error in number of closest ships (not 2 or 3), check function track_calculation')

    # Calculate the point on the line AB or BD that is nearest to point C (either of the riggs),
    # using trigonometry (shortest distance from a straight line to a points):

    dist, closest_point, dist_ship_a, dist_ship_b, dist_ship_d, ship_triangle = \
        distance_calculation([ship_a.latitude, ship_a.longitude], [ship_b.latitude, ship_b.longitude],
                             [d_lat, d_lon], rig_coords)

    # print(ship_triangle)
    if ship_triangle == 'ABC':
        if ship_d is not None:
            ship_three = ship_d
        else:
            ship_three = None

        if dist_ship_a is None:
            # print('ship_a is the closest point')
            # print(ship_a.name)
            # print('NEW LIST')
            date_closest_point = ship_a.date
            info_ship = ship_a
            ship_one = ship_a
            ship_two = ship_b

        elif dist_ship_b is None:
            # print('ship_b is the closest point')
            # print(ship_b.name)
            # print('NEW LIST')
            date_closest_point = ship_b.date
            info_ship = ship_b
            ship_one = ship_a
            ship_two = ship_b

        # Calculating the date and time for the closest position using the time_closest_point function
        else:
            # print('a point in ABC is the closest point')
            # print(ship_a.name)
            # print('NEW LIST')
            date_closest_point, info_ship = time_closest_point(ship_a, ship_b, closest_point, dist_ship_a, dist_ship_b)
            ship_one = ship_a
            ship_two = ship_b

    elif ship_triangle == 'BCD':
        ship_one = ship_a
        if dist_ship_b is None:
            # print('ship_b is the closest point')
            # print(ship_b.name)
            # print('NEW LIST')
            date_closest_point = ship_b.date
            info_ship = ship_b
            ship_three = ship_d
            ship_two = ship_b

        elif dist_ship_d is None:
            # print('ship_d is the closest point')
            # print(ship_d.name)
            # print('NEW LIST')
            date_closest_point = ship_d.date
            info_ship = ship_d
            ship_three = ship_d
            ship_two = ship_b

        else:
            # print('a point in BCD is the closest point')
            # print(ship_b.name)
            # print('NEW LIST')
            date_closest_point, info_ship = time_closest_point(ship_b, ship_d, closest_point, dist_ship_b, dist_ship_d)
            ship_three = ship_d
            ship_two = ship_b

    # Appends the information for the track to a tuple that is returned
    # 0: ship name, 1: date of closest point , 2: MMSI, 3: length, 4: width, 5: static draught,
    # 6: speed over ground at the closest point, 7: IMO number, 8: distance to the instrument, 9: ship type,
    # 10: coordinated for the closest point, 11: the two ship-objects used to calculate the closest point.
    return [info_ship.name, date_closest_point, info_ship.MMSI, info_ship.length, info_ship.width,
            info_ship.static_draught, info_ship.sog, info_ship.IMO, dist, info_ship.ship_type_t, closest_point,
            [ship_one, ship_two, ship_three], info_ship.cog]
                     

def distance_to_instruments_sphere(shiplist, rig_list):
    """
    Takes the shiplist, and for each list of ship objects in the list (all objects belong to the same track) a tuple is
    added to the new_list. Each tuple in the new_list contains information about the track and ship object. The closest
    point on the track and the corresponding shortest distance between the instruments in the rig_list, are calculated.
    The returned list of tuples contains all the information needed to plot the tracks in the ship list with a legend.
    :param shiplist: A a list of shiplists where each list contains measurements from positions within a certain radius
     of the instruments listed in the rig_list, all from the the same ship passage. Lists of ship objects.
    :param rig_list: A list of tuples, each tuple contains information about an instrument (longitude, latitude, name).
    :return: A list of tuples containing the following 12 parameters:
                0: A list of the lat/lon coordinates used to calculate the shortest distance, can be used for plotting
                   the track
                1: The name of the ship
                2: A list with the date and time for the closest point on the "track" [datenum] for each instrument.
                3: MMSI (based on the last position before the closest point)
                4: Length (based on the last position before the closest point)
                5: Width (based on the last position before the closest point)
                6: Static draught (based on the last position before the closest point)
                7: A list with the speed over ground at the closest point for each of the instruments (average between
                   the points used to calculate the closest point, or if the closest point is one of the ship objects,
                   that points speed is given)
                8: IMO (number based on the last position before the closest point)
                9: Distance to Lars instrument [km]
                10: Distance to Anders instrument [km]
                11: Ship type (str)
                12: A list of coordinates for the closest point on the track to each instrument in lat, lon.
                13: A list with the ship objects of the two points used to calculate the closest point in lat, lon.
                14: Course over ground (cog), the direction of the ship [degrees]
                15: position_info, the ship objects related to the coordinates in the ship track.
    """

    # Converting the positions of the rigs into cartesian coordinates
    L_rig = rig_list[0]
    Lars = (L_rig[0], L_rig[1])       # Longitude and latitude in decimal degrees
    A_rig = rig_list[1]
    Anders = (A_rig[0], A_rig[1])
    new_list = []

    # A track is a list of ships (xx positions for the same ship corresponding to the closest track to the instruments)
    for track in shiplist:
        # Position info is a list of ship objects
        position_info = []
        segments = []

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
                    # position_info.append([lon, lat, ship.distance, time, name, MMSI, length, width, s_draught,
                    # sog, IMO, ship, ship_type])

                    # Save coordinates for plotting the track
                    segments.append([ship.latitude, ship.longitude, ship.date, ship])     # TODO: Added ship.date & ship
                    iteration += 1

        if len(position_info) < 2 and len(segments) < 2:
            pass
        else:
            # Using the new checked track list (position_info), the two or three ship objects closest to the two
            # instruments are found, and they will be used to calculate the shortest distance to the instruments.
            # L_rig[2] gives the name of the rig (either "Anders rigg" or "Lars rigg", and the name is the coordinates
            # for the rig.

            lars_track = track_calculation(position_info, L_rig[2], Lars)
            anders_track = track_calculation(position_info, A_rig[2], Anders)

            # Appends the information for the track to the new_list
            # 0: line segments for the track (including corresponding date and ship object), 1: ship name,
            # 2: list of date of closest point for [Lars, Anders],
            # 3: MMSI, 4: length, 5: width, 6: static draught, 7: A list of the speed over ground at the first of the
            # two closest ship objects for each rig [Lars, Anders], 8: IMO number, 9: distance to Lars instrument,
            # 10: distance to Anders instrument, 11: ship type, 12: The coordinates for the closest point [Lars, Anders]
            # 13: the ship objects used used to calculate the closest point [LArs, Anders].
            # 14: the course over ground for each rig [Lars, Anders]
            new_list.append([segments, lars_track[0], [lars_track[1], anders_track[1]], lars_track[2], lars_track[3],
                             lars_track[4], lars_track[5], [lars_track[6], anders_track[6]], lars_track[7],
                             lars_track[8], anders_track[8], lars_track[9], [lars_track[10], anders_track[10]],
                             [lars_track[11], anders_track[11]], [lars_track[12], anders_track[12]]])

    return new_list


def time_closest_point(ship_a, ship_b, closest_point, dist_ship_a, dist_ship_b):
    """
    Calculates the approximate time the ship passes the closest point, based on the time of the ship objects a and b,
    the distance between the first ship of the two and the closest point, and the averaged speed of ship objects a & b.
    It returns the apporximate time for the closest point and the ship object that preceded the closest point (in time)
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
        av_speed = (ship_a.sog + ship_b.sog) / 2                        # Average speed over ground in knots
        if av_speed == 0:
            av_speed = 0.01

        # Find ship-object with the earliest date. Starting date and distance to the instruments will be form this point
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

        # TODO: remove this print statement when I have found the error
        # print('first_point_date = ' + str(first_point.date) + ', first_distance = ' + str(first_distance) +
        #      ', av_speed = ' + str(av_speed))
        # TODO: remove this print statement when I have found the error
        # print('ship_a.sog = '+ str(ship_a.sog) + ', ship_b.sog = ' + str(ship_b.sog))

    return date_closest_point, first_point


def find_closest(distance_list, nr_ships, rig='name'):
    """
    Returns a list with the nr_ships number of tracks in the distance list that are the closest to the specified rig.
    The list will have the same format as the distance_list, so it can be used in the xy_name_ship_tracks2() function.
    :param distance_list: A list of the distance_list format.
    :param nr_ships: The number of closest ships to include
    :param rig: The rig parameter can take the value of 'Lars' or 'Anders', which will decide which rig to measure the
    distance to.
    :return: A list of distance_list format, but only containing the nr_ships items with the tracks closest to the
    instruments
    """

    if rig == 'Lars':
        sort_value = 9
    elif rig == 'Anders':
        sort_value = 10
    else:
        raise ValueError  # The rig name is not correct, it must be Anders or Lars

    sorted_list = sorted(distance_list, key=itemgetter(sort_value))

    new_list = sorted_list[0:nr_ships]

    return new_list



