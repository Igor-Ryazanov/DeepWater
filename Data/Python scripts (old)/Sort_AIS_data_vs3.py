import datetime
from geopy import distance
from csv import writer
import numpy as np
# This code makes it possible to import basemap toolkit. The new version of conda does something that places
# the PROJ_LIB in the "wrong" place, creating a KeyError: 'PROJ_LIB' without this code
import os
os.environ['PROJ_LIB'] = r'\Users\nylunda\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.1.0-hfa6e2cd_1\Library\share'


def sort_ship_list(shiplist, start_date, end_date, ship_types, *args):

    # for ship in AIS_ship_list:
    ship_names = set()
    ships = []
    special_name_list = None
    if args:
        special_name_list = args

    # Make a list of the ship types and/or ship names included in the ship_types list or args/special_name_list
    if special_name_list is not None:
        for ship in shiplist:
            if ship.name in special_name_list:
                ship_names.add(ship.name)

    elif ship_types[0] == 'all':                  # If all types are included
        for ship in shiplist:
            ship_names.add(ship.name)
    else:
        for ship in shiplist:                     # When specific types are specified
            if ship.ship_type_t in ship_types:
                ship_names.add(ship.name)
            else:
                pass

    # Make a list with ships for each ship names in the set and then add each list to the master list "ships"
    for name in ship_names:
        s_list = []
        for ship in shiplist:
            if name == ship.name:
                s_list.append(ship)
            else:
                pass
        ships.append(s_list)

    # For testing if the list is split properly
    """
    for item in ships:
        print('NEW SHIP')
        for ship in item:
            print(ship.name + str(ship.date))
    """

    large_ship_list = []
    for small_ship_list in ships:             # ships contains a list of ships lists, small_ship_list is a list of ships
        new_ship_list = []
        iteration = 0
        for item in small_ship_list:            # item is a ship in the  small_ship_list
            if item == small_ship_list[0]:
                # The first item in the list is always kept
                new_ship_list.append(item)
                iteration += 1
            elif item == small_ship_list[iteration - 1]:
                # There are double items, I'm not sure why, but they are passed
                iteration += 1
            elif item.date - small_ship_list[iteration - 1].date > datetime.timedelta(hours=0.1):
                # TODO: Change the timedelta object to change the time included in the same track.
                # If the time difference between the last item and the current item is > 0.1 hour, a new list is created
                # The limit is set to 6 minutes, as the pilot boats sometimes drive around in the area and thus create
                # several tracks tjhat are treated as the same, which creates strange tracks. 0.1 was ok when testing.
                #print(item.name, str(item.date))
                large_ship_list.append(new_ship_list)
                new_ship_list = []
                new_ship_list.append(item)
                iteration += 1
            else:
                new_ship_list.append(item)
                iteration += 1

        large_ship_list.append(new_ship_list)

    # For testing if the list is split properly
    """
    for item in large_ship_list:
        print('NEW SHIP')
        for ship in item:
            print(ship.name + str(ship.date))
    """

    # Sort so only the passages within the given time limit are included
    final_ship_list = []
    for ship_list in large_ship_list:
        time_ship_list = []                     # Create a new list to which ships within the time limit will be added
        for ship in ship_list:
            if end_date > ship.date > start_date:
                time_ship_list.append(ship)     # Only add ships within the time limit
            else:
                pass
        final_ship_list.append(time_ship_list)  # The final ship list only has tracks from within the time limit

    return final_ship_list


def xy_name_ship_tracks(shiplist, my_map):
    list_xs = []
    list_ys = []
    names_legend = []
    for item in shiplist:
        names_legend.append((item[0].name, str(item[0].date)))

        xs = []
        ys = []
        for s in item:
            xlon, ylat = my_map(s.longitude, s.latitude)
            xs.append(xlon)
            ys.append(ylat)

        list_xs.append(xs)
        list_ys.append(ys)

    return list_xs, list_ys, names_legend


def xy_name_ship_tracks2(distance_list, my_map, rig='name'):
    """
    Uses the information from distance_list to create lists of x-coordinates, y-coordinates and legend information, for
    plotting on the basemap my_map. Each coordinate list contains lists representing different tracks (where each track
    has its own list). The legend has one list per track with information useful for a legend.
    :param distance_list: A list of the output format from the function distance_to_instruments_sphere().
    :param my_map: The basemap on which the tracks shall be plotted, and to which the coordinates must be related.
    :param rig: The rig parameter can take the value of 'Lars' or 'Anders', which will retrieve different distances to
    the track from the distance_list. #TODO: It should only be possible to use these two, but now anything is ok.
    :return: Five lists:
                1) list_xs: A list of lists, where each list contains the x-coordinates for a ship track, compatible
                for plotting on the basemap my_map.
                2) list_ys: A list of lists, where each list contains the y-coordinates for a ship track, compatible
                for plotting on the basemap my_map.
                3) names_legend: A list where each item correlates with a track in the x and y coordinates. The list
                contains the following information: Ship name (str), distance from rig to track [m] (str),
                date (str), speed over ground (kts), ship type
                4) list_point_xs: A list with the x-coordinates for each of the closest points on the plotted ship
                tracks, compatible for plotting on the basemap my_map.
                5) list_point_ys: A list of lists of the y-coordinates for each of the closest points on the plotted
                ship tracks, compatible for plotting on the basemap my_map.
    """

    degree_sign = u'\N{DEGREE SIGN}'

    if rig == 'Lars':
        rig_value = 9
        list_value = 0
        rig_name = 'Lars'
    elif rig == 'Anders':
        rig_value = 10
        list_value = 1
        rig_name = 'Anders'
    else:
        raise ValueError      # The rig name is not correct, it must be Anders or Lars

    list_xs = []
    list_ys = []
    list_dates = []
    list_ships = []
    names_legend = []
    list_point_xs = []
    list_point_ys = []
    a_xs = []
    a_ys = []
    b_xs = []
    b_ys = []
    d_xs = []
    d_ys = []

    # Creating the legend list
    for item in distance_list:
        # Ship name (str), distance from rig to track [m] (str), date (str), speed over ground (kts)
        if item[6] is np.nan:
            draught = 'NaN'
        else:
            draught = str(round(item[6]))

        try:
            speed = str(round(item[4]))
        except ValueError:
            speed = 'NaN'

        # [3] = date string, [1] = distance from instrument, [2] = rig name, [4] = speed, [6] = length,
        # [7] = width, [8] = draught, [9] = cog, [5] = ship type, [0] = ship name/MMSI
        names_legend.append(str(''
                                '{3}    {1:>8} m from {2:<6}    {4:>8} kts     L={6:>3}     W={7:>2}    D={8:>2}     '
                                'cog={9:>3}{10}     {5:>10}  {0:>20}'.format(item[1], str(round(item[rig_value] * 1000)),
                                                                         rig_name,
                                                                         str(item[2][list_value].strftime('%Y-%m-%d  '
                                                                                                          '%H:%M:%S')),
                                                                         str(item[7][list_value]), item[11],
                                                                         speed, str(item[5]), draught,
                                                                         str(round(item[14][list_value])),
                                                                             degree_sign)))

        # item[12] is a list of coordinates for the closest point on the track to each instrument in lat, lon.
        point_xlon, point_ylat = my_map(item[12][list_value][1], item[12][list_value][0])
        list_point_xs.append(point_xlon)
        list_point_ys.append(point_ylat)

        # item[13] is a list of lists, where each list contains the two ship objects used to calculate the closest point
        # The first list is for Lars (given by list_value) and the second for Anders.
        # The [list_value][0] gives the rig (list_value) and the ship object (0 or 1 in the list).
        a_xlon, a_ylat = my_map(item[13][list_value][0].longitude, item[13][list_value][0].latitude)
        b_xlon, b_ylat = my_map(item[13][list_value][1].longitude, item[13][list_value][1].latitude)
        a_xs.append(a_xlon), a_ys.append(a_ylat), b_xs.append(b_xlon), b_ys.append(b_ylat)

        if item[13][list_value][2] is not None:
            d_xlon, d_ylat = my_map(item[13][list_value][2].longitude, item[13][list_value][2].latitude)
        else:
            d_xlon, d_ylat = None, None
        d_xs.append(d_xlon), d_ys.append(d_ylat)

        stop = len(item[0])
        xs = []
        ys = []
        dates = []
        ship_objects = []

        # Creating the x and y coordinate lists
        for value in range(0, stop):
            xlon, ylat = my_map(item[0][value][1], item[0][value][0])
            xs.append(xlon)
            ys.append(ylat)
            dates.append(item[0][value][2])
            ship_objects.append(item[0][value][3])

        list_xs.append(xs)
        list_ys.append(ys)
        list_dates.append(dates)
        list_ships.append(ship_objects)

    return list_xs, list_ys, names_legend, list_point_xs, list_point_ys, a_xs, a_ys, b_xs, b_ys, d_xs, d_ys, list_dates, list_ships


def distance_from_instruments(point_list, shiplist):
    """
    Creates a list of ship objects where the distance to a given point(s) is added to the ship object.
    :param point_list: A list with tuples of the format (lat, lon, 'name of point')
    :param shiplist: A list of ship objects.
    :return: A list of ship objects where information about the distance in meters to the given points have been added
    to the ship object. The format of the ship.distance is a list of  tuples. Each list item contains a tuple with
    (distance [m], 'Name of point' [str]).
    """

    new_ship_list = []

    for ship in shiplist:
        if ship.latitude == '' or ship.latitude == np.nan:
            print(ship.name, ship.date)
        ship_pos = (ship.latitude, ship.longitude)
        distances = []

        for item in point_list:
            point = (item[0], item[1])
            title = item[2]

            dist = distance.distance(ship_pos, point).km

            distances.append((dist*1000, title))
        ship.distance = distances
        # print(ship.distance)
        new_ship_list.append(ship)

    return new_ship_list


def find_close_ships(shiplist, radius):
    """
    Creates a list of ship objects where the objects are within the specified radius of the point(s) of interest.
    The ship objects must contain information in the ship.distance parameter, as the distance value is used to
    see if the distance is within the radius or not.
    :param shiplist: A list of ship objects containing information about latitude, longitude and distance.
    :param radius: The specified radius [m].
    :return: List of ship objects that are withing the specified radius of the point(s) defined in the ship.distance.
    """

    list_close_ships = []

    for item in shiplist:  # Item = list of ship objects

        # This part remove data points that are outside the time period of measurements
        # TODO: This one has no effect, as it just passes the item and then continues with the for-loop. It should be
        #  "break" for it to work.
        if item.date < datetime.datetime(2018, 8, 28, 10, 30):
            pass

        for rigg in item.distance:
            dis = rigg[0]  # s.distance[]
            if dis < radius:
                list_close_ships.append(item)

    return list_close_ships


def save_csv_file_AIS(filename, shiplist, radius):

    with open(filename, 'w', newline='') as outfile:  # The newline='' removes the blank line after every row

        list_ships = []

        csvWriter = writer(outfile)
        csvWriter.writerow(['Date', 'Name', 'IMO', 'MMSI', 'Draught', 'Type', 'Distance to Lars', 'Distance to Anders',
                            'Course over ground (cog) [Degrees]'])

        for item in shiplist:  # Item = list of ship objects

            # This part removed data points that are outside the time period of measurements
            if item.date < datetime.datetime(2018, 8, 28, 10, 30):
                pass

            for rigg in item.distance:
                dis = rigg[0]  # s.distance[]
                if dis < radius:
                    list_ships.append(item)

        for ship in list_ships:
            csvWriter.writerow([ship.date, ship.name, ship.IMO, ship.MMSI, ship.static_draught, ship.ship_type_t,
                                round(ship.distance[0][0], 2), round(ship.distance[1][0], 2), ship.cog])


def save_csv_file_close_ships(filename, shiplist):
    """
    Takes the shiplist and writes a csv-file with information about each ship passage/track.
    :param filename: The name of the csv-file that is created and saved.
    :param shiplist: A list of tuples where each tuple contains information about a specific ship track. The same format
     as the output of find_closest and distance_to_instrument_sphere.
    :return: A saved cvs-file of the same name as the filename, containing the following information (header included):
            0) Distance to Lars instrument in m
            1) The date the ship passes the closest point on the track, string with format [YYYY-MM-DD HH:MM:SS]
            2) Speed over ground in knots
            3) Ship name [str]
            4) Ship type [str]
            5) Static draught of the ship [m]
            6) Length of the ship [m]
            7) Width of the ship.
            8) Course over ground (cog) [degrees]
    """
    with open(filename, 'w', newline='') as outfile:  # The newline='' removes the blank line after every row

        csvWriter = writer(outfile)
        csvWriter.writerow(['Distance to Lars [m]', 'Date [YYYY-MM-DD HH:MM:SS]', 'Speed over ground', 'Ship name',
                            'Ship type', 'Draught', 'Length', 'Width', 'Course over ground'])

        for ship in shiplist:
            # Ship name (str), distance from rig to track [m] (str), date (str), speed over ground (kts)
            if ship[6] is np.NaN:
                draught = 'NaN'
            else:
                draught = str(round(ship[6]))

            # It is made to only write the info for Lars instrumnet (all the cases where the second index is 0).

            csvWriter.writerow([str(round(ship[9] * 1000)), str(ship[2][0].strftime('%Y-%m-%d  %H:%M:%S')),
                                str(round(ship[7][0])), ship[1], ship[11], draught, ship[4], ship[5], ship[14][0]])

