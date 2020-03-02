import datetime
import numpy


def parse_str_to_int_or_float(string):
    if string == '' :
        return numpy.nan
    else:
        try:
            return float(string)
        except ValueError:
            return int(string)


class ShipAIS:
    def __init__(self):

        self.latitude = numpy.nan              # [Decimal]. Degrees in decimal form.
        self.longitude = numpy.nan             # [Decimal]. Degrees in decimal form.
        self.time_stamp = numpy.nan            # [string][dd mmm yyyy HH:MM:SS.000] UTC. Base station time stamp: The time when the signal is received at the base station.
        self.sog = numpy.nan                   # [Knots]. Speed over ground
        self.cog = numpy.nan                   # [Degrees]. Course over ground
        self.MMSI = numpy.nan                  # [9 digits]. Maritime Mobile Service Identity
        self.name = ''                         # [string]. Ship name.
        self.IMO = numpy.nan                   # [7 digits]. International Maritime Organisation number.
        self.size_a = numpy.nan                # [m]. The length between the transmitter and the bow. 0-511 m, 511 = 511 or greater. A + B = length of ship.
        self.size_b = numpy.nan                # [m]. The length between the transmitter and the stern. 0-511 m, 511 = 511 or greater.
        self.size_c = numpy.nan                # [m]. The length between the transmitter and the starboard side. 0-63 m, 63 = 63 m or grater. C + D = Width of the ship
        self.size_d = numpy.nan                # [m]. The length between the transmitter and the portside. 0-63 m, 63 = 63 m or grater.
        self.static_draught = numpy.nan        # [m]. Maximum present static draught of ship. In 1/10 m, 255 = draught 25.5 m or greater, 0 = not available = default
        self.ship_type_c = numpy.nan           # [1-3 digits, usually 2]. A 1-3-digit number representing a type of ship. 0 = not available or no ship = default, 1-99 = defined
        self.ship_type_t = ''                  # [string]. Text string with name of ship type category.

        self.length = numpy.nan                # [m]
        self.width = numpy.nan                 # [m]
        self.current_draught = numpy.nan       # [m]. Added from Port IT dataset
        self.arr_draught = numpy.nan           # [m]
        self.dep_draught = numpy.nan           # [m]
        self.arr_dep = ''                      # [string] Indicating if the ship is arriving or departing.
        self.date = numpy.nan                  # [dd mm yyyy HH:MM:SS]. Type: datenum
        self.distance = []                     # list of tuples. Each list item contains a tuple with (distance [m], 'Name of point' [str]). Distance from ships position to point[s].

    #def parse_str_to_int_or_float(self, string):
     #   try:
      #      return int(string)
       # except ValueError:
        #    return float(string)

    def read_ship(self, line):
        #if float(line[0]) == '' or float(line[0]) == numpy.nan:
        #    print(str(line[6].lower()), datetime.datetime.strptime(line[2], '%d %b %Y %H:%M:%S.%f %Z'))
        self.latitude = float(line[0])
        self.longitude = float(line[1])
        self.time_stamp = str(line[2])
        self.date = datetime.datetime.strptime(line[2], '%d %b %Y %H:%M:%S.%f %Z')
        self.sog = float(line[3])
        self.cog = float(line[4])
        self.MMSI = int(line[5])
        self.name = str(line[6].lower())

        # These values are sometimes missing, therefore there is a try and except statement.
        try:
            self.IMO = int(line[7])
        except ValueError:
            if line[7] == '':
                self.IMO = numpy.nan
        try:
            self.size_a = float(line[8])
        except ValueError:
            if line[8] == '':
                self.size_a = numpy.nan
        try:
            self.size_b = float(line[9])
        except ValueError:
            if line[9] == '':
                self.size_b = numpy.nan
        try:
            self.size_c = float(line[10])
        except ValueError:
            if line[10] == '':
                self.size_c = numpy.nan
        try:
            self.size_d = float(line[11])
        except ValueError:
            if line[11] == '':
                self.size_d = numpy.nan
        try:
            self.ship_type_c = int(line[13])
        except ValueError:
            if line[13] == '':
                self.ship_type_c = numpy.nan
        try:
            self.ship_type_t = str(line[14].lower())
        except ValueError:
            if line[14] == '':
                self.ship_type_t = numpy.nan

        self.length = self.size_a + self.size_b  # [m]
        self.width = self.size_c + self.size_d

        self.static_draught = parse_str_to_int_or_float(line[12])







