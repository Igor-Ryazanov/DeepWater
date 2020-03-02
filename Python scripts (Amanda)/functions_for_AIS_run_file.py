import datetime
from load_AIS_GBG_harbour import ship_list_2, ship_list


def find_files(start_date, end_date):
    # Create a dictionary for loading the AIS csv files
    AIS_file_dict = {'28-29 aug': "18354Rikhie01_Amanda300318354-0.csv", '29-30 aug': "18354Rikhie01_Amanda300318354-1.csv",
                     '30-31 aug': "18354Rikhie01_Amanda300318354-2.csv", '31-1 aug/sep': "18354Rikhie01_Amanda300318354-3.csv",
                     '1-2 sep': "18354Rikhie01_Amanda300318354-4.csv", '2-3 sep': "18354Rikhie01_Amanda300318354-5.csv",
                     '3-4 sep': "18354Rikhie01_Amanda300318354-6.csv", '4-5 sep': "18354Rikhie01_Amanda300318354-7.csv",
                     '5-6 sep': "18354Rikhie01_Amanda300318354-8.csv", '6-7 sep': "18354Rikhie01_Amanda300318354-9.csv",
                     '7-8 sep': "18354Rikhie01_Amanda300318354-10.csv", '8-9 sep': "18354Rikhie01_Amanda300318354-11.csv",
                     '9-10 sep': "18354Rikhie01_Amanda300318354-12.csv", '10-11 sep': "18354Rikhie01_Amanda300318354-13.csv",
                     '11-12 sep': "18354Rikhie01_Amanda300318354-14.csv", '12-13 sep': "18354Rikhie01_Amanda300318354-15.csv",
                     '13-14 sep': "18354Rikhie01_Amanda300318354-16.csv", '14-15 sep': "18354Rikhie01_Amanda300318354-17.csv",
                     '15-16 sep': "18354Rikhie01_Amanda300318354-18.csv", '16-17 sep': "18354Rikhie01_Amanda300318354-19.csv",
                     '17-18 sep': "18354Rikhie01_Amanda300318354-20.csv", '18-19 sep': "18354Rikhie01_Amanda300318354-21.csv",
                     '19-20 sep': "18354Rikhie01_Amanda300318354-22.csv", '20-21 sep': "18354Rikhie01_Amanda300318354-23.csv",
                     '21-22 sep': "18354Rikhie01_Amanda300318354-24.csv", '22-23 sep': "18354Rikhie01_Amanda300318354-25.csv",
                     '23-24 sep': "18354Rikhie01_Amanda300318354-26.csv", '24-25 sep': "18354Rikhie01_Amanda300318354-27.csv",
                     '25 sep': "18354Rikhie01_Amanda300318354-28.csv"}
    filenames = []

    if datetime.datetime(2018, 8, 28, 9, 31) <= start_date < datetime.datetime(2018, 8, 29, 9, 30):
        filenames.append(AIS_file_dict['28-29 aug'])

    if datetime.datetime(2018, 8, 29, 9, 30) <= start_date < datetime.datetime(2018, 8, 30, 9, 30) or \
            datetime.datetime(2018, 8, 29, 9, 30) <= end_date < datetime.datetime(2018, 8, 30, 9, 30) or \
            start_date < datetime.datetime(2018, 8, 29, 9, 30) and end_date > datetime.datetime(2018, 8, 30, 9, 30):
        filenames.append(AIS_file_dict['29-30 aug'])

    if datetime.datetime(2018, 8, 30, 9, 30) <= start_date < datetime.datetime(2018, 8, 31, 9, 30) or \
            datetime.datetime(2018, 8, 30, 9, 30) <= end_date < datetime.datetime(2018, 8, 31, 9, 30) or \
            start_date < datetime.datetime(2018, 8, 30, 9, 30) and end_date > datetime.datetime(2018, 8, 31, 9, 30):
        filenames.append(AIS_file_dict['30-31 aug'])

    if datetime.datetime(2018, 8, 31, 9, 30) <= start_date < datetime.datetime(2018, 9, 1, 9, 30) or \
            datetime.datetime(2018, 8, 31, 9, 30) <= end_date < datetime.datetime(2018, 9, 1, 9, 30) or \
            start_date < datetime.datetime(2018, 8, 31, 9, 30) and end_date > datetime.datetime(2018, 9, 1, 9, 30):
        filenames.append(AIS_file_dict['31-1 aug/sep'])

    for item in range(1, 24):
        if datetime.datetime(2018, 9, item, 9, 30) <= start_date < datetime.datetime(2018, 9, item + 1, 9, 30) or \
                datetime.datetime(2018, 9, item, 9, 30) <= end_date < datetime.datetime(2018, 9, item + 1, 9, 30) or \
                start_date < datetime.datetime(2018, 9, item, 9, 30) and end_date > datetime.datetime(2018, 9, item + 1, 9, 30):
            filenames.append(AIS_file_dict[str(item) + '-' + str(item + 1) + ' sep'])

    if datetime.datetime(2018, 9, 25, 9, 30) <= start_date < datetime.datetime(2018, 9, 25, 17, 00) or \
            datetime.datetime(2018, 9, 25, 9, 30) <= end_date < datetime.datetime(2018, 9, 25, 17, 30):
        filenames.append(AIS_file_dict['25 sep'])

    return filenames

