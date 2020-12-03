# Create and load .
from collections import OrderedDict
import datetime
from shapely.geometry import Polygon
from shapely import speedups
speedups.disable()
import json


class Readfile():

    def __init__(self, json_data='', adjust_date=False):

        self.satellite = ''
        self.wavelength = ''
        self.polarisation = ''
        self.swath = ''

        # Main variables of  (times are all in seconds)
        self.adjust_date = adjust_date
        self.orig_az_first_pix_time = 'None'
        self.orig_ra_first_pix_time = 'None'
        self.ra_first_pix_time = ''
        self.az_first_pix_time = ''
        self.ra_time_step = ''
        self.az_time_step = ''
        self.date = ''

        # Polygon and image size
        self.polygon = ''
        self.poly_coor = ''
        self.size = []
        self.center_heading = []
        self.center_lat = []
        self.center_lon = []
        self.center_line = 0
        self.center_pixel = 0

        # Information for ramping/deramping
        self.FM_ref_az = ''
        self.FM_ref_ra = ''
        self.FM_polynomial = []
        self.DC_ref_az = ''
        self.DC_ref_ra = ''
        self.DC_polynomial = []
        self.steering_rate = []

        # First line/pixel
        self.first_line = 0
        self.first_pixel = 0
        self.orig_first_pixel = 0
        self.orig_first_line = 0

        # Information on source file
        self.first_line_tiff = []
        self.last_line_tiff = []
        self.slice = ''
        self.source_file = ''

        if json_data == '':
            self.json_dict = OrderedDict()
        else:
            self.load_json(json_data=json_data)

    def create_readfile(self, json_dict):
        # Create a  object from original metadata of SAR sensor. Basically this requires an ordered dict of
        # all important data for processing. Check the Sentinel swath metadata function for needed code.

        self.json_dict = json_dict
        self.load_json(json_data=self.json_dict)

    def update_json(self):
        # Save to xml format. Update is only done for the parts that can be changing like the polygon azimuth and range
        # times. (Mainly used with concatenation of image.)

        # Image size
        self.json_dict['Number_of_lines'] = int(self.size[0])
        self.json_dict['Number_of_pixels'] = int(self.size[1])

        # Polygon and coordinates
        self.json_dict['Scene_ul_corner_longitude'] = float(self.poly_coor[0][0])
        self.json_dict['Scene_ul_corner_latitude'] = float(self.poly_coor[0][1])
        self.json_dict['Scene_ur_corner_longitude'] = float(self.poly_coor[1][0])
        self.json_dict['Scene_ur_corner_latitude'] = float(self.poly_coor[1][1])
        self.json_dict['Scene_lr_corner_longitude'] = float(self.poly_coor[2][0])
        self.json_dict['Scene_lr_corner_latitude'] = float(self.poly_coor[2][1])
        self.json_dict['Scene_ll_corner_longitude'] = float(self.poly_coor[3][0])
        self.json_dict['Scene_ll_corner_latitude'] = float(self.poly_coor[3][1])
        self.json_dict['Scene_center_heading'] = float(self.center_heading)
        self.json_dict['Scene_center_latitude'] = float(self.center_lat)
        self.json_dict['Scene_center_longitude'] = float(self.center_lon)
        self.json_dict['Scene_center_pixel'] = int(self.center_pixel)
        self.json_dict['Scene_center_line'] = int(self.center_line)

        # Azimuth and range timing
        self.json_dict['First_pixel_azimuth_time (UTC)'] = self.seconds2time(self.az_first_pix_time, self.date)
        self.json_dict['Range_time_to_first_pixel (2way) (ms)'] = float(self.ra_first_pix_time * 1000)

        # line/pixel
        self.json_dict['First_line'] = self.first_line
        self.json_dict['First_pixel'] = self.first_pixel

        # adjust date
        self.json_dict['Adjust_date'] = self.adjust_date

        return self.json_dict

    def save_json(self, json_path):
        # Save .json file
        self.update_json()

        with open(json_path, 'w+') as file:
            json.dump(self.json_dict, file, indent=3)

    def load_json(self, json_data='', json_path=''):
        # Load from json data source

        if json_path:
            with open(json_path) as file:
                self.json_dict = json.load(file, object_pairs_hook=OrderedDict)
        else:
            self.json_dict = json_data

        self.satellite = self.json_dict['SAR_processor']
        self.wavelength = self.json_dict['Radar_wavelength (m)']
        self.polarisation = self.json_dict['Polarisation']
        self.swath = self.json_dict['Swath']

        # First find the azimuth and range timing
        self.first_line_str = self.json_dict['First_pixel_azimuth_time (UTC)']

        # Adjust date
        if 'Adjust_date' in self.json_dict.keys():
            self.adjust_date = self.json_dict['Adjust_date']
        else:
            self.adjust_date = False

        if self.json_dict['First_pixel_azimuth_time (UTC)'] != 'None':
            self.az_first_pix_time, self.date = self.time2seconds(self.json_dict['First_pixel_azimuth_time (UTC)'], self.adjust_date)
            self.ra_first_pix_time = self.json_dict['Range_time_to_first_pixel (2way) (ms)'] * 1e-3
        if self.json_dict['Orig_first_pixel_azimuth_time (UTC)'] != 'None':
            self.orig_az_first_pix_time, self.date = self.time2seconds(self.json_dict['Orig_first_pixel_azimuth_time (UTC)'], self.adjust_date)
            self.orig_ra_first_pix_time = self.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] * 1e-3

        self.az_time_step = self.json_dict['Azimuth_time_interval (s)']
        self.ra_time_step = 1 / self.json_dict['Range_sampling_rate (computed, MHz)'] / 1000000

        # FM
        if 'FM_reference_azimuth_time' in self.json_dict.keys():
            self.FM_ref_az = self.time2seconds(self.json_dict['FM_reference_azimuth_time'], self.adjust_date)
            self.FM_ref_ra = self.json_dict['FM_reference_range_time']
            self.FM_polynomial = []
            self.FM_polynomial.append(self.json_dict['FM_polynomial_constant_coeff (Hz, early edge)'])
            self.FM_polynomial.append(self.json_dict['FM_polynomial_linear_coeff (Hz/s, early edge)'])
            self.FM_polynomial.append(self.json_dict['FM_polynomial_quadratic_coeff (Hz/s/s, early edge)'])

        # DC
        if 'DC_reference_azimuth_time' in self.json_dict.keys():
            self.DC_ref_az = self.time2seconds(self.json_dict['DC_reference_azimuth_time'], self.adjust_date)
            self.DC_ref_ra = self.json_dict['DC_reference_range_time']
            self.DC_polynomial = []
            self.DC_polynomial.append(self.json_dict['Xtrack_f_DC_constant (Hz, early edge)'])
            self.DC_polynomial.append(self.json_dict['Xtrack_f_DC_linear (Hz/s, early edge)'])
            self.DC_polynomial.append(self.json_dict['Xtrack_f_DC_quadratic (Hz/s/s, early edge)'])
            self.steering_rate = self.json_dict['Azimuth_steering_rate (deg/s)']

        # Image original .tiff
        if 'First_pixel (w.r.t. tiff_image)' in self.json_dict.keys():
            self.first_line_tiff = self.json_dict['First_line (w.r.t. tiff_image)']
            self.first_pixel_tiff = self.json_dict['First_pixel (w.r.t. tiff_image)']
            self.source_file = self.json_dict['Datafile']
        self.slice = self.json_dict['Slice']

        # polygons
        self.poly_coor = [[self.json_dict['Scene_ul_corner_longitude'], self.json_dict['Scene_ul_corner_latitude']],
                          [self.json_dict['Scene_ur_corner_longitude'], self.json_dict['Scene_ur_corner_latitude']],
                          [self.json_dict['Scene_lr_corner_longitude'], self.json_dict['Scene_lr_corner_latitude']],
                          [self.json_dict['Scene_ll_corner_longitude'], self.json_dict['Scene_ll_corner_latitude']]]
        self.polygon = Polygon(self.poly_coor)
        self.size = [self.json_dict['Number_of_lines'], self.json_dict['Number_of_pixels']]
        self.center_heading = self.json_dict['Scene_center_heading']
        self.center_lat = self.json_dict['Scene_center_latitude']
        self.center_lon = self.json_dict['Scene_center_longitude']
        self.center_pixel = self.json_dict['Scene_center_pixel']
        self.center_line = self.json_dict['Scene_center_line']

        # Line numbers
        self.first_line = self.json_dict['First_line']
        self.first_pixel = self.json_dict['First_pixel']
        self.orig_first_pixel = self.json_dict['Orig_first_line']
        self.orig_first_line = self.json_dict['Orig_first_pixel']

    @staticmethod
    def time2seconds(date_string, adjust_date=False):

        if date_string == 0:
            return 0, 'no date'

        time = (datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f') -
                                  datetime.datetime.strptime(date_string[:10], '%Y-%m-%d'))
        seconds = time.seconds + time.microseconds / 1000000.0
        date = date_string[:10]

        if adjust_date and seconds < 43200: # If the date is adjusted and in the first half of the day.
            orig_date = datetime.datetime.strptime(date_string[:10], '%Y-%m-%d')
            date = (orig_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            seconds += 86400

        return seconds, date

    @staticmethod
    def seconds2time(seconds, date):
        datetime_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        time = datetime.timedelta(seconds=seconds)

        date_str = (datetime_date + time).strftime('%Y-%m-%dT%H:%M:%S.%f')
        return date_str
