"""
This function creates a dummy .res file for a certain overpass time. This is mainly helpfull in case we do not have
the downloaded data, but want to get information on the orbit and ground track geometry.

Needed inputs are:
- first pixel azimuth time
- first pixel range time
- satellite (s1A or s1B)

Possible inputs are:
- no of lines (default 30000)
- no of pixels (default 65000)

"""

import os
import collections
import numpy as np

from rippl.meta_data.image_data import ImageData
from rippl.SAR_sensors.sentinel.sentinel_precise_orbit import SentinelOrbitsDatabase


class DummyRes(object):

    def __init__(self, first_pixel_time, first_line_time, satellite='1A', orbit_folder='', pixels=65000, lines=30000):
        # Setup of dummy file variables:
        # First pixel time should be a datetime object

        self.first_line_time_str = first_line_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
        self.first_line_time = first_line_time
        self.first_pixel_time = first_pixel_time
        
        if satellite in ['1A', '1B']:
            self.satellite = satellite
        
        self.pixels = pixels
        self.lines = lines

        self.orbit_folder = orbit_folder

        self.header = collections.OrderedDict()
        self.readfiles = collections.OrderedDict()
        self.crop = collections.OrderedDict()
        self.orbits = collections.OrderedDict()
        
    def __call__(self):
        # When we call the function perform all steps and create the burst .res files

        self.create_header()
        self.create_orbit()
        self.create_readfiles()
        self.create_crop()

        # In the last step we combine header, readfile.py, orbit and crop to create burst resfiles
        self.burst_meta = []

        dummy = ImageData('', 'single')
        dummy.insert(self.readfiles, 'readfile.py')
        dummy.insert(self.orbit, 'orbits')
        dummy.insert(self.crop, 'crop')
        dummy.header = self.header
        self.burst_meta.append(dummy)

        return dummy

    def create_crop(self):
        # Create information on the crop (this is of course not really done.)

        self.crop['Data_output_file'] = ''
        self.crop['Data_output_format'] = 'complex_int'

        self.crop['Data_lines'] = str(self.lines)
        self.crop['Data_pixels'] = str(self.pixels)
        self.crop['Data_first_pixel'] = str(1)
        self.crop['Data_first_line'] = str(1)
        self.crop['Data_first_line (w.r.t. tiff_image)'] = str(1)
        self.crop['Data_last_line (w.r.t. tiff_image)'] = str(self.lines)

    def create_readfiles(self):
        # Create the dummy readfile.py information

        self.readfiles['SAR_PROCESSOR'] = 'Sentinel-' + self.satellite[1:]

        self.readfiles['rangePixelSpacing'] = '2.329562e+00'
        self.readfiles['azimuthPixelSpacing'] = '1.392254e+01'
        self.readfiles['RADAR_FREQUENCY (HZ)'] = '5.405000454334350e+09'
        self.readfiles['Radar_wavelength (m)'] = '0.055465760'

        self.readfiles['First_pixel_azimuth_time (UTC)'] = self.first_line_time_str
        self.readfiles['Range_time_to_first_pixel (2way) (ms)'] = self.first_pixel_time
        self.readfiles['Pulse_Repetition_Frequency (computed, Hz)'] = '4.864863102995529e+02'
        self.readfiles['Range_sampling_rate (computed, MHz)'] = '64.345238126'

        self.readfiles['Number_of_lines_original'] = str(self.lines)
        self.readfiles['Number_of_pixels_original'] = str(self.pixels)
        self.readfiles['First_pixel (w.r.t. tiff_image)'] = '1'
        self.readfiles['Last_pixel (w.r.t. tiff_image)'] = str(self.pixels)
        self.readfiles['First_line (w.r.t. tiff_image)'] = '1'
        self.readfiles['Last_line (w.r.t. tiff_image)'] = str(self.lines)

    def create_header(self):

        self.header = collections.OrderedDict()

        self.header['row_1'] = ['===============================================\n']
        self.header['MASTER RESULTFILE:'] = ''
        self.header['Created by'] = 'Doris TU Delft'
        self.header['row_2'] = 'PyDoris (Delft o-o Radar Interferometric Software)'
        self.header['Version'] = 'PyDoris v1.0.0'
        self.header['row_3'] = ['===============================================\n']

    def create_orbit(self):
        # Read the orbit database
        precise = os.path.join(self.orbit_folder, 'precise')
        restituted = os.path.join(self.orbit_folder, 'restituted')
        if not os.path.exists(precise) or not os.path.exists(restituted):
            print('The orbit folder and or precise/restituted sub-folder do not exist.')
        else:
            self.orbits = SentinelOrbitsDatabase(precise, restituted)

        orbit_dat = self.orbits.interpolate_orbit(self.first_line_time, 'POE', satellite=self.satellite)

        # If there are no precise orbit files available switch back to .xml file information
        if orbit_dat == False:
            print('No corresponding orbit found.')
            return dict()

        self.orbit = collections.OrderedDict()

        self.orbit['row_0'] = ['t(s)', 'X(m)', 'Y(m)', 'Z(m)', 'velX(m)', 'velY(m)', 'velZ(m)']
        t = orbit_dat['orbitTime']
        x = orbit_dat['orbitX']
        y = orbit_dat['orbitY']
        z = orbit_dat['orbitZ']
        vel_x = orbit_dat['velX']
        vel_y = orbit_dat['velY']
        vel_z = orbit_dat['velZ']
        self.orbit['NUMBER_OF_DATAPOINTS'] = str(len(t))

        # Save the rows
        for n in np.arange(len(t)):
            self.orbit['row_' + str(n + 1)] = ["{:.6f}".format(t[n]),
                                                     "{:.7f}".format(float(x[n])),
                                                     "{:.7f}".format(float(y[n])),
                                                     "{:.7f}".format(float(z[n])),
                                                     "{:.7f}".format(float(vel_x[n])),
                                                     "{:.7f}".format(float(vel_y[n])),
                                                     "{:.7f}".format(float(vel_z[n]))]
