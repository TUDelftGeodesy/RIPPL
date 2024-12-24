# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import datetime
import copy
import os
import logging

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

# Load harmonie data
from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_database import HarmonieDatabase
from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_load_file import HarmonieData

# Load ECMWF data
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_download import CDSdownload
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_load_file import CDSData

# Ray tracing and NWP model interpolation
from rippl.NWP_model_delay.ray_tracing_NWP_data.model_ray_tracing import ModelRayTracing
from rippl.NWP_model_delay.ray_tracing_NWP_data.model_to_delay import ModelToDelay
from rippl.NWP_model_delay.ray_tracing_NWP_data.model_interpolate_delays import ModelInterpolateDelays
from rippl.NWP_model_delay.timing_NWP_data.change_timing import AdjustTiming

class NWPDelay(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], time_delays=[], time_step=5,
                 time_correction=True, geometry_correction=True, split_signal=False,
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False,
                 nwp_model_database_folder='', model_name='era5', model_level_type='pressure_levels',
                 latlim=[-90, 90], lonlim=[-180, 180], spline_type='cubic'):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        # Define the model type and model level type
        model_names = ['era5', 'cerra', 'harmonie']
        if not model_name in model_names:
            raise TypeError('Model name must be one of the following: ' + model_names)

        level_types = ['pressure_levels', 'model_levels']
        if not model_level_type in level_types:
            raise TypeError('Level type must be one of the following: ' + level_types)

        if model_level_type == 'pressure_levels' and model_name == 'harmonie':
            raise TypeError('harmonie model can only be calculated using model levels!')

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = model_name + '_nwp_delay'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = []
        self.output_info['data_types'] = []

        # Define all the outputs
        geometry_types = ['', '_geometry_corrected'] if geometry_correction else ['']
        signal_types = ['_aps', '_hydrostatic_delay', '_wet_delay', '_liquid_delay'] if split_signal else ['_aps']
        for geom in geometry_types:
            for sig in signal_types:
                self.output_info['file_names'].append(model_name + sig + geom)
                self.output_info['data_types'].append('real4')
                for time_delay in time_delays:
                    self.output_info['file_names'].append(model_name + sig + geom + '_' + str(time_delay) + '_minutes')
                    self.output_info['data_types'].append('real4')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc', 'reference_slc',
                                          'reference_slc', 'reference_slc']
        self.input_info['process_names'] = ['geocode', 'geocode', 'radar_geometry', 'radar_geometry', 'radar_geometry',
                                            'dem']
        self.input_info['file_names'] = ['lat', 'lon', 'incidence_angle', 'azimuth_angle', 'incidence_angle', 'dem']
        self.input_info['polarisations'] = ['', '', '', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor', 'in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '', '', '']
        self.input_info['aliases_processing'] = ['lat', 'lon', 'incidence', 'azimuth_angle', 'incidence_out', 'dem']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['model_name'] = model_name
        self.settings['model_level_type'] = model_level_type
        self.settings['latlim'] = latlim
        self.settings['lonlim'] = lonlim

        self.settings['split_signal'] = split_signal
        self.settings['signal_types'] = signal_types
        self.settings['geometry_correction'] = geometry_correction
        self.settings['geometry_types'] = geometry_types
        self.settings['spline_type'] = spline_type

        self.settings['time_correction'] = time_correction
        self.settings['time_delays'] = time_delays
        self.settings['time_step'] = time_step
        # Check if steps are a multiple of the time step
        if len(time_delays) > 0:
            if not np.max(np.remainder(np.array(time_delays), time_step)) == 0:
                raise ValueError('All selected times should be a multiple of the time step!')

        if nwp_model_database_folder:
            self.settings['nwp_model_database'] = nwp_model_database_folder
        else:
            settings = UserSettings()
            settings.load_settings()
            self.settings['nwp_model_database'] = settings.settings['paths']['NWP_model_database']

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the secondary_slc and reference_slc image.

        :return:
        """

        readfile = self.processing_images['secondary_slc'].readfiles['original']
        datetime_date = datetime.datetime.strptime(readfile.date, '%Y-%m-%d')
        time = datetime.timedelta(seconds=readfile.az_first_pix_time)
        overpass = datetime_date + time

        if self.settings['model_name'] == 'harmonie':
            data = self.load_harmonie_data(overpass)
        elif self.settings['model_name'] in ['era5', 'cerra']:
            data = self.load_ecmwf_data(overpass)
        model_time = datetime.datetime.strptime(list(data.model_data.keys())[0], '%Y%m%dT%H%M')
        total_time_delays = self.get_time_delays(overpass, model_time)

        # Load the geometry
        ray_delays = ModelRayTracing(split_signal=self.settings['split_signal'])
        incidence_angles = self['incidence']
        ray_delays.load_geometry(self['azimuth_angle'], 90 - incidence_angles, self['lat'], self['lon'], self['dem'])

        # Correct delay time for needed time steps
        adjust_time = AdjustTiming(time_step=self.settings['time_step'])
        adjust_time.load_model_delay(copy.deepcopy(data.model_data))
        adjust_time.change_timing(time_diff=total_time_delays)

        # Do the ray tracing for the different moments in time.
        for t, time_delay in zip(total_time_delays, [0] + self.settings['time_delays']):

            new_date = (model_time + datetime.timedelta(minutes=int(t))).strftime('%Y%m%dT%H%M')
            # And convert the data to delays
            model_delays = ModelToDelay()

            adjusted_data = dict()
            adjusted_data[new_date] = adjust_time.model_data[new_date]
            model_delays.load_model_delay(adjusted_data)
            model_delays.model_to_delay()

            # Convert model delays to delays over specific rays
            ray_delays.load_delay(model_delays.delay_data)
            ray_delays.calc_cross_sections()
            ray_delays.find_point_delays(spline_type=self.settings['spline_type'])
            model_delays.remove_delay(new_date)

            # Finally resample to the full grid
            # The input and output grid should be exactly the same projection to make this work!
            in_coor = self.coordinate_systems['in_coor']
            out_coor = self.coordinate_systems['out_coor_chunk']
            if in_coor.grid_type != out_coor.grid_type:
                raise TypeError('Input and output coordinates should be the same')
            elif in_coor.grid_type == 'geographic':
                if in_coor.geo != out_coor.geo:
                    raise ('Used geographic system should be the same. Aborting')
            elif in_coor.grid_type == 'projection':
                if in_coor.proj4_str != out_coor.proj4_str:
                    raise ('Used projection should be the same. Aborting')

            # Then the resampling itself
            dem = self['dem']

            # Start with the original slanted data
            point_delays = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem,
                                                  split=self.settings['split_signal'])
            point_delays.add_delays(ray_delays.spline_delays)
            point_delays.interpolate_points()

            # Save the different signal types
            for signal_type, signal_type_func in zip(self.settings['signal_types'], point_delays.run_data):
                data_str = self.settings['model_name'] + signal_type
                if t is not total_time_delays[0]:
                    data_str = data_str + '_' + str(time_delay) + '_minutes'
                self[data_str] = point_delays.interp_delays[signal_type_func][new_date].astype(np.float32)

            # Then process the geometry corrected data
            if self.settings['geometry_correction']:
                point_delays_corrected = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem * 0,
                                                                split=self.settings['split_signal'])
                point_delays_corrected.add_delays(ray_delays.spline_delays)
                point_delays_corrected.interpolate_points()

                for signal_type, signal_type_func in zip(self.settings['signal_types'], point_delays.run_data):
                    data_str_corrected = self.settings['model_name'] + signal_type + '_geometry_corrected'
                    if t is not total_time_delays[0]:
                        data_str_corrected = data_str + '_' + str(time_delay) + '_minutes'

                    self[data_str_corrected] = (point_delays_corrected.interp_delays[signal_type_func][new_date] *
                                                np.cos(np.deg2rad(self['incidence_out']))).astype(np.float32)

            # Finally remove data and proceed to new time delay.
            ray_delays.remove_delay(new_date)
            logging.info('Created delay image for ' + new_date)

    def get_time_delays(self, overpass, model_time):

        model_name = self.settings['model_name']
        total_time_delays = [0]
        if self.settings['time_correction']:
            correct_time = int(np.round((overpass - model_time).total_seconds() / (60 * self.settings['time_step'])))
            total_time_delays[0] = total_time_delays[0] + correct_time

        for t in self.settings['time_delays']:
            if self.settings['time_correction']:
                t = t + correct_time
            total_time_delays.append(t)

        return total_time_delays

    def load_harmonie_data(self, overpass):
        """ Load the Harmonie data if needed """
        harmonie_archive = HarmonieDatabase()
        filename, fc_time = harmonie_archive(overpass)

        # Load the Harmonie data if available
        if filename:
            data = HarmonieData()
            data.load_harmonie(fc_time, filename)
        else:
            logging.info('No harmonie_nl data available for ' + overpass.strftime('%Y-%m-%dT%H:%M:%S.%f'))
            return True

        return data

    def load_ecmwf_data(self, overpass):
        """ Load the ECMWF data if needed """

        model_name = self.settings['model_name']
        model_level_type = self.settings['model_level_type']
        if model_name == 'era5':
            if model_level_type == 'model_levels':
                data_type = 'reanalysis-era5-complete'
            else:
                data_type = 'reanalysis-era5-pressure-levels'
        elif model_name == 'cerra':
            if model_level_type == 'model_levels':
                data_type = 'reanalysis-cerra-model-levels'
            else:
                data_type = 'reanalysis-cerra-pressure-levels'

        # Check which file is needed for download.
        down = CDSdownload([], latlim=self.settings['latlim'], lonlim=self.settings['lonlim'],
                           data_type=data_type)
        filename = down.check_overpass_available(overpass)

        # Load the ecmwf data if available
        if filename[0]:
            data = CDSData()
            data.load_ecmwf(overpass, filename[0], filename[1])
        else:
            logging.info('No ecmwf data available for ' + overpass.strftime('%Y-%m-%dT%H:%M:%S.%f'))
            return True

        return data
