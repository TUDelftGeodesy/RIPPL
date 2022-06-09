# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import datetime
import copy
import os

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

from rippl.NWP_model_delay.load_NWP_data.harmonie.harmonie_database import HarmonieDatabase
from rippl.NWP_model_delay.load_NWP_data.harmonie.harmonie_load_file import HarmonieData
from rippl.NWP_model_delay.ray_tracing.model_ray_tracing import ModelRayTracing
from rippl.NWP_model_delay.ray_tracing.model_to_delay import ModelToDelay
from rippl.NWP_model_delay.ray_tracing.model_interpolate_delays import ModelInterpolateDelays
from rippl.NWP_model_delay.advection_timing_NWP_data.change_timing import AdjustTiming


class HarmonieAPS(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], correct_time_delay=True, time_delays=[], split_signal=False,
                 slave='slave', coreg_master='coreg_master', overwrite=False, coreg_crop=True, nwp_model_data=''):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'harmonie_aps'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['harmonie_aps', 'harmonie_geometry_corrected_aps']
        self.output_info['data_types'] = ['real4', 'real4']

        self.time_delays = time_delays
        self.correct_time_delay = correct_time_delay
        if correct_time_delay:
            self.output_info['file_types'].append('harmonie_time_corrected_aps')
            self.output_info['file_types'].append('harmonie_time_geometry_corrected_aps')
            self.output_info['data_types'].extend(['real4', 'real4'])
            for time_delay in time_delays:
                self.output_info['file_types'].append('harmonie_time_corrected_aps_' + str(time_delay) + '_minutes')
                self.output_info['file_types'].append('harmonie_time_geometry_corrected_aps_' + str(time_delay) + '_minutes')
                self.output_info['data_types'].extend(['real4', 'real4'])
        else:
            for time_delay in time_delays:
                self.output_info['file_types'].append('harmonie_aps_' + str(time_delay) + '_minutes')
                self.output_info['file_types'].append('harmonie_geometry_corrected_aps_' + str(time_delay) + '_minutes')
                self.output_info['data_types'].extend(['real4', 'real4'])

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'radar_ray_angles', 'radar_ray_angles', 'radar_ray_angles', 'dem']
        self.input_info['file_types'] = ['lat', 'lon', 'incidence_angle', 'azimuth_angle', 'incidence_angle', 'dem']
        self.input_info['polarisations'] = ['', '', '', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor', 'in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '', '', '']
        self.input_info['type_names'] = ['lat', 'lon', 'incidence', 'azimuth_angle', 'incidence_out', 'dem']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.split = split_signal
        if nwp_model_data:
            self.nwp_model_data = nwp_model_data
        else:
            settings = UserSettings()
            settings.load_settings()
            self.nwp_model_data = settings.NWP_model_database

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(HarmonieAPS, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the slave and coreg_master image.

        :return:
        """

        readfile = self.processing_images['slave'].readfiles['original']
        datetime_date = datetime.datetime.strptime(readfile.date, '%Y-%m-%d')
        time = datetime.timedelta(seconds=readfile.az_first_pix_time)
        overpass = datetime_date + time

        harmonie_archive = HarmonieDatabase(database_folder=os.path.join(self.nwp_model_data, 'harmonie_data'))
        filename, date = harmonie_archive(overpass)

        # Load the Harmonie data if available
        if filename[0]:
            data = HarmonieData()
            data.load_harmonie(date[0], filename[0])
        else:
            print('No harmonie data available for ' + date[0].strftime('%Y-%m-%dT%H:%M:%S.%f'))
            return True

        proc_date = date[0].strftime('%Y%m%dT%H%M')

        # Load the geometry
        ray_delays = ModelRayTracing(split_signal=self.split)
        incidence_angles = self['incidence']
        ray_delays.load_geometry(self['azimuth_angle'], 90 - incidence_angles, self['lat'], self['lon'], self['dem'])

        total_time_delays = [0]
        data_strs = ['harmonie_aps']
        data_strs_corrected = ['harmonie_geometry_corrected_aps']
        if self.correct_time_delay:
            correct_time = int(np.round((overpass - date[0]).total_seconds() / 60))
            total_time_delays.append(correct_time)
            data_strs_corrected.append('harmonie_time_geometry_corrected_aps')
            data_strs.append('harmonie_time_corrected_aps')

        for t in self.time_delays:
            if self.correct_time_delay:
                total_time_delays.append(t + correct_time)
                data_strs.append('harmonie_time_corrected_aps_' + str(t) + '_minutes')
                data_strs_corrected.append('harmonie_time_geometry_corrected_aps_' + str(t) + '_minutes')
            else:
                total_time_delays.append(t)
                data_strs.append('harmonie_aps_' + str(t) + '_minutes')
                data_strs_corrected.append('harmonie_geometry_corrected_aps_' + str(t) + '_minutes')

        for t, data_str, data_str_corrected in zip(total_time_delays, data_strs, data_strs_corrected):

            new_date = (date[0] + datetime.timedelta(minutes=int(t))).strftime('%Y%m%dT%H%M')
            # Correct delay time
            adjust_time = AdjustTiming(time_step=5)
            adjust_time.load_model_delay(copy.deepcopy(data.model_data))
            adjust_time.change_timing(time_diff=[int(t)])

            # And convert the data to delays
            model_delays = ModelToDelay(data.levels)

            adjusted_data = dict()
            adjusted_data[new_date] = adjust_time.model_data[new_date]
            adjusted_data['latitudes'] = adjust_time.model_data['latitudes']
            adjusted_data['longitudes'] = adjust_time.model_data['longitudes']
            adjusted_data['geo_h'] = adjust_time.model_data['geo_h']
            model_delays.load_model_delay(adjusted_data)
            model_delays.model_to_delay()

            # Convert model delays to delays over specific rays
            ray_delays.load_delay(model_delays.delay_data)
            ray_delays.calc_cross_sections()
            ray_delays.find_point_delays()
            model_delays.remove_delay(new_date)

            # Finally resample to the full grid
            # The input and output grid should be exactly the same projection to make this work!
            in_coor = self.coordinate_systems['in_coor']
            out_coor = self.coordinate_systems['block_coor']
            if in_coor.grid_type != out_coor.grid_type:
                raise TypeError('Input and output coordinates should be the same')
            elif in_coor.grid_type == 'geographic':
                if in_coor.geo != out_coor.geo:
                    raise('Used geographic system should be the same. Aborting')
            elif in_coor.grid_type == 'projection':
                if in_coor.proj4_str != out_coor.proj4_str:
                    raise('Used projection should be the same. Aborting')

            # Then the resampling itself
            dem = self['dem']
            point_delays = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem, split=self.split)
            point_delays.add_delays(ray_delays.spline_delays)
            point_delays.interpolate_points()

            dem = self['dem']
            point_delays_corrected = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem * 0, split=self.split)
            point_delays_corrected.add_delays(ray_delays.spline_delays)
            point_delays_corrected.interpolate_points()

            self[data_str] = point_delays.interp_delays['total'][new_date]
            self[data_str_corrected] = point_delays_corrected.interp_delays['total'][new_date] * np.cos(np.deg2rad(self['incidence_out']))
            ray_delays.remove_delay(new_date)

            print('Created delay image for ' + new_date)

        # Test results
        """
        import matplotlib.pyplot as plt
        
        for key in data_strs_corrected:
            plt.figure()
            plt.imshow(self[key])
            plt.show()
        
        """
