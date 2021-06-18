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

from rippl.NWP_model_delay.download_NWP_data.ecmwf.ecmwf_download import ECMWFdownload
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_load_file import ECMWFData
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_type import ECMWFType
from rippl.NWP_model_delay.ray_tracing.model_ray_tracing import ModelRayTracing
from rippl.NWP_model_delay.ray_tracing.model_to_delay import ModelToDelay
from rippl.NWP_model_delay.ray_tracing.model_interpolate_delays import ModelInterpolateDelays


class EcmwfAPS(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], split_signal=False,
                 slave='slave', coreg_master='coreg_master', overwrite=False, coreg_crop=True, nwp_model_data='',
                 latlim=[-90, 90], lonlim=[-180, 180]):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'ecmwf_aps'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['ecmwf_aps', 'ecmwf_geometry_corrected_aps']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master',
                                          'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'radar_ray_angles', 'radar_ray_angles',
                                            'radar_ray_angles', 'dem']
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
        self.latlim = latlim
        self.lonlim = lonlim
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
        super(ecmwfAPS, self).__init__(
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
        data_type = 'era5'
        time_interp = 'nearest'
        
        ecmwf_type = ECMWFType(data_type)
        download_folder = os.path.join(self.nwp_model_data, 'ecmwf_data')
        # Download the needed files
        down = ECMWFdownload(self.latlim, self.lonlim, download_folder, data_type=data_type)
        down.prepare_download([overpass])

        # Index the needed files
        data = ECMWFData(ecmwf_type.levels)

        # Load the ecmwf data if available
        if len(down.filenames) > 0:
            data.load_ecmwf(overpass, down.filenames[0])
        else:
            print('No ecmwf data available for ' + date[0].strftime('%Y-%m-%dT%H:%M:%S.%f'))
            return True

        proc_date = overpass.strftime('%Y%m%dT%H%M')

        # Load the geometry
        ray_delays = ModelRayTracing(split_signal=self.split)
        incidence_angles = self['incidence']
        ray_delays.load_geometry(self['azimuth_angle'], 90 - incidence_angles, self['lat'], self['lon'], self['dem'])

        total_time_delays = [0]
        data_str = ['ecmwf_aps']
        data_str_corrected = ['ecmwf_geometry_corrected_aps']

        # And convert the data to delays
        model_delays = ModelToDelay(data.levels)
        model_delays.load_model_delay(data.model_data)
        model_delays.model_to_delay()

        # Convert model delays to delays over specific rays
        ray_delays.load_delay(model_delays.delay_data)
        ray_delays.calc_cross_sections()
        ray_delays.find_point_delays()

        # Finally resample to the full grid
        # The input and output grid should be exactly the same projection to make this work!
        in_coor = self.coordinate_systems['in_coor']
        out_coor = self.coordinate_systems['block_coor']
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
        point_delays = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem, split=self.split)
        point_delays.add_delays(ray_delays.spline_delays)
        point_delays.interpolate_points()

        dem = self['dem']
        point_delays_corrected = ModelInterpolateDelays(in_coor=in_coor, out_coor=out_coor, heights=dem * 0,
                                                        split=self.split)
        point_delays_corrected.add_delays(ray_delays.spline_delays)
        point_delays_corrected.interpolate_points()

        date = list(point_delays.interp_delays['total'].keys())[0]
        self['ecmwf_aps'] = point_delays.interp_delays['total'][date]
        self['ecmwf_geometry_corrected_aps'] = point_delays_corrected.interp_delays['total'][date] * np.cos(np.deg2rad(self['incidence_out']))
