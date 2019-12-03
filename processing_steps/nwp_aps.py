# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import os
import datetime

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.readfile import Readfile

from rippl.NWP_simulations.harmonie.harmonie_database import HarmonieDatabase
from rippl.NWP_simulations.harmonie.harmonie_load_file import HarmonieData
from rippl.NWP_simulations.ECMWF.ecmwf_type import ECMWFType
from rippl.NWP_simulations.ECMWF.ecmwf_download import ECMWFdownload
from rippl.NWP_simulations.ECMWF.ecmwf_load_file import ECMWFData
from rippl.NWP_simulations.model_ray_tracing import ModelRayTracing
from rippl.NWP_simulations.model_to_delay import ModelToDelay
from rippl.NWP_simulations.model_interpolate_delays import ModelInterpolateDelays
from rippl.NWP_simulations.radar_data import RadarData


class NwpApsGrid(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', nwp_type='harmonie', split=False, time_shifts=[0], nwp_data_folder='',
                 lat_lim=[45, 56], lon_lim=[-2, 12],
                 out_coor=[], ray_tracing_coor=[],
                 slave='slave', coreg_master='coreg_master', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str nwp_type: Type of NWP we are calculating. Options are ecmwf_oper, ecmwf_era5 or harmonie
        :param bool split: Do we want to split the data in wet/hydrostatic delay
        :param list[int] time_shifts: The shifts in minutes from the original timing of the data image. This is usefull
                to detect time shifts in the

        :param CoordinateSystem ray_tracing_coor: Coordinate system which is used to do the interpolation.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        if nwp_type not in ['emcwf_oper', 'ecmwf_era5', 'harmonie']:
            raise TypeError('Only the NWP types emcwf_oper, ecmwf_era5 and harmonie are supported.')
            if nwp_type == 'ecmwf_oper':
                self.ecmwf_type = 'oper'
            elif nwp_type == 'ecmwf_era5':
                self.ecmwf_type = 'era5'
        if split == False:
            split_types = ['']
        else:
            split_types = ['', '_hydrostatic', '_wet']

        self.lat_lim = lat_lim
        self.lon_lim = lon_lim
        self.process_name = 'nwp_aps'

        # Start with reading the input data. If the data is missing give a warning quite this processing step.
        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'radar_ray_angles'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = []
        self.output_info['data_types'] = []

        for split_type in split_types:
            for time_shift in time_shifts:
                self.output_info['file_types'].append(nwp_type + '_' + str(time_shift))
                self.output_info['data_types'].append('real4')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'dem', 'dem', 'radar_ray_angles', 'radar_ray_angles', 'reproject', 'reproject']
        self.input_info['file_types'] = ['lat', 'lon', 'dem', 'dem', 'incidence_angle', 'azimuth_angle', 'in_coor_lines', 'in_coor_pixels']
        self.input_info['data_types'] = ['real4', 'real4', 'real4', 'real4', 'real4', 'real4', 'real4', 'real4']
        self.input_info['polarisations'] = ['', '', '', '', '', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id, data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor', 'out_coor', 'in_coor', 'in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '', '', '', 'in_coor', 'in_coor']
        self.input_info['type_names'] = ['lat', 'lon', 'dem', 'out_dem', 'incidence_angle', 'azimuth_angle', 'lines', 'pixels']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.nwp_data_folder = nwp_data_folder
        self.lat_lim = lat_lim
        self.lon_lim = lon_lim
        self.split = split
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.settings = dict()
        self.nwp_type = nwp_type

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(NwpApsGrid, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Here we do a ray tracing through a weather model.

        :return:
        """

        if self.nwp_type == 'ecmwf':
            data, date = self.load_ecmwf()
        elif self.nwp_type == 'harmonie':
            data, date = self.load_harmonie()

        proc_date = date[0].strftime('%Y%m%dT%H%M')
        self.ray_tracing(data, proc_date)

    def load_harmonie(self):
        """
        Load Harmonie data

        """

        readfile = self.processing_images['slave'].readfiles['original']    # type: Readfile
        overpass = readfile.datetime
        harmonie_archive = HarmonieDatabase(database_folder=self.nwp_data_folder)
        filename, date = harmonie_archive(overpass)

        # Load the Harmonie data if available
        if filename[0]:
            data = HarmonieData()
            data.load_harmonie(date[0], filename[0])
        else:
            print('No harmonie data available for ' + date[0].strftime('%Y-%m-%dT%H:%M:%S.%f'))
            return

        return data, date

    def load_ecmwf(self):
        """
        Load ECMWF data

        """

        readfile = self.processing_images['slave'].readfiles['original']    # type: Readfile
        overpass = readfile.datetime
        time_interp = 'nearest'
        ecmwf_type = ECMWFType(self.ecmwf_type)
        radar_data = RadarData(time_interp, ecmwf_type.t_step, 0)
        radar_data.match_overpass_weather_model(overpass)

        # Load the ECMWF data
        # Download the needed files
        # For simplicity we limit ourselves here to europe if the lat/lon limits are not given.
        if len(self.latlim) != 2:
            self.latlim = [45, 56]
        if len(self.lonlim) != 2:
            self.lonlim = [-2, 12]

        down = ECMWFdownload(self.latlim, self.lonlim, self.nwp_data_folder, data_type=self.ecmwf_type)
        down.prepare_download(radar_data.date_times)

        # Check if file is already available.
        if not os.path.exists(down.filenames[0] + '_atmosphere.grb') or not os.path.exists(
                down.filenames[0] + '_surface.grb'):
            print('ECMWF file ' + down.filenames[0] + '_atmosphere.grb not found. Skipping because download is switched of')
            return

        data = ECMWFData(ecmwf_type.levels)
        data.load_ecmwf(down.dates[0], down.filenames[0])

        proc_date = down.dates[0].strftime('%Y%m%dT%H%M')

        # Load the data files
        data = ECMWFData(ecmwf_type.levels)
        data.load_ecmwf(down.dates[0], down.filenames[0])

        return data, proc_date


    def ray_tracing(self, data, proc_date):
        # The ray tracing methods which are independent of the input source.

        # Load the geometry
        ray_delays = ModelRayTracing(split_signal=self.split)
        geometry_lines = self.coordinate_systems['in_coor'].interval_lines
        geometry_pixels = self.coordinate_systems['in_coor'].interval_pixels

        ray_delays.load_geometry(geometry_lines, geometry_pixels,
                                 self['azimuth_angle'], 90 - self['incidence_angle'],
                                 self['lat'], self['lon'], self['dem'])

        # And convert the data to delays
        geoid_file = os.path.join(self.nwp_data_folder, 'egm96.raw')

        model_delays = ModelToDelay(data.levels, geoid_file)
        model_delays.load_model_delay(data.model_data)
        model_delays.model_to_delay()

        # Convert model delays to delays over specific rays
        ray_delays.load_delay(model_delays.delay_data)
        ray_delays.calc_cross_sections()
        ray_delays.find_point_delays()
        model_delays.remove_delay(proc_date)

        # Finally resample to the full grid
        point_delays = ModelInterpolateDelays(geometry_lines, geometry_pixels, split=self.split)
        point_delays.add_interp_points(np.ravel(self['lines']), np.ravel(self['pixels']),
                                       np.ravel(self['out_dem']))

        point_delays.add_delays(ray_delays.spline_delays)
        point_delays.interpolate_points()

        # Save the file
        self['nwp_aps'] = point_delays.interp_delays['total'][proc_date].astype(np.float32)
        if self.split:
            self['nwp_aps_hydrostatic'] = point_delays.interp_delays['hydrostatic'][proc_date].astype(np.float32)
            self['nwp_aps_wet'] = point_delays.interp_delays['wet'][proc_date].astype(np.float32)
        ray_delays.remove_delay(proc_date)
