"""
This class contains a full doris data_stack. This links also to several functions which can be solved based on a
full stack approach. This is also the basis from which we create network of interferograms, which can be used for
different purposes.

Examples of functions are:
- network solution for ESD
- network solution for Ionosphere (or combined with ESD)
- sbas or squisar solutions for the whole network
- network solution by including harmonie data or ECMWF data with the interferograms

"""

import os
import datetime
import numpy as np
from shapely.ops import cascaded_union
from shapely import speedups
speedups.disable()

from rippl.meta_data.slc import SLC
from rippl.meta_data.interferogram import Interferogram
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload
from rippl.meta_data.interferogram_network import InterferogramNetwork
from rippl.meta_data.image_processing_concatenate import ImageConcatData, ImageProcessingData
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes



class Stack(object):

    def __init__(self, data_stack_folder='', SAR_type='Sentinel-1', data_stack_name=''):

        if not data_stack_folder:
            settings = UserSettings()
            settings.load_settings()

            if not data_stack_name:
                raise NotADirectoryError('The data_stack name is empty, so directory cannot be created!')
            if SAR_type not in settings.sar_sensor_names:
                raise ValueError('SAR_type should be ' + ' '.join(settings.sar_sensor_names))
            self.data_stack_folder = os.path.join(settings.radar_data_stacks, SAR_type, data_stack_name)
        else:
            self.data_stack_folder = data_stack_folder

        # List of images and interferograms
        self.slcs = dict()
        self.slc_dates = []
        self.ifgs = dict()
        self.ifg_dates = []

        self.dates = []

        # Oversight of the different interferograms and images
        self.ifg_matrix = []
        self.baseline_matrix = []

        # master
        self.master_date = ''
        self.slice_names = []

        # Specific information master slices
        self.master_slice_swath_no = []
        self.master_slice_pol = []
        self.master_slice_lat = []
        self.master_slice_lon = []
        self.master_slice_date = []
        self.master_slice_az_time = []
        self.master_slice_x = []
        self.master_slice_y = []
        self.master_slice_z = []
        self.master_slice_time = []
        self.master_slice_seconds = []
        self.master_slice_range_seconds = []
        self.master_date_boundary = False

        # Finally also give the slice numbers (we will start with 500 so we can count down if needed.)
        self.master_slice_number = []
        self.master_slice_names = []
        self.slice_number = []

    def read_master_slice_list(self):
        # az_time, yyyy-mm-ddThh:mm:ss.ssssss, swath x, slice i, x xxxx, y yyyy, z zzzz, lat ll.ll, lon ll.ll, pol pp

        list_file = os.path.join(self.data_stack_folder, 'master_slice_list')
        if not os.path.exists(list_file):
            print('No existing master slices list found')
            return
        if len(self.master_slice_az_time) != 0:
            print('Master slice list already loaded!')
            return

        with open(list_file, 'r+') as l:
            for line in l:
                sl = line.split(',')
                time = sl[0].split(' ')[1]
                t = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')
                name = 'slice_' + sl[2].split(' ')[1] + '_swath_' + sl[1].split(' ')[1]
                if not name in self.master_slice_names:
                    self.master_slice_az_time.append(time)
                    self.master_slice_time.append(t)
                    self.master_slice_date.append(sl[0].split(' ')[1][:10])
                    self.master_slice_swath_no.append(int(sl[1].split(' ')[1]))
                    self.master_slice_number.append(int(sl[2].split(' ')[1]))
                    self.master_slice_x.append(int(sl[3].split(' ')[1]))
                    self.master_slice_y.append(int(sl[4].split(' ')[1]))
                    self.master_slice_z.append(int(sl[5].split(' ')[1]))
                    self.master_slice_lat.append(float(sl[6].split(' ')[1]))
                    self.master_slice_lon.append(float(sl[7].split(' ')[1]))
                    self.master_slice_seconds.append(float(t.hour * 3600 + t.minute * 60 + t.second) + float(t.microsecond) / 1000000)
                    self.master_slice_names.append(name)
                    self.master_date = t.strftime('%Y%m%d')

            for seconds in self.master_slice_seconds:
                if seconds < 3600 or seconds > 82800:
                    self.master_date_boundary = True

    def read_stack(self, start_date='', end_date='', start_dates='', end_dates='', date='', dates='', time_window=''):
        # This function reads the whole stack in memory. A stack consists of:
        # - images > with individual slices (yyyymmdd)
        # - interferograms > with individual slices if needed. (yyyymmdd_yyyymmdd)
        # First date and last give the maximum and minimum date to load (in case we want to load only a part of the stack.
        # Note: The master date is always loaded!

        # Create a list of search windows with start and end dates
        if isinstance(date, datetime.datetime):
            dates = [date]
        elif isinstance(dates, datetime.datetime):
            dates = [dates]

        if isinstance(dates, list):
            for date in dates:
                if not isinstance(date, datetime.datetime):
                    raise TypeError('Input dates should be datetime objects.')

            if isinstance(time_window, datetime.timedelta):
                self.start_dates = [date - time_window for date in dates]
                self.end_dates = [date + time_window for date in dates]
            else:
                self.start_dates = [date.replace(hour=0, minute=0, second=0, microsecond=0) for date in dates]
                self.end_dates = [date + datetime.timedelta(days=1) for date in self.start_dates]

        elif isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            self.start_dates = [start_date]
            self.end_dates = [end_date]
        elif isinstance(start_dates, list) and isinstance(end_dates, list):
            self.start_dates = start_dates
            self.end_dates = end_dates
            valid_dates = [isinstance(start_date, datetime.datetime) * isinstance(end_date, datetime.datetime) for
                           start_date, end_date in zip(start_dates, end_dates)]
            if np.sum(valid_dates) < len(valid_dates):
                raise TypeError('Input dates should be datetime objects.')
        else:
            raise TypeError('You should define a start or end date or a list of dates to search for Sentinel-1 data! '
                            'Dates should be datetime objects')

        self.start_dates = np.array(self.start_dates)
        self.end_dates = np.array(self.end_dates)

        dirs = next(os.walk(self.data_stack_folder))[1]
        all_image_dirs = sorted([os.path.join(self.data_stack_folder, x) for x in dirs if len(x) == 8 and x != 'coverage'])
        image_dirs = []
        all_ifg_dirs = sorted([os.path.join(self.data_stack_folder, x) for x in dirs if len(x) == 17])
        ifg_dirs = []

        for image_dir in all_image_dirs:
            date = datetime.datetime.strptime(os.path.basename(image_dir), '%Y%m%d')
            if np.sum((self.start_dates < date) * (self.end_dates > date)) > 0:
                image_dirs.append(image_dir)

        for ifg_dir in all_ifg_dirs:
            date_1 = datetime.datetime.strptime(os.path.basename(ifg_dir)[:8], '%Y%m%d')
            date_2 = datetime.datetime.strptime(os.path.basename(ifg_dir)[9:], '%Y%m%d')
            if np.sum((self.start_dates < date_1) * (self.end_dates > date_1)) > 0 and \
                    np.sum((self.start_dates < date_2) * (self.end_dates > date_2)):
                ifg_dirs.append(ifg_dir)

        # Load individual images.
        for image_dir in image_dirs:
            if image_dir not in self.slc_dates:
                image_key = os.path.basename(image_dir)
                self.slcs[image_key] = SLC(image_dir, slice_list=self.master_slice_names, adjust_date=self.master_date_boundary)
                self.slcs[image_key].load_full_meta()
                self.slcs[image_key].load_slice_meta()
                self.slc_dates.append(os.path.basename(image_dir))

        # Load master date information.
        cmaster_image = self.slcs[self.master_date]

        # Load ifgs
        for ifg_dir in ifg_dirs:
            if ifg_dir not in self.ifg_dates:

                if os.path.basename(ifg_dir)[:8] in self.slcs.keys():
                    master_slc = self.slcs[os.path.basename(ifg_dir)[:8]]
                else:
                    master_slc = ''
                if os.path.basename(ifg_dir)[9:] in self.slcs.keys():
                    slave_slc = self.slcs[os.path.basename(ifg_dir)[9:]]
                else:
                    slave_slc = ''

                self.ifgs[ifg_dir[-17:]] = Interferogram(ifg_dir, master_slc=master_slc, slave_slc=slave_slc,
                                                         coreg_slc=cmaster_image, slice_list=self.master_slice_names,
                                                         adjust_date=self.master_date_boundary)
                self.ifg_dates.append(os.path.basename(ifg_dir))

        # combine the ifg and image dates
        self.dates = sorted(set(self.ifg_dates) - set(self.slc_dates))

    def reload_stack(self):
        """
        Reload the metadata for the full stack

        :return:
        """

        for slc_key in self.slcs.keys():
            slc = self.slcs[slc_key]            # type: SLC
            slc.load_full_meta()
            slc.load_slice_meta()

        for ifg_key in self.ifgs.keys():
            ifg = self.ifgs[ifg_key]            # type: Interferogram
            ifg.load_full_meta()
            ifg.load_slice_meta()

    def create_interferogram_network(self, image_baselines=[], network_type='temp_baseline',
                                     temporal_baseline=60, temporal_no=3, spatial_baseline=2000):
        # This method will call the create interferogram network class.
        # Run after reading in the data_stack.

        network = InterferogramNetwork(self.slcs.keys(), self.master_date, image_baselines, network_type,
                                       temporal_baseline, temporal_no, spatial_baseline)
        ifg_pairs = network.ifg_pairs

        # Finally create the requested ifg if they do not exist already
        ifg_ids = self.ifgs.keys()
        cmaster_key = self.master_date
        date_int = np.sort([int(key) for key in self.slcs.keys()])

        for ifg_pair in ifg_pairs:

            master_key = str(date_int[ifg_pair[0]])
            slave_key = str(date_int[ifg_pair[1]])

            ifg_key_1 = master_key + '_' + slave_key
            ifg_key_2 = slave_key + '_' + master_key

            if not ifg_key_1 in ifg_ids and not ifg_key_2 in ifg_ids:
                folder = os.path.join(self.data_stack_folder, ifg_key_1)
                ifg = Interferogram(folder, slave_slc=self.slcs[slave_key], master_slc=self.slcs[master_key], coreg_slc=self.slcs[cmaster_key])
                self.ifgs[ifg_key_1] = ifg

    def stack_data_iterator(self, processes=[], coordinates=[], in_coordinates=[], data_ids=[], polarisations=[], process_types=[],
                            slc_date=False, ifg_date=False, slc=True, ifg=True, full_image=True, slices=False, data=True):

        processes_out = []
        process_ids_out = []
        file_types_out = []
        coordinates_out = []
        in_coordinates_out = []
        slice_names_out = []
        images_out = []
        image_dates_out = []
        image_types_out = []

        if slc_date:
            if not slc_date in self.slcs.keys():
                print('The selected SLC data should be part of the stack')
                return
        if ifg_date:
            if not ifg_date in self.ifgs.keys():
                print('The selected interferogram date should be part of the stack')
                return

        # Get all the full-images or slices for one image/interferogram or all.
        if slc:
            if slc_date:
                slc_dates = [slc_date]
            else:
                slc_dates = self.slcs.keys()

            for date in slc_dates:
                slice_names_slc, processes_slc, process_ids_slc, coordinates_slc, in_coordinates_slc, file_types_slc, images_slc \
                    = self.slcs[date].concat_image_data_iterator(processes, coordinates, in_coordinates, data_ids, polarisations,
                                                                 process_types, full_image, slices, data)
                process_ids_out += process_ids_slc
                file_types_out += file_types_slc
                slice_names_out += slice_names_slc
                images_out += images_slc
                coordinates_out += coordinates_slc
                in_coordinates_out += in_coordinates_slc
                image_dates_out += [date for i in range(len(processes_slc))]
                image_types_out += ['slc' for i in range(len(processes_slc))]

        if ifg:
            if ifg_date:
                ifg_dates = [ifg_date]
            else:
                ifg_dates = self.ifgs.keys()

            for date in ifg_dates:
                slice_names_ifg, processes_ifg, process_ids_ifg, coordinates_ifg, in_coordinates_ifg, file_types_ifg, images_ifg \
                    = self.ifgs[date].concat_image_data_iterator(processes, coordinates, in_coordinates, data_ids, polarisations,
                                                                 process_types, full_image, slices, data)
                process_ids_out += process_ids_ifg
                file_types_out += file_types_ifg
                slice_names_out += slice_names_ifg
                images_out += images_ifg
                coordinates_out += coordinates_ifg
                in_coordinates_out += in_coordinates_ifg
                image_dates_out += [date for i in range(len(processes_ifg))]
                image_types_out += ['ifg' for i in range(len(processes_ifg))]

        return image_types_out, image_dates_out, slice_names_out, processes_out, process_ids_out, coordinates_out, \
               in_coordinates_out, file_types_out, images_out

    def download_SRTM_dem(self, srtm_folder=None, username=None, password=None, buffer=0.5, rounding=0.5, srtm_type='SRTM3', parallel=True, n_processes=4):
        """
        Downloads the needed srtm data for this data_stack. srtm_folder is the folder the downloaded srtm tiles are
        stored.
        Username and password can be obtained at https://lpdaac.usgs.gov
        Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

        Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
        Description srtm q data: https://lpdaac.usgs.gov/node/505

        :param srtm_folder:
        :param username:
        :param password:
        :param buffer:
        :param rounding:
        :param srtm_type:
        :param parallel:
        :return:
        """

        settings = UserSettings()
        settings.load_settings()
        if not srtm_folder:
            srtm_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['srtm'])
        if not username:
            username = settings.NASA_username
        if not password:
            password = settings.NASA_password

        download = SrtmDownload(srtm_folder, username, password, srtm_type, n_processes=n_processes)
        download(self.slcs[self.master_date].data.meta, buffer=buffer, rounding=rounding, parallel=parallel)

    def download_Tandem_X_dem(self, tandem_x_folder=None, username=None, password=None, buffer=0.5, rounding=0.5, lon_resolution=3,
                              parallel=True, n_processes=1):
        """
        Downloads the needed TanDEM-X data for this data_stack. srtm_folder is the folder the downloaded srtm tiles are
        stored. Details on the product and login can be found under: https://geoservice.dlr.de/web/dataguide/tdm90/

        :param TanDEM-X_folder:
        :param username:
        :param password:
        :param buffer:
        :param rounding:
        :param lon_resolution:
        :param parallel:
        :param n_processes:
        :return:
        """

        settings = UserSettings()
        settings.load_settings()
        if not tandem_x_folder:
            tandem_x_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['tdx'])
        if not username:
            username = settings.DLR_username
        if not password:
            password = settings.DLR_password

        download = TandemXDownload(tandem_x_folder, username, password, lon_resolution=lon_resolution, n_processes=n_processes)
        download(self.slcs[self.master_date].data.meta, buffer=buffer, rounding=rounding)

    def create_coverage_shp_kml_geojson(self):
        """
        Create shapefiles for the full coverage and the individual swaths and bursts.

        Returns
        -------

        """

        # First load the bursts of the master date.
        master = self.slcs[self.master_date]            # type: ImageConcatData
        master.load_slice_meta()

        slice_shapes = []
        slice_names = []
        for key in master.slice_data:
            slice = master.slice_data[key]              # type: ImageProcessingData
            slice_shapes.append(slice.readfiles['original'].polygon.buffer(0.0001))
            slice_names.append(key)

        if not os.path.exists(os.path.join(self.data_stack_folder, 'coverage')):
            os.mkdir(os.path.join(self.data_stack_folder, 'coverage'))

        # Create the burst image file
        shapes = ReadWriteShapes()
        shapes.shapes = slice_shapes
        shapes.shape_names = slice_names
        shapes.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'bursts.kml'))
        shapes.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'bursts.geojson'))
        shapes.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'bursts.shp'))

        # Now concatenate the individual swaths and create the swath shp/kml/geojson files
        swath_names = [slice_name[-7:] for slice_name in slice_names]
        swath_unique_names = np.unique(np.array(swath_names))
        swath_shapes = []

        for swath in swath_unique_names:
            swath_ids = np.ravel(np.argwhere(np.array(swath_names) == swath))
            swath_shapes.append(cascaded_union(np.array(slice_shapes)[swath_ids]))
        swaths = ReadWriteShapes()
        swaths.shapes = swath_shapes
        swaths.shape_names = list(swath_unique_names)
        swaths.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'swaths.kml'))
        swaths.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'swaths.geojson'))
        swaths.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'swaths.shp'))

        # Finally do the same thing for the full image
        full_image_shape = cascaded_union(slice_shapes)
        full_image = ReadWriteShapes()
        full_image.shapes = [full_image_shape]
        full_image.shape_names = ['full_image']
        full_image.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'full_image.kml'))
        full_image.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'full_image.geojson'))
        full_image.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'full_image.shp'))
