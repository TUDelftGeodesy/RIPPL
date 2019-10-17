# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class NwpAps(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', nwp_type='harmonie', split=False, time_shifts=[0], nwp_data_folder='',
                 lat_lim=[45, 56], lon_lim=[-2, 12],
                 out_coor=[], ray_tracing_coor=[],
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[],
                 in_data_ids=[],
                 slave=[], coreg_master=[]):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str nwp_type: Type of NWP we are calculating. Options are ecmwf_oper, ecmwf_era5 or harmonie
        :param bool split: Do we want to split the data in wet/hydrostatic delay
        :param list[int] time_shifts: The shifts in minutes from the original timing of the data image. This is usefull
                to detect time shifts in the

        :param CoordinateSystem ray_tracing_coor: Coordinate system which is used to do the interpolation.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined in_coor. Options are 'in_coor', 'out_coor' or anything else defined in the
                coordinate system input.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

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
        if split == False:
            split_types = ['']
        else:
            split_types = ['', '_hydrostatic', '_wet']

        self.lat_lim = lat_lim
        self.lon_lim = lon_lim
        self.process_name = 'nwp_aps'
        file_types = []
        data_types = []

        for split_type in split_types:
            for time_shift in time_shifts:
                file_types.append(nwp_type)
                data_types.append('real4')

        if len(in_image_types) == 0:
            in_image_types = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
        if len(in_coor_types) == 0:
            in_coor_types = ['in_coor', 'in_coor']
        if len(in_data_ids) == 0:
            in_data_ids = ['none', 'none']
        if len(in_polarisations) == 0:
            in_polarisations = ['none', 'none']
        if len(in_processes) == 0:
            in_processes = ['geocode', 'geocode', 'resample_dem', 'radar_ray_angles', 'radar_ray_angles']
        if len(in_file_types) == 0:
            in_file_types = ['lat', 'lon', 'dem', 'incidence_angle', 'azimuth_angle']

        # The names for the inputs that will be used during the processing.
        in_type_names = ['lat', 'lon', 'dem', 'incidence_angle', 'azimuth_angle']

        super(NwpAps, self).__init__(
            process_name=self.process_name,
            data_id=data_id,
            file_types=file_types,
            process_dtypes=data_types,
            out_coor=out_coor,
            in_coor_types=in_coor_types,
            in_type_names=in_type_names,
            in_image_types=in_image_types,
            in_processes=in_processes,
            in_file_types=in_file_types,
            in_polarisations=in_polarisations,
            in_data_ids=in_data_ids,
            slave=slave,
            coreg_master=coreg_master)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        readfile = self.out_processing_image.readfiles['original']
        overpass_time = readfile.datetime

        # Start with reading the input data. If the data is missing give a warning quite this processing step.
        if