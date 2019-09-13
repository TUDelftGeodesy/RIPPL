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
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class Geocode(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], dem_type='',
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_data_ids=[],
                 coreg_master=[]):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined coor_in. Options are 'coor_in', 'coor_out' or anything else defined in the
                coordinate system input.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'geocode'
        file_types = ['X', 'Y', 'Z', 'lat', 'lon']
        data_types = ['real8', 'real8', 'real8', 'real4', 'real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        if len(in_image_types) == 0:
            in_image_types = ['coreg_master']  # In this case only the slave is used, so the input variable master,
            # coreg_master, ifg and processing_images are not needed.
            # However, if you override the default values this could change.
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in']  # Same here but then for the coor_out and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = [dem_type]
        in_polarisations = ['none']
        if len(in_processes) == 0:
            in_processes = ['radar_dem']
        if len(in_file_types) == 0:
            in_file_types = ['radar_dem']

        in_type_names = ['dem']

        # Initialize process
        super(Geocode, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       in_type_names=in_type_names,
                       in_coor_types=in_coor_types,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       coreg_master=coreg_master)

    def process_calculations(self):
        """
        With geocoding we find the XYZ values using the DEM height and line/pixel spacing. From the XYZ values the
        latitude and longitude can directly be derived too.

        :return:
        """

        # Get orbit
        orbit = self.in_processing_images['coreg_master'].find_best_orbit('original')
        orbit_interp = OrbitCoordinates(coordinates=self.block_coor, orbit=orbit)

        # Add DEM values
        orbit_interp.height = self['dem']

        # Calculate cartesian coordinates
        orbit_interp.lph2xyz()
        self['X'] = np.reshape(orbit_interp.xyz[0, :], self.block_coor.shape)
        self['Y'] = np.reshape(orbit_interp.xyz[1, :], self.block_coor.shape)
        self['Z'] = np.reshape(orbit_interp.xyz[2, :], self.block_coor.shape)

        # Calculate lat/lon values
        orbit_interp.xyz2ell()
        self['lat'] = np.reshape(orbit_interp.lat, self.block_coor.shape)
        self['lon'] = np.reshape(orbit_interp.lon, self.block_coor.shape)
