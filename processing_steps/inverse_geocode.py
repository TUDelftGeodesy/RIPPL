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


class InverseGeocode(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], in_processes=[], in_file_types=[], in_data_ids=[], coreg_master=[], overwrite=False):

        """
        This function is used to find the line/pixel coordinates of the dem grid. These can later on be used to
        calculate the radar dem grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

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

        self.process_name = 'inverse_geocode'
        file_types = ['lines', 'pixels']
        data_types = ['real4', 'real4']


        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        in_image_types = ['coreg_master']
        in_coor_types = ['coor_in']
        if len(in_data_ids) == 0:
            in_data_ids = [data_id]
        in_polarisations = ['none']
        if len(in_processes) == 0:
            in_processes = ['create_dem']
        if len(in_file_types) == 0:
            in_file_types = ['dem']

        in_type_names = ['dem']

        super(InverseGeocode, self).__init__(
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
                       coreg_master=coreg_master,
                       out_processing_image='coreg_master',
                       overwrite=overwrite)

    def process_calculations(self):
        """
        We calculate the line and pixel coordinates of the DEM grid as a preparation for the conversion to a radar grid

        :return:
        """

        processing_data = self.in_processing_images['coreg_master']

        coordinates = self.coor_in
        orbit = processing_data.find_best_orbit('original')
        coordinates.load_readfile(processing_data.readfiles['original'])

        # Evaluate the orbit and create orbit coordinates object.
        orbit_interp = OrbitCoordinates(coordinates=coordinates, orbit=orbit)

        if coordinates.grid_type == 'geographic':
            lats, lons = self.block_coor.create_latlon_grid()
        elif coordinates.grid_type == 'projection':
            x, y = self.block_coor.create_xy_grid()
            lats, lons = self.block_coor.proj2ell(x, y)

        xyz = orbit_interp.ell2xyz(np.ravel(lats), np.ravel(lons), np.ravel(self['dem']))
        lines, pixels = orbit_interp.xyz2lp(xyz)

        self['lines'] = np.reshape(lines, coordinates.shape)
        self['pixels'] = np.reshape(pixels, coordinates.shape)
        