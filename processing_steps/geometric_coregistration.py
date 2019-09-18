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


class GeometricCoregistration(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], coor_out=[], dem_type='SRTM1',
                 in_image_types=[], in_processes=[], in_file_types=[], in_data_ids=[],
                 slave=[], coreg_master=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
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

        self.process_name = 'geometric_coregistration'
        file_types = ['lines', 'pixels']
        data_types = ['real8', 'real8']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        if len(in_image_types) == 0:
            in_image_types = ['coreg_master', 'coreg_master', 'coreg_master']  # In this case only the slave is used, so the input variable master,
            # coreg_master, ifg and processing_images are not needed.
            # However, if you override the default values this could change.
        in_coor_types = ['coor_out', 'coor_out', 'coor_out']  # Same here but then for the coor_out and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = ['', '', '']
        in_polarisations = ['none', 'none', 'none']
        if len(in_processes) == 0:
            in_processes = ['geocode', 'geocode', 'geocode']
        if len(in_file_types) == 0:
            in_file_types = ['X', 'Y', 'Z']

        in_type_names = ['X_coreg', 'Y_coreg', 'Z_coreg']

        # Inititialize process
        super(GeometricCoregistration, self).__init__(
                       process_name=self.process_name,
                       file_types=file_types,
                       data_id=data_id,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       coor_out=coor_out,
                       in_coor_types=in_coor_types,
                       in_type_names=in_type_names,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       coreg_master=coreg_master,
                       out_processing_image='slave',
                       overwrite=overwrite)

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the slave and coreg_master image.

        :return:
        """

        # Get the orbits
        orbit_slave = self.in_processing_images['slave'].find_best_orbit('original')
        orbit_coreg_master = self.in_processing_images['coreg_master'].find_best_orbit('original')

        # Now initialize the orbit estimation.
        orbit_interp = OrbitCoordinates(coordinates=self.coor_in, orbit=orbit_slave)
        xyz = np.vstack((np.ravel(self['X_coreg'])[None, :], np.ravel(self['Y_coreg'])[None, :], np.ravel(self['Z_coreg'])[None, :]))
        lines, pixels = orbit_interp.xyz2lp(xyz)

        self['lines'] = np.reshape(lines, self.block_coor.shape)
        self['pixels'] = np.reshape(pixels, self.block_coor.shape)
