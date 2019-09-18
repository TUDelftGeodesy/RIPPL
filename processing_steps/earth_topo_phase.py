# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.processing_steps.deramp import Deramp


class EarthTopoPhase(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', coor_in=[], in_image_types=[], in_coor_types=[], in_processes=[],
                 in_file_types=[], in_polarisations=[], in_data_ids=[], slave=[], overwrite=False):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        self.process_name = 'earth_topo_phase'
        file_types = ['earth_topo_phase_corrected']
        data_types = ['complex_short']

        if len(in_image_types) == 0:
            in_image_types = ['slave', 'slave']  # In this case only the slave is used, so the input variable master,
            # coreg_master, ifg and processing_images are not needed.
            # However, if you override the default values this could change.
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in', 'coor_in']  # Same here but then for the coor_out and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = ['none', '']
        if len(in_polarisations) == 0:
            in_polarisations = [polarisation, 'none']
        if len(in_processes) == 0:
            in_processes = ['reramp', 'geometric_coregistration']
        if len(in_file_types) == 0:
            in_file_types = ['reramped', 'pixels']

        in_type_names = ['input_data', 'pixels']

        super(EarthTopoPhase, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id, polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       in_coor_types=in_coor_types,
                       in_type_names=in_type_names,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       overwrite=overwrite)

    def process_calculations(self):
        """
        Because of the change in baselines, every pixel is shifted in azimuth time. This also influences the delay time
        to the same point on the ground for different orbits. Therefore we correct here for this effect using the
        geometrical shift in range.

        :return:
        """

        processing_data = self.in_processing_images['slave']
        if not isinstance(processing_data, ImageProcessingData):
            print('Input data missing')

        readfile = processing_data.readfiles['original']
        orbit = processing_data.find_best_orbit('original')
        coor_in = self.in_images['pixels'].in_coordinates

        # Calculate azimuth/range grid and ramp.
        ra_in = np.tile(((np.arange(self.coor_out.shape[1]) + coor_in.first_pixel) * coor_in.ra_step + coor_in.ra_time)
                        [None, :], (self.coor_out.shape[0], 1))

        ra_shift = (self['pixels']  * self.coor_out.ra_step) + self.coor_out.ra_time - ra_in
        c = 299792458
        ramp = np.exp(-1j * (ra_shift * c / readfile.wavelength) * 2 * np.pi).astype(np.complex64)

        # Finally calced the deramped image.
        self['earth_topo_phase_corrected'] = self['input_data'] * np.conjugate(ramp)
