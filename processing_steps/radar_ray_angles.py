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


class RadarRayAngles(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[],
                 in_processes=[], in_file_types=[], in_data_ids=[],
                 coreg_master=[]):

        """
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

        self.process_name = 'radar_ray_angles'
        file_types = ['off_nadir_angle', 'heading', 'incidence_angle', 'azimuth_angle']
        data_types = ['real4', 'real4', 'real4', 'real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        in_image_types = ['coreg_master', 'coreg_master', 'coreg_master']
        in_coor_types = ['coor_in', 'coor_in', 'coor_in']
        if len(in_data_ids) == 0:
            in_data_ids = ['', '', '']
        in_polarisations = ['none', 'none', 'none']
        if len(in_processes) == 0:
            in_processes = ['geocode', 'geocode', 'geocode']
        if len(in_file_types) == 0:
            in_file_types = ['X', 'Y', 'Z']

        in_type_names = ['X', 'Y', 'Z']

        # Initialize processing step
        super(RadarRayAngles, self).__init__(
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
                       out_processing_image='coreg_master')

    def process_calculations(self):
        """
        Here we calculate the angles that the ray between the satellite and a point on earth make with respect to the
        earths surface.
        The calculated angles are:
        - incidence angle - the angle of the radar signal at the satellite wrt the line to the earth centre.
        - heading - the azimuth angle at which the satellite travels
        - elevation angle - the angle of the radar signal at the point on the ground w.r.t. the earth ellipsoid
        - azimuth angle - the azimuth angle of a point on the ground towards the satellite

        :return:
        """

        # Get the orbit and initialize orbit coordinates
        orbit = self.in_processing_images['coreg_master'].find_best_orbit('original')
        orbit_interp = OrbitCoordinates(coordinates=self.block_coor, orbit=orbit)

        # Calc xyz and velocity vector of satellite orbit.
        orbit_interp.lp_time()

        # Calc angles based on xyz information from geocoding
        orbit_interp.xyz = np.concatenate((self['X'][:, None], self['Y'][:, None], self['Z'][:, None]))
        orbit_interp.xyz2scatterer_azimuth_elevation()
        orbit_interp.xyz2orbit_heading_off_nadir()

        self['off_nadir_angle'] = orbit_interp.off_nadir_angle
        self['heading'] = orbit_interp.heading
        self['incidence_angle'] = (0.5 * np.pi) - orbit_interp.elevation_angle
        self['azimuth_angle'] = orbit_interp.azimuth_angle
