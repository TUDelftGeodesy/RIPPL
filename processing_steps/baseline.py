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
from rippl.orbit_geometry.orbit_interpolate import OrbitInterpolate


class Baseline(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[],
                 in_processes=[], in_file_types=[], in_data_ids=[],
                 slave=[], coreg_master=[]):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

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

        self.process_name = 'radar_ray_angles'
        file_types = ['perpendicular_baseline', 'parallel_baseline', 'total_baseline']
        data_types = ['real4', 'real4', 'real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        in_image_types = ['coreg_master', 'coreg_master', 'coreg_master', 'slave']
        in_coor_types = ['coor_in', 'coor_in', 'coor_in', 'coor_in']
        if len(in_data_ids) == 0:
            in_data_ids = ['', '', '', '']
        in_polarisations = ['none', 'none', 'none', 'none']
        if len(in_processes) == 0:
            in_processes = ['geocode', 'geocode', 'geocode', 'geometrical_coreg']
        if len(in_file_types) == 0:
            in_file_types = ['X', 'Y', 'Z', 'pixels']

        # Initialize processing step
        super(Baseline, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
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
        orbit_coreg_master = self.in_processing_images['coreg_master'].find_best_orbit('original')
        orbit_slave = self.in_processing_images['slave'].find_best_orbit('original')

        s_orbit = OrbitInterpolate(orbit_slave)
        m_orbit = OrbitInterpolate(orbit_coreg_master)

        # TODO add input coordinate system to processing steps and make it accessible.
        # TODO Make input coordinate systems accessible in processes and image data objects.
        coor_slave = self.in_images['lines'].coor_in
        s_az = self.lines * coor_slave.az_step + coor_slave.az_time
        m_az = self.block_coor.interval_lines * self.block_coor.az_step + self.block_coor.az_time

        # Calc xyz coreg master and slave.
        s_orbit.evaluate_orbit_spline(s_az)
        m_orbit.evaluate_orbit_spline(m_az)

        s_xyz = np.concatenate((s_orbit.x, s_orbit.y, s_orbit.z))
        m_xyz = np.concatenate((m_orbit.x, m_orbit.y, m_orbit.z))
        baseline_squared = np.sum((m_xyz - s_xyz) ** 2, axis=0).astype(np.float32)

        p_xyz = np.concatenate((self['X'], self['Y'], self['Z']))
        s_xyz -= p_xyz
        m_xyz -= p_xyz
        self['parallel_baseline'] = np.sqrt(np.sum((m_xyz) ** 2, axis=0)) - np.sqrt(np.sum((s_xyz) ** 2, axis=0))

        m_xyz = m_xyz / np.sqrt(np.sum(m_xyz ** 2, axis=0))
        s_xyz = s_xyz / np.sqrt(np.sum(s_xyz ** 2, axis=0))
        p_xyz = p_xyz / np.sqrt(np.sum(p_xyz ** 2, axis=0))

        angle_m = np.arccos(np.einsum('ij,ij->j', m_xyz, p_xyz)).astype(np.float32)
        del m_xyz
        angle_s = np.arccos(np.einsum('ij,ij->j', s_xyz, p_xyz)).astype(np.float32)
        del s_xyz, p_xyz

        self['perpendicular_baseline'] = np.sqrt(baseline_squared -  self['parallel_baseline'] ** 2)

        # Define direction of baseline.
        pos_neg = angle_m < angle_s

        self['perpendicular_baseline'][pos_neg] *= -1
        baseline_squared[pos_neg] *= -1
        del angle_m, angle_s

        # Finally calculate the other baselines too.
        self['total_baseline'] = np.sqrt(baseline_squared)
