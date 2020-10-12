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
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class Baseline(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[],
                 in_processes=[], in_file_types=[], in_data_ids=[],
                 slave='slave', coreg_master='coreg_master', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'baseline'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['perpendicular_baseline', 'parallel_baseline']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'slave']
        self.input_info['process_types'] = ['geocode', 'geocode', 'geocode', 'geometric_coregistration']
        self.input_info['file_types'] = ['X', 'Y', 'Z', 'coreg_lines']
        self.input_info['polarisations'] = ['', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['type_names'] = ['X', 'Y', 'Z', 'lines']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(Baseline, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

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
        orbit_coreg_master = self.processing_images['coreg_master'].find_best_orbit('original')
        orbit_slave = self.processing_images['slave'].find_best_orbit('original')

        s_orbit = OrbitInterpolate(orbit_slave)
        m_orbit = OrbitInterpolate(orbit_coreg_master)

        readfile_slave = self.processing_images['slave'].readfiles['original']
        readfile_master = self.processing_images['coreg_master'].readfiles['original']
        coor_slave = CoordinateSystem()
        coor_slave.create_radar_coordinates()
        coor_slave.load_readfile(readfile=readfile_slave)
        s_az = self['lines'] * coor_slave.az_step + coor_slave.az_time

        # Get xyz of points on the ground.
        p_xyz = np.concatenate((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))

        if self.coordinate_systems['block_coor'].grid_type == 'radar_coordinates':
            self.coordinate_systems['block_coor'].create_radar_lines()
            m_az = self.coordinate_systems['block_coor'].interval_lines * self.block_coor.az_step + self.block_coor.az_time
        else:
            calc_az = OrbitCoordinates(orbit=orbit_coreg_master, readfile=readfile_master)
            lines, pixels = calc_az.xyz2lp(p_xyz)
            m_az = readfile_master.az_first_pix_time + readfile_master.az_time_step * lines

        # Calc xyz coreg master and slave.
        s_orbit.fit_orbit_spline()
        s_xyz = s_orbit.evaluate_orbit_spline(np.ravel(s_az))[0]
        lines = np.ravel(np.tile(np.arange(self.block_coor.shape[0])[:, None], (1, self.block_coor.shape[1]))).astype(np.int32)
        m_orbit.fit_orbit_spline()
        if self.coordinate_systems['block_coor'].grid_type == 'radar_coordinates':
            m_xyz = m_orbit.evaluate_orbit_spline(m_az)[0][:, lines]
        else:
            m_xyz = m_orbit.evaluate_orbit_spline(m_az)[0]

        baseline_squared = np.reshape(np.sum((m_xyz - s_xyz) ** 2, axis=0), self.block_coor.shape)

        s_xyz -= p_xyz
        m_xyz -= p_xyz
        self['parallel_baseline'] = np.reshape(np.sqrt(np.sum((m_xyz) ** 2, axis=0)) - np.sqrt(np.sum((s_xyz) ** 2, axis=0)), self.block_coor.shape)

        m_xyz = m_xyz / np.sqrt(np.sum(m_xyz ** 2, axis=0))
        s_xyz = s_xyz / np.sqrt(np.sum(s_xyz ** 2, axis=0))
        p_xyz = p_xyz / np.sqrt(np.sum(p_xyz ** 2, axis=0))

        angle_m = np.arccos(np.einsum('ij,ij->j', m_xyz, p_xyz)).astype(np.float32)
        del m_xyz
        angle_s = np.arccos(np.einsum('ij,ij->j', s_xyz, p_xyz)).astype(np.float32)
        del s_xyz, p_xyz

        # Define direction of baseline.
        pos_neg = np.reshape(angle_m < angle_s, self.block_coor.shape)
        del angle_m, angle_s

        self['perpendicular_baseline'] = np.sqrt(baseline_squared - self['parallel_baseline']**2)
        self['perpendicular_baseline'][pos_neg] *= -1
