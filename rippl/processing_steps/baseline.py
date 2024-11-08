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
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'baseline'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['perpendicular_baseline', 'parallel_baseline']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc', 'secondary_slc']
        self.input_info['process_names'] = ['geocode', 'geocode', 'geocode', 'geometric_coregistration']
        self.input_info['file_names'] = ['X', 'Y', 'Z', 'coreg_lines']
        self.input_info['polarisations'] = ['', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['aliases_processing'] = ['X', 'Y', 'Z', 'lines']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

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
        orbit_reference_slc = self.processing_images['reference_slc'].find_best_orbit('original')
        orbit_secondary_slc = self.processing_images['secondary_slc'].find_best_orbit('original')

        s_orbit = OrbitInterpolate(orbit_secondary_slc)
        m_orbit = OrbitInterpolate(orbit_reference_slc)

        readfile_secondary_slc = self.processing_images['secondary_slc'].readfiles['original']
        readfile_primary_slc = self.processing_images['reference_slc'].readfiles['original']
        coor_secondary_slc = CoordinateSystem()
        coor_secondary_slc.create_radar_coordinates()
        coor_secondary_slc.load_readfile(readfile=readfile_secondary_slc)
        s_az = self['lines'] * coor_secondary_slc.az_step + coor_secondary_slc.az_time

        # Get xyz of points on the ground.
        p_xyz = np.concatenate((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))

        if self.coordinate_systems['out_coor_chunk'].grid_type == 'radar_coordinates':
            self.coordinate_systems['out_coor_chunk'].create_radar_lines()
            m_az = self.coordinate_systems['out_coor_chunk'].interval_lines * self.coordinate_systems['out_coor_chunk'].az_step + self.coordinate_systems['out_coor_chunk'].az_time
            lines = np.ravel(
                np.tile(np.arange(self.coordinate_systems['out_coor_chunk'].shape[0])[:, None], (1, self.coordinate_systems['out_coor_chunk'].shape[1]))).astype(np.int32)
        else:
            calc_az = OrbitCoordinates(orbit=orbit_reference_slc, readfile=readfile_primary_slc)
            lines, pixels = calc_az.xyz2lp(p_xyz)
            m_az = readfile_primary_slc.az_first_pix_time + readfile_primary_slc.az_time_step * lines

        # Calc xyz reference_slc and secondary_slc.
        s_orbit.fit_orbit_spline()
        s_xyz = s_orbit.evaluate_orbit_spline(np.ravel(s_az))[0]
        m_orbit.fit_orbit_spline()
        if self.coordinate_systems['out_coor_chunk'].grid_type == 'radar_coordinates':
            m_xyz = m_orbit.evaluate_orbit_spline(m_az)[0][:, lines]
        else:
            m_xyz = m_orbit.evaluate_orbit_spline(m_az)[0]

        baseline_squared = np.reshape(np.sum((m_xyz - s_xyz) ** 2, axis=0), self.coordinate_systems['out_coor_chunk'].shape)

        s_xyz -= p_xyz
        m_xyz -= p_xyz
        self['parallel_baseline'] = np.reshape(np.sqrt(np.sum((m_xyz) ** 2, axis=0)) - np.sqrt(np.sum((s_xyz) ** 2, axis=0)), self.coordinate_systems['out_coor_chunk'].shape)

        m_xyz = m_xyz / np.sqrt(np.sum(m_xyz ** 2, axis=0))
        s_xyz = s_xyz / np.sqrt(np.sum(s_xyz ** 2, axis=0))
        p_xyz = p_xyz / np.sqrt(np.sum(p_xyz ** 2, axis=0))

        angle_m = np.arccos(np.einsum('ij,ij->j', m_xyz, p_xyz)).astype(np.float32)
        del m_xyz
        angle_s = np.arccos(np.einsum('ij,ij->j', s_xyz, p_xyz)).astype(np.float32)
        del s_xyz, p_xyz

        # Define direction of baseline.
        pos_neg = np.reshape(angle_m < angle_s, self.coordinate_systems['out_coor_chunk'].shape)
        del angle_m, angle_s

        self['perpendicular_baseline'] = np.sqrt(baseline_squared - self['parallel_baseline']**2)
        self['perpendicular_baseline'][pos_neg] *= -1
