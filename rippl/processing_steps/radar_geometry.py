# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import logging

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates

from rippl.processing_steps.deramp import Deramp


class RadarGeometry(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[], reference_slc='reference_slc', overwrite=False, squint_angle=True,
                 off_nadir_angle=True, heading=True, incidence_angle=True, azimuth_angle=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'radar_geometry'
        self.output_info['image_type'] = 'reference_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = []
        self.output_info['data_types'] = []
        if off_nadir_angle:
            self.output_info['file_names'].append('off_nadir_angle')
            self.output_info['data_types'].append('real4')
        if heading:
            self.output_info['file_names'].append('heading')
            self.output_info['data_types'].append('real4')
        if incidence_angle:
            self.output_info['file_names'].append('incidence_angle')
            self.output_info['data_types'].append('real4')
        if azimuth_angle:
            self.output_info['file_names'].append('azimuth_angle')
            self.output_info['data_types'].append('real4')
        if squint_angle:
            self.output_info['file_names'].append('squint_angle')
            self.output_info['data_types'].append('real4')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc', 'reference_slc']
        self.input_info['process_names'] = ['geocode', 'geocode', 'geocode', 'dem']
        self.input_info['file_names'] = ['X', 'Y', 'Z', 'dem']
        self.input_info['data_types'] = ['real4', 'real4', 'real4', 'real4']
        self.input_info['polarisations'] = ['', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['aliases_processing'] = ['X', 'Y', 'Z', 'dem']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc
        self.settings = dict()
        self.settings['squint_angle'] = squint_angle
        self.settings['off_nadir_angle'] = off_nadir_angle
        self.settings['incidence_angle'] = incidence_angle
        self.settings['heading'] = heading
        self.settings['azimuth_angle'] = azimuth_angle

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

        out_coor = self.coordinate_systems['out_coor_chunk']        # type: CoordinateSystem

        # Get the orbit and initialize orbit coordinates
        readfile = self.processing_images['reference_slc'].readfiles['original']
        orbit = self.processing_images['reference_slc'].find_best_orbit('original')
        out_coor.create_radar_lines()
        orbit_interp = OrbitCoordinates(coordinates=out_coor, orbit=orbit)

        # Calc xyz and velocity vector of satellite orbit.
        orbit_interp.height = self['dem']
        orbit_interp.xyz = np.vstack((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))

        if out_coor.grid_type != 'radar_coordinates':
            orbit_interp.load_readfile(readfile)
            lines, pixels = orbit_interp.xyz2lp(orbit_interp.xyz)
            orbit_interp.lp_time(lines, pixels, regular=False)
        else:
            orbit_interp.lp_time()

        # Calc angles based on xyz information from geocoding
        if self.settings['incidence_angle'] or self.settings['azimuth_angle']:
            orbit_interp.xyz2scatterer_azimuth_elevation()
        if self.settings['off_nadir_angle'] or self.settings['heading']:
            orbit_interp.xyz2orbit_heading_off_nadir()

        if self.settings['squint_angle']:
            if out_coor.grid_type == 'radar_coordinates':
                az_grid, ra_grid = Deramp.az_ra_time(out_coor)
                self['squint_angle'] = Deramp.calc_ramp(readfile, orbit, az_grid, ra_grid, calc_squint=True)[1].astype(np.float32)
            else:
                logging.info('Not able to calculate squint if coordinate system is not a radar grid.')
        if self.settings['off_nadir_angle']:
            self['off_nadir_angle'] = np.reshape(orbit_interp.off_nadir_angle, out_coor.shape).astype(np.float32)
        if self.settings['heading']:
            self['heading'] = np.reshape(orbit_interp.heading, out_coor.shape).astype(np.float32)
        if self.settings['incidence_angle']:
            self['incidence_angle'] = np.reshape(90 - orbit_interp.elevation_angle, out_coor.shape).astype(np.float32)
        if self.settings['azimuth_angle']:
            self['azimuth_angle'] = np.reshape(orbit_interp.azimuth_angle, out_coor.shape).astype(np.float32)
