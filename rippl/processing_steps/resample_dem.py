# Try to do all calculations using numpy functions.
import numpy as np
from scipy.interpolate import RectBivariateSpline
import os

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_irregular2regular import Irregular2Regular
from rippl.user_settings import UserSettings
from rippl.external_dems.geoid import GeoidInterp
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class ResampleDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], resample_type='delaunay', reference_slc='reference_slc',
                 buffer=0, rounding=0, min_height=0, max_height=0, overwrite=False, dem_type='', ocean=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampling etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'dem'
        self.output_info['image_type'] = 'reference_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['dem']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        if ocean:
            # In this case we can assume that the full image DEM is at sea level, so only geoid is relevant.
            self.input_info['image_types'] = []
            self.input_info['process_names'] = []
            self.input_info['file_names'] = []
            self.input_info['polarisations'] = []
            self.input_info['data_ids'] = []
            self.input_info['coor_types'] = []
            self.input_info['in_coor_types'] = []
            self.input_info['aliases_processing'] = []
        elif resample_type == 'delaunay':
            self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc']
            self.input_info['process_names'] = ['dem', 'inverse_geocode', 'inverse_geocode']
            self.input_info['file_names'] = ['dem', 'lines', 'pixels']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor']
            self.input_info['in_coor_types'] = ['', '', '']
            self.input_info['aliases_processing'] = ['input_dem', 'lines', 'pixels']
        elif in_coor.grid_type == 'radar_coordinates' or out_coor.grid_type == 'radar_coordinates':
            self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc']
            self.input_info['process_names'] = ['dem', 'reproject', 'reproject']
            self.input_info['file_names'] = ['dem', 'in_coor_lines', 'in_coor_pixels']
            self.input_info['data_types'] = ['real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', 'in_coor', 'in_coor']
            self.input_info['aliases_processing'] = ['input_dem', 'lines', 'pixels']
        else:
            self.input_info['image_types'] = ['reference_slc', 'reference_slc']
            self.input_info['process_names'] = ['dem', 'crop']
            self.input_info['file_names'] = ['dem', 'crop']
            self.input_info['polarisations'] = ['', '']
            self.input_info['data_ids'] = [data_id, '']
            self.input_info['coor_types'] = ['in_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', '']
            self.input_info['aliases_processing'] = ['input_dem', 'out_coor_grid']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['ocean'] = ocean
        settings = UserSettings()
        settings.load_settings()
        self.settings['geoid_file'] = os.path.join(settings.settings['paths']['DEM_database'], 'geoid', 'egm96.dat')
        self.settings['dem_type'] = dem_type
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = buffer
        self.settings['in_coor']['rounding'] = rounding
        self.settings['in_coor']['min_height'] = min_height
        self.settings['in_coor']['max_height'] = max_height

    def process_calculations(self):
        """
        Calculate radar DEM using the

        :return:
        """

        coor_chunk = self.coordinate_systems['out_coor_chunk']
        if self.settings['ocean'] and coor_chunk.grid_type == 'radar_coordinates':
            # Only geoid has to be calculated. Start with (lat/lon) coordinates of the corners
            orbit = self.processing_images['reference_slc'].find_best_orbit('original')
            self.coordinate_systems['out_coor_chunk'].create_radar_lines()
            orbit_interp = OrbitCoordinates(coordinates=self.coordinate_systems['out_coor_chunk'], orbit=orbit)

            # Get lines/pixels convex hull and get lat/lon for these pixels
            orbit_interp.height = np.zeros((4))
            lines, pixels = CoorNewExtend.radar_convex_hull(coor_chunk, corners_only=True)
            orbit_interp.lp_time(lines=lines, pixels=pixels)
            orbit_interp.lph2xyz()
            orbit_interp.xyz2ell()
            lats = orbit_interp.lat
            lons = orbit_interp.lon
            pixels_r = np.arange(coor_chunk.shape[1]) / coor_chunk.shape[1]
            lines_r = np.arange(coor_chunk.shape[0]) / coor_chunk.shape[2]
            # Do a bilinear interpolation
            grid_lats = ((lines_r * lats[3] + (1 - lines_r) * lats[0])[:, None] * (1 - pixels_r)[None, :] +
                        (lines_r * lats[2] + (1 - lines_r) * lats[1])[:, None] * pixels_r[None, :])
            grid_lons = ((lines_r * lons[3] + (1 - lines_r) * lons[0])[:, None] * (1 - pixels_r)[None, :] +
                        (lines_r * lons[2] + (1 - lines_r) * lons[1])[:, None] * pixels_r[None, :])

            geoid = GeoidInterp.create_geoid(egm_96_file=self.settings['geoid_file'], lat=grid_lats, lon=grid_lons, download=False)
            self['dem'] = np.reshape(geoid, coor_chunk.shape)
        elif coor_chunk.grid_type == 'radar_coordinates':
            resample = Irregular2Regular(self['lines'], self['pixels'], self['input_dem'], coordinates=coor_chunk)
            self['dem'] = resample().astype(np.float32)
            if np.isnan(self['dem']).any():
                raise TypeError('Resampled DEM contains NaN values! Please adjust the input size of your input DEM to '
                                'be sure the radar grid is fully covered, using buffer/rounding/min height/max height.')
        else:
            if coor_chunk.grid_type == 'geographic':
                lats, lons = coor_chunk.create_latlon_grid()
            elif coor_chunk.grid_type == 'projection':
                x, y = coor_chunk.create_xy_grid()
                lats, lons = coor_chunk.proj2ell(x, y)

            in_coor = self.coordinate_systems['in_coor_chunk']
            lats_in = in_coor.lat0 + (in_coor.first_line + np.arange(in_coor.shape[0])) * in_coor.dlat
            lons_in = in_coor.lon0 + (in_coor.first_pixel + np.arange(in_coor.shape[1])) * in_coor.dlon

            if self.settings['ocean']:
                geoid = GeoidInterp.create_geoid(egm_96_file=self.settings['geoid_file'], lat=lats_in, lon=lons_in,
                                                 download=False)
                self['dem'] = geoid
            else:
                bilinear_interp = RectBivariateSpline(lats_in, lons_in, self['input_dem'])
                self['dem'] = bilinear_interp.ev(lats, lons).astype(np.float32)
