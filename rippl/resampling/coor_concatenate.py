'''
This class is used as a first step to concatenate different bursts or slices.
The main steps are:
- Check if coordinate systems are the same.
- Check if all coordinate systems align (there can be a shift of 1/2 pixel or something)
- Align all coordinate system, using the same lat0/lon0/x0/y0/ra_time/az_time

'''
import numpy as np
import copy

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.meta_data.readfile import Readfile
from rippl.meta_data.orbit import Orbit
from rippl.resampling.coor_new_extend import CoorNewExtend


class CoorConcatenate():

    def __init__(self, coor_systems, adjust_date=False):
        # Input coordinate systems a list of coordinate systems.

        self.coor_systems = coor_systems
        self.concat_coor, self.sync_coors = CoorConcatenate.create_concat_coordinates(self.coor_systems, adjust_date=adjust_date)

        # Information on readfiles that will be updated.
        self.readfile = []

    def update_readfiles(self, readfile, orbit):
        # type: (CoorConcatenate, Readfile, Orbit) -> bool
        # Update readfile for concatenated image. This is only relevant for the first creation of radar image.

        if not isinstance(readfile, Readfile):
            print('Input variable should be a Readfile object')
            return
        if not isinstance(orbit, Orbit):
            print('Input variable should be a Orbit object')
            return
        if not self.concat_coor.grid_type == 'radar_coordinates':
            print('A new readfile for the concatenated image can only be created using radar coordinate concatenation')
            return

        readfile = copy.deepcopy(readfile)
        self.concat_coor.load_orbit(orbit)          # type: CoordinateSystem
        self.concat_coor.create_radar_lines()

        # First update the relevant values
        readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)'] = 'None'
        readfile.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] = 'None'
        readfile.ra_first_pix_time = self.concat_coor.ra_time
        readfile.az_first_pix_time = self.concat_coor.az_time
        readfile.ra_time_step = self.concat_coor.ra_step
        readfile.az_time_step = self.concat_coor.az_step
        readfile.json_dict['Orig_first_line'] = 'None'
        readfile.json_dict['Orig_first_pixel'] = 'None'
        readfile.first_pixel = self.concat_coor.first_pixel
        readfile.first_line = self.concat_coor.first_line
        readfile.center_lat = self.concat_coor.center_lat
        readfile.center_lon = self.concat_coor.center_lon
        readfile.center_line = self.concat_coor.center_line
        readfile.center_pixel = self.concat_coor.center_pixel
        readfile.center_heading = self.concat_coor.center_heading

        # Get the oblique mercator projection
        orbit_calculations = OrbitCoordinates(orbit=orbit)
        orbit_calculations.load_coordinate_system(self.concat_coor)
        proj_string = orbit_calculations.create_mercator_projection(UTM=False)

        # Calculate the lat/lon coverage of the image based on known radar coordinates
        new_coor = CoordinateSystem()
        new_coor.create_projection(1, 1, proj4_str=proj_string)
        new_coor.load_orbit(orbit)
        new_coverage = CoorNewExtend(self.concat_coor, new_coor)
        coor = new_coverage.out_coor
        y_coor = [coor.y0 + coor.first_line * coor.dy, coor.y0 + (coor.first_line + coor.shape[0]) * coor.dy]
        x_coor = [coor.x0 + coor.first_pixel * coor.dx, coor.x0 + (coor.first_pixel + coor.shape[1]) * coor.dx]
        readfile.poly_coor = [coor.proj2ell(y_coor[1], x_coor[0]), coor.proj2ell(y_coor[1], x_coor[1]),
                              coor.proj2ell(y_coor[0], x_coor[1]), coor.proj2ell(y_coor[0], x_coor[0])]
        readfile.size = [self.concat_coor.shape[0] + self.concat_coor.first_line,
                         self.concat_coor.shape[1] + self.concat_coor.first_pixel]


        # Then remove all irrelevant parts of the new readfiles image (no link to source data or something)
        for key in ['First_line (w.r.t. tiff_image)', 'Last_line (w.r.t. tiff_image)',
                    'First_pixel (w.r.t. tiff_image)', 'Last_pixel (w.r.t. tiff_image)',
                    'Datafile', 'Dataformat', 'Number_of_bursts', 'Burst_number_index',
                    'Number_of_lines_swath', 'Number_of_pixels_swath',
                    'DC_reference_azimuth_time', 'DC_reference_range_time',
                    'Xtrack_f_DC_constant (Hz, early edge)', 'Xtrack_f_DC_linear (Hz/s, early edge)', 'Xtrack_f_DC_quadratic (Hz/s/s, early edge)',
                    'FM_reference_azimuth_time', 'FM_reference_range_time',
                    'FM_polynomial_constant_coeff (Hz, early edge)', 'FM_polynomial_linear_coeff (Hz/s, early edge)', 'FM_polynomial_quadratic_coeff (Hz/s/s, early edge)']:

            readfile.json_dict.pop(key)

        self.readfile = readfile

    @staticmethod
    def check_same_coordinates(coor_systems):
        # type:

        for coor_system in coor_systems:
            coor_system.create_coor_id()
        for coor_system in coor_systems:
            if not coor_system.short_id_str == coor_systems[0].short_id_str:
                return False

        return True

    @staticmethod
    def coordinates_alignment(coor_systems):
        # type:

        if not CoorConcatenate.check_same_coordinates(coor_systems):
            return False

        # make a list with the origin
        orig_lines = np.zeros(len(coor_systems))
        orig_pixels = np.zeros(len(coor_systems))
        step_lines = np.zeros(len(coor_systems))
        step_pixels = np.zeros(len(coor_systems))
        first_lines = np.zeros(len(coor_systems))
        first_pixels = np.zeros(len(coor_systems))
        for id, coor_system in enumerate(coor_systems):
            orig_lines[id], orig_pixels[id], first_lines[id], first_pixels[id], step_lines[id], step_pixels[id] = \
                CoorConcatenate.get_alignment(coor_system)

        for step_line, step_pixel in zip(step_lines, step_pixels):
            if step_line != step_lines[0] or step_pixel != step_pixels[0]:
                print('Step sizes of coordinate systems is not the same, concatenation is not possible.')
                return False

        for orig_line, orig_pixel, first_line, first_pixel, step_line, step_pixel in \
                zip(orig_lines, orig_pixels, first_lines, first_pixels, step_lines, step_pixels):
            # Check if lines and pixels are aligned for concatenation.
            line_rest = (((orig_lines[0] - first_lines[0]) - (orig_line - first_line)) % step_line) / step_line
            pixel_rest = (((orig_pixels[0] - first_pixels[0]) - orig_pixel + first_pixel) % step_pixel) / step_pixel

            if (0.01 > line_rest and 0.99 < line_rest) or (0.01 > pixel_rest and 0.99 < pixel_rest):
                print('Coordinate systems do not align!')
                return False

        return [orig_lines, orig_pixels, first_lines, first_pixels, step_lines, step_pixels]

    @staticmethod
    def synchronize_coordinates(coor_systems):
        # type: (List(CoordinateSystem)) -> (List(CoordinateSystem))
        # Transform all images to a coordinate system with the

        # First get the alignment
        align_data = CoorConcatenate.coordinates_alignment(coor_systems)
        if not align_data:
            return False
        else:
            [orig_lines, orig_pixels, first_lines, first_pixels, step_lines, step_pixels] = align_data

        # Than find the lowest
        orig_line = np.min(orig_lines)
        orig_pixel = np.min(orig_pixels)
        new_coors = []
        for coordinate_system in coor_systems:
            new_coors.append(CoorConcatenate.adjust_orig_vals(coordinate_system, orig_line, orig_pixel))

        return new_coors

    @staticmethod
    def create_concat_coordinates(coor_systems, adjust_date=True):
        # type:
        # first synchronize all coordinate systems.
        sync_coors = CoorConcatenate.synchronize_coordinates(coor_systems)

        # Than get the first/last pixels
        align_data = CoorConcatenate.coordinates_alignment(sync_coors)
        if not align_data:
            return False
        else:
            [orig_lines, orig_pixels, first_lines, first_pixels, step_lines, step_pixels] = align_data

        new_first_line = np.min(first_lines)
        new_first_pix = np.min(first_pixels)

        max_line = np.max([first_line + coor.shape[0] for first_line, coor in zip(first_lines, sync_coors)])
        max_pix = np.max([first_pixel + coor.shape[1] for first_pixel, coor in zip(first_pixels, sync_coors)])

        new_shape = [int(max_line - new_first_line), int(max_pix - new_first_pix)]
        concat_coor = copy.deepcopy(sync_coors[0])              # type: CoordinateSystem
        line_id = np.argmin(orig_lines)
        concat_coor.az_time = orig_lines[line_id] * concat_coor.az_step
        if adjust_date and concat_coor.az_time < 7200:
            concat_coor.az_time += 86400
        pix_id = np.argmin(orig_pixels)
        concat_coor.ra_time = orig_pixels[pix_id] * concat_coor.ra_step
        concat_coor.shape = new_shape
        concat_coor.first_line = int(new_first_line)
        concat_coor.first_pixel = int(new_first_pix)
        concat_coor.center_lat = np.mean([coor.center_lat for coor in coor_systems])
        concat_coor.center_lon = np.mean([coor.center_lon for coor in coor_systems])
        concat_coor.center_heading = np.mean([coor.center_heading for coor in coor_systems])
        concat_coor.center_pixel = np.int(np.mean([coor.center_pixel for coor in coor_systems]))
        concat_coor.center_line = np.int(np.mean([coor.center_line for coor in coor_systems]))

        return concat_coor, sync_coors

    @staticmethod
    def get_alignment(coor):
        # type:

        if coor.grid_type == 'radar_coordinates':
            orig_line = coor.az_time / coor.az_step
            orig_pix = coor.ra_time / coor.ra_step

            step_line = coor.multilook[0] / coor.oversample[0]
            step_pix = coor.multilook[1] / coor.oversample[1]

            first_line = coor.first_line / (coor.multilook[0] / coor.oversample[0])
            first_pix = coor.first_pixel / (coor.multilook[1] / coor.oversample[1])

        elif coor.grid_type == 'geographic':
            orig_line = coor.lat0
            orig_pix = coor.lon0

            step_line = coor.dlat
            step_pix = coor.dlon

            first_line = coor.first_line
            first_pix = coor.first_pixel

        elif coor.grid_type == 'projection':
            orig_line = coor.y0
            orig_pix = coor.x0

            step_line = coor.dy
            step_pix = coor.dx

            first_line = coor.first_line
            first_pix = coor.first_pixel

        return orig_line, orig_pix, first_line, first_pix, step_line, step_pix

    @staticmethod
    def adjust_orig_vals(coor, orig_line, orig_pix):
        # type:
        # Adjust the original line/pixel by changing the first line/pixel

        if coor.grid_type == 'radar_coordinates':
            az_diff = (coor.az_time / coor.az_step) - orig_line
            ra_diff = (coor.ra_time / coor.ra_step) - orig_pix
            coor.az_time += az_diff * coor.az_step
            coor.ra_time += ra_diff * coor.ra_step

            coor.first_line = int(np.round(coor.first_line + az_diff * (coor.multilook[0] / coor.oversample[0])))
            coor.first_pixel = int(np.round(coor.first_pixel + ra_diff * (coor.multilook[1] / coor.oversample[1])))

        elif coor.grid_type == 'geographic':
            coor.first_line = int(coor.first_line + (coor.lat0 - orig_line) / coor.dlat)
            coor.first_pixel = int(coor.first_pixel + (coor.lon0 - orig_pix) / coor.dlon)

            coor.lat0 = orig_line
            coor.lon0 = orig_pix

        elif coor.grid_type == 'projection':
            coor.first_line = int(coor.first_line + (coor.y0 - orig_line) / coor.dy)
            coor.first_pixel = int(coor.first_pixel + (coor.x0 - orig_pix) / coor.dx)

            coor.y0 = orig_line
            coor.x0 = orig_pix

        return coor
