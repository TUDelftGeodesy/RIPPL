'''
This class holds some functions to do a multilooking for regular grids. This is generally much faster than for
irregular input grids and it is therefore helpfull to provide a seperate function.

'''
import numpy as np

from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class MultilookRegular(object):


    def __init__(self, coor_in, coor_out):
        # Check whether the two coordinate systems are compatible.

        self.coor_in = coor_in
        self.coor_out = coor_out
        self.check_same_coordinate_system(coor_in, coor_out)

        self.lines_in, self.pixels_in = self.pixel_line_spacing(coor_in, coor_out)
        self.coverage = self.multilook_coverage(self.lines_in, self.pixels_in)
        if self.coverage == 0:
            print('Warning: input and output of multilooking system do not overlap.')

    def __call__(self, data_in):
        # Run the actual multilooking script.
        data_out = self.regular_multilook(data_in, self.lines_in, self.pixels_in)

        return data_out

    @staticmethod
    def check_same_coordinate_system(coor_in, coor_out):
        # type: (CoordinateSystem, CoordinateSystem) -> None
        # This function is to check whether the same coordinate system is used for input and output data.

        if not coor_in.grid_type == coor_out.grid_type:
            return False
        if coor_in.grid_type in ['geographic', 'projection']:
            if coor_in.ellipse_type != coor_out.ellipse_type:
                return False
            if coor_in.grid_type == 'projection':
                if coor_in.proj4_str != coor_out.proj4_str:
                    return False
        if coor_in.grid_type == 'radar_coordinates':
            if coor_in.radar_grid_date != coor_out.radar_grid_date:
                return False

        # If passed all other tests return True
        return True

    @staticmethod
    def pixel_line_spacing(coor_in, coor_out):
        # type: (CoordinateSystem, CoordinateSystem) -> None
        # This function is used to get the pixel and line coordinate with respect to the output image.

        if coor_in.grid_type == 'radar_coordinates':
            # offset between radar grids
            off = (coor_out.az_time - coor_in.az_time) / coor_in.az_step + coor_out.first_line - coor_in.first_line
            lines_in = (coor_in.interval_lines - off) / coor_out.multilook[0]
            off = (coor_out.ra_time - coor_in.ra_time) / coor_in.ra_step + coor_out.first_pixel - coor_in.first_pixel
            pixels_in = (coor_in.interval_pixels - off) / coor_out.multilook[1]
        elif coor_in.grid_type == 'geographic':
            lines = coor_in.lat0 + (coor_in.first_line + np.arange(coor_in.shape)) * coor_in.dlat
            lines_in = (lines - (coor_out.lat0 + coor_out.first_line * coor_out.dlat)) / coor_out.dlat
            pixels = coor_in.lon0 + (coor_in.first_line + np.arange(coor_in.shape)) * coor_in.dlon
            pixels_in = (pixels - (coor_out.lon0 + coor_out.first_line * coor_out.dlon)) / coor_out.dlon
        elif coor_in.grid_type == 'projection':
            lines = coor_in.y0 + (coor_in.first_line + np.arange(coor_in.shape)) * coor_in.dy
            lines_in = (lines - (coor_out.y0 + coor_out.first_line * coor_out.dy)) / coor_out.dy
            pixels = coor_in.x0 + (coor_in.first_line + np.arange(coor_in.shape)) * coor_in.dx
            pixels_in = (pixels - (coor_out.x0 + coor_out.first_line * coor_out.dx)) / coor_out.dx

        return lines_in, pixels_in

    @staticmethod
    def multilook_coverage(lines_in, pixels_in, shape_out):
        # Calculate the percentage of the input image that is covered by the output image.

        line_part = np.sum((lines_in > 0) * (lines_in > shape_out[0])) / len(lines_in)
        pixel_part = np.sum((pixels_in > 0) * (pixels_in > shape_out[1])) / len(pixels_in)

        return line_part * pixel_part

    @staticmethod
    def regular_multilook(values, lines_in, pixels_in, shape):
        # Multilooking of a grid, where the output grid is possibly

        lin_in = list(np.floor(lines_in).astype(np.int32))
        pix_in = list(np.floor(pixels_in).astype(np.int32))

        lin_id = [lin_in.index(x) for x in np.arange(shape[0])]
        pix_id = [pix_in.index(x) for x in np.arange(shape[1])]
        last_lin = lin_id[-1] + 1
        last_pix = pix_id[-1] + 1

        values_out = np.add.reduceat(np.add.reduceat(values[:last_lin, :last_pix], lin_id), pix_id, axis=1)

        return values_out
