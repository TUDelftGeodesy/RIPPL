'''
This class holds some functions to do a multilooking for regular grids. This is generally much faster than for
irregular input grids and it is therefore helpfull to provide a seperate function.

'''
import numpy as np
import logging

from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class MultilookRegular(object):


    def __init__(self, in_coor, out_coor):
        # Check whether the two coordinate systems are compatible.

        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor        # type: CoordinateSystem
        self.coordinate_systems['out_coor'] = out_coor      # type: CoordinateSystem
        if not in_coor.same_coordinates(out_coor, strict=False):
            raise TypeError('Not possible to do regular multilooking with different types of coordinate systems. '
                            'Only location and grid size can be different. Otherwise use irregular multilooking.')

        self.multilooked = []
        self.samples = []

        self.lines_in, self.pixels_in = self.pixel_line_spacing(in_coor, out_coor)
        self.coverage = self.multilook_coverage(self.lines_in, self.pixels_in)
        if self.coverage == 0:
            logging.info('Warning: input and output of multilooking system do not overlap. Result will consist of only zero '
                  'values.')

    def __call__(self, data_in):
        # Run the actual multilooking script.
        self.multilooked, self.samples = self.regular_multilook(data_in, self.lines_in, self.pixels_in)

    @staticmethod
    def pixel_line_spacing(in_coor, out_coor):
        # type: (CoordinateSystem, CoordinateSystem) -> None
        # This function is used to get the pixel and line coordinate with respect to the output image.

        if in_coor.grid_type == 'radar_coordinates':
            # offset between radar grids
            off = (out_coor.az_time - in_coor.az_time) / in_coor.az_step + out_coor.first_line - in_coor.first_line
            lines_in = (in_coor.interval_lines - off) / out_coor.multilook[0]
            off = (out_coor.ra_time - in_coor.ra_time) / in_coor.ra_step + out_coor.first_pixel - in_coor.first_pixel
            pixels_in = (in_coor.interval_pixels - off) / out_coor.multilook[1]
        elif in_coor.grid_type == 'geographic':
            lines = in_coor.lat0 + (in_coor.first_line + np.arange(in_coor.shape)) * in_coor.dlat
            lines_in = (lines - (out_coor.lat0 + out_coor.first_line * out_coor.dlat)) / out_coor.dlat
            pixels = in_coor.lon0 + (in_coor.first_line + np.arange(in_coor.shape)) * in_coor.dlon
            pixels_in = (pixels - (out_coor.lon0 + out_coor.first_line * out_coor.dlon)) / out_coor.dlon
        elif in_coor.grid_type == 'projection':
            lines = in_coor.y0 + (in_coor.first_line + np.arange(in_coor.shape)) * in_coor.dy
            lines_in = (lines - (out_coor.y0 + out_coor.first_line * out_coor.dy)) / out_coor.dy
            pixels = in_coor.x0 + (in_coor.first_line + np.arange(in_coor.shape)) * in_coor.dx
            pixels_in = (pixels - (out_coor.x0 + out_coor.first_line * out_coor.dx)) / out_coor.dx

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
        number_of_samples_line = np.unique(lin_id, return_counts=True)[1]
        number_of_samples_pix = np.unique(pix_id, return_counts=True)[1]

        number_of_samples = number_of_samples_line[:, None] * number_of_samples_pix[None, :]

        return values_out, number_of_samples
