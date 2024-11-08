"""

This class holds the methods to do an irregular multilooking. The possible steps are:
radar coordinates > lat/lon coordinates
radar coordinates > projected coordinates
lat/lon coordinates > radar coordinates

Other combinations are not needed in current processing, but could be added later on.

The processing consists of 2 steps
1. Calculation of how the coordinates of one input set fits in the system of the other. (Results can be saved as a seperate step)
2. Actual multilooking based on a sort and cumulative sum only.

"""

import numpy as np

from rippl.resampling.grid_transforms import GridTransforms
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class MultilookIrregular(object):

    def __init__(self, in_coor, out_coor, sort_ids='', output_ids='', sum_ids=''):
        # type: (MultilookIrregular, CoordinateSystem, CoordinateSystem, np.ndarray, np.ndarray, np.ndarray) -> None
        # Load in and output coordinates
        
        self.in_coor = in_coor
        self.out_coor = out_coor
        
        if len(sort_ids) > 0 and len(sum_ids) > 0 and len(output_ids) > 0:
            self.sort_ids = sort_ids
            self.sum_ids = sum_ids
            self.output_ids = output_ids

        self.multilooked = []
        self.samples = []

    def create_conversion_grid(self, lines='', pixels=''):
        # type: (MultilookIrregular) -> None
        # Create the conversion factors using sort_ids, sum_ids, output_ids

        # We reverse the input and output grid because we want to know the coordinates of the input grid in the output
        # grid, while with resampling it is the other way around.

        if not isinstance(lines, (list, np.ndarray)) or not isinstance(pixels, (list, np.ndarray)):
            grid_transform = GridTransforms(self.out_coor, self.in_coor)
            lines, pixels = grid_transform()
        
        self.sort_ids, self.sum_ids, self.output_ids = self.conversion_grid(lines, pixels, self.out_coor.shape)

    def apply_multilooking(self, data, remove_unvalid=False):
        # type: (MultilookIrregular, np.ndarray) -> None
        # Do the actual multilooking.

        # Pre-assign data.
        self.multilooked = np.zeros(shape=self.out_coor.shape).astype(data.dtype)
        self.samples = np.zeros(shape=self.out_coor.shape).astype(np.int32)

        # Select valid pixels
        valid = ~np.isnan(data) * ~(data == 0)
        data[~valid] = 0

        # Add to output grids.
        self.multilooked[np.unravel_index(self.output_ids, self.multilooked.shape)] = \
            np.add.reduceat(np.ravel(data)[np.ravel(self.sort_ids)], np.ravel(self.sum_ids))

        # If remove unvalid repeat procedure for the valid grid, otherwise just use the sum_ids
        if remove_unvalid:
            self.samples[np.unravel_index(self.output_ids, self.multilooked.shape)] = \
                np.add.reduceat(np.ravel(valid)[np.ravel(self.sort_ids)], np.ravel(self.sum_ids))
        else:
            self.samples[np.unravel_index(self.output_ids, self.multilooked.shape)] = np.diff(
                np.concatenate((np.ravel(self.sum_ids), np.array([len(self.sort_ids)]))))

    def conversion_grid(self, lines, pixels, shape):
        # type: (np.ndarray, np.ndarray, list) -> (np.ndarray, np.ndarray, np.ndarray)
        # This step finds the id of the input grids

        new_lines = lines - self.out_coor.first_line
        new_pixels = pixels - self.out_coor.first_pixel
        inside = (new_lines > 0) * (new_pixels > 0) * (new_lines < shape[0]) * (new_pixels < shape[1])

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        flat_id = np.floor(new_lines).astype(np.int32) * shape[1] + np.floor(new_pixels).astype(np.int32)

        # Sort ids and find number of pixels in every grid cell
        flat_id[inside == False] = -1
        num_outside = np.sum(~inside)
        sort_ids = np.argsort(np.ravel(flat_id))[num_outside:]
        [output_ids, no_ids] = np.unique(np.ravel(flat_id)[sort_ids], return_counts=True)
        sum_ids = np.cumsum(no_ids) - no_ids

        return sort_ids, sum_ids, output_ids
