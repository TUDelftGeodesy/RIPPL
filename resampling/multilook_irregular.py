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

    def __init__(self, coor_in, coor_out, sort_ids=[], output_ids=[], sum_ids=[]):
        # type: (MultilookIrregular, CoordinateSystem, CoordinateSystem, np.ndarray, np.ndarray, np.ndarray) -> None
        # Load in and output coordinates
        
        self.coor_in = coor_in
        self.coor_out = coor_out
        
        if len(sort_ids) > 0 and len(sum_ids) > 0 and len(output_ids) > 0:
            self.sort_ids = sort_ids
            self.sum_ids = sum_ids
            self.output_ids = output_ids

        self.multilooked = []
    
    def create_conversion_grid(self):
        # type: (MultilookIrregular) -> None
        # Create the conversion factors using sort_ids, sum_ids, output_ids

        # We reverse the input and output grid because we want to know the coordinates of the input grid in the output
        # grid, while with resampling it is the other way around.
        grid_transform = GridTransforms(self.coor_out, self.coor_in)
        lines, pixels = grid_transform()
        
        self.sort_ids, self.sum_ids, self.output_ids = self.conversion_grid(lines, pixels, self.coor_in.shape)

    def apply_multilooking(self, data):
        # type: (MultilookIrregular, np.ndarray) -> None
        # Do the actual multilooking.

        # Preassign data.
        self.multilooked = np.zeros(shape=self.coor_out.shape).astype(data.dtype)

        # Add to output grids.
        self.multilooked[np.unravel_index(self.output_ids, self.multilooked.shape)] = \
            np.add.reduceat(np.ravel(data)[np.ravel(self.sort_ids)], np.ravel(self.sum_ids))

    @staticmethod
    def conversion_grid(lines, pixels, shape):
        # type: (np.ndarray, np.ndarray, list) -> (np.ndarray, np.ndarray, np.ndarray)
        # This step finds the id of the input grids
        inside = (lines > 0) * (pixels > 0) * (lines < shape[0]) * (lines < shape[1])

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        flat_id = np.floor(lines) * shape[0] + np.floor(pixels)

        # Sort ids and find number of pixels in every grid cell
        flat_id[inside == False] = -1
        num_outside = np.sum(~inside)
        sort_ids = np.argsort(np.ravel(flat_id))[num_outside:]
        [output_ids, no_ids] = np.unique(flat_id[np.unravel_index(sort_ids, shape)], return_counts=True)
        sum_ids = np.cumsum(no_ids) - no_ids

        return sort_ids, sum_ids, output_ids
