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

