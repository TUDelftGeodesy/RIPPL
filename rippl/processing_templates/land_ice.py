from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
import gdal
from osgeo import osr
from osgeo import ogr
import os


class LandIce(GeneralPipelines):

    def calc_ice_movement(self):
        """
        Calculates the movement of the ice using the resampled data as a starting point.

        :return:
        """

        print('Working on this')
