# This class analyzes one SLC using its orbits/datafile
# The main functionality of this class is:
#   - Read in a data and metafile in memory
#   - Write information in the shape of a .res file
#   - Read information as a .res file, including datafile and further processed files
# It uses other classes to do calculations or perform analysis.
#   - deramping_reramping > deramping and reramping of the image.
#   - spectral_analysis > create a split spectrum image / high-pass filter / low-pass filter in range and azimuth
#   - orbit_coordinates > to calculate the orbit and ground location of lines and pixels
#   - resample (function) > resample using new output pixels
# Finally, there is an option to determine the overlap between different slices. This can be used for:
#   - ESD calculations
#   - Movements in azimuth direction, which are basically the same as ESD
#   - Ionospheric corrections, which is also calculated in the same way.
# These three steps can be calculated

# An image can either consist of one image and its corresponding information or a number of slices, which will be the
# bursts in the case of Sentinel. It is therefore important to say whether you want to work with the slices or the full
# image.

# This test is used to test a number of processing steps on one slave and slice burst from Sentinel-1 data.
import os
import time
import copy
from joblib import Parallel, delayed
from coordinate_system import CoordinateSystem
from image_data import ImageData
from collections import OrderedDict
import numpy as np

from find_coordinates import FindCoordinates

from orbit_dem_functions.srtm_download import SrtmDownload
from processing_steps.radar_dem import RadarDem
from processing_steps.geocode import Geocode
from processing_steps.earth_topo_phase import EarthTopoPhase
from processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
from processing_steps.interfero import Interfero
from processing_steps.concatenate import Concatenate
from parallel_functions import create_ifg, geocoding, inverse_geocode, resampling, create_dem, create_dem_lines
from processing_steps.inverse_geocode import InverseGeocode

# from pipeline import Pipeline

class Image(ImageData):

    """
    :type reference_image = Image
    :type slice_list = list
    """

    def __init__(self, folder, reference_image='', slice_list='', update_full_image=False):
        # Either give an xml_file or a res_file as input to define the metadata of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        self.res_file = os.path.join(folder, 'info.res')

        # Add the reference image. We use this image to reference other images to. If this image is missing coregistration
        # is not possible.
        self.slice_reference = reference_image

        # Create the list of burst of this image
        self.slices = dict()

        # Read the individual slices ('slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol)
        # Length of these folders should be 20 characters
        self.slice_folders = next(os.walk(folder))[1]
        self.slice_names = sorted([x for x in self.slice_folders if len(x) == 20])

        # If a slice list is defined remove the slices which are not included.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))
        self.slice_folders = [os.path.join(folder, x) for x in self.slice_names]

        for slice_folder, slice_name in zip(self.slice_folders, self.slice_names):
            self.slices[slice_name] = ImageData(os.path.join(slice_folder, 'info.res'), 'single')

        self.check_valid_burst_res()

        if not os.path.exists(self.res_file) or update_full_image:
            concat_dat = Concatenate([self.slices[key] for key in self.slices.keys()])
            concat_dat.concat.write(self.res_file)
        ImageData.__init__(self, self.res_file, res_type='single')


    def check_valid_burst_res(self):
        # This function does some basic checks whether all bursts in this image are correct.

        # Process keys and expected lengths. (0 if unknown or variable)
        process_keys = dict([('readfiles', 66), ('orbits', 0), ('crop', 8), ('import_DEM', 0), ('inverse_geocode', 0),
                             ('radar_DEM', 0), ('geocode', 0), ('azimuth_elevation_angle', 0), ('deramp', 0),
                             ('sim_amplitude', 0), ('coreg_readfiles', 0),('coreg_orbits', 0), ('coreg_crop', 0),
                             ('geometrical_coreg', 0),('correl_coreg', 0), ('combined_coreg', 0),('master_timing', 0),
                             ('oversample', 0),('resample', 0), ('reramp', 0),('earth_topo_phase', 0), ('filt_azi', 0),
                             ('filt_range', 0), ('NWP_phase', 0), ('structure_function', 0), ('split_spectrum', 0)])


        # Check process control and expected length.
        for slice, slice_folder in zip(self.slice_names, self.slice_folders):
            err = False

            if slice not in self.slices.keys():
                print('slice ' + slice + ' is missing in ' + self.folder)
                continue
            else:
                res_dat = self.slices[slice]

            # Try to find the different parts of this slice
            processes = res_dat.process_control.keys()

            for process in process_keys:
                if process not in processes:
                    print('Something wrong with the process control section in ' + res_dat.folder)
                    err = True
                    break

                elif res_dat.process_control[process] not in ['0', '1']:
                    print('Something wrong with the process control section in ' + res_dat.folder)
                    err = True
                    break

                if res_dat.process_control[process] == '1':

                    if process not in res_dat.processes.keys():
                        print('Process ' + process + ' mentioned in process control but no description further on.')
                        print('Error in ' + res_dat.folder)
                        err = True
                        break

                    elif process_keys[process] != 0:
                        if len(res_dat.processes[process].keys()) != process_keys[process]:
                            print('Expected length of ' + str(process_keys[process]) + ' for process ' + process + ' is different from the real length ' + str(len(res_dat.processes[process].keys())))
                            print('Error in ' + res_dat.folder)
                            err = True
                            break

            if err:
                print('Slice ' + res_dat.folder + ' removed from image.')
                self.slice_folders.remove(slice_folder)
                self.slice_names.remove(slice)
                self.slices.pop(slice)

    def add_master_res_info(self):
        # This function adds the .res information specific for the master image. For this image both resampline and the
        # topography/earth phase are not needed as it is the reference itself. Therefore these steps are added to the
        # .res file but simply referenced to the original data crop.

        # Note that this could cause a problem if multilooking of the original data is applied before creating the ifg.
        # However, this is not advised in almost all cases

        coor = CoordinateSystem()
        coor.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])

        for step in ['resample', 'reramp', 'earth_topo_phase']:
            for slice in self.slices.keys():

                res_info = OrderedDict()
                coor.add_res_info(self.slices[slice])

                res_info = coor.create_meta_data([step], ['complex_int'])
                res_info[step + '_output_file'] = 'crop.raw'
                slice.image_add_processing_step(step, res_info)
