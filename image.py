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
from image_data import ImageData
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
        process_keys = dict([('readfiles', 66), ('orbits', 0), ('crop', 8), ('import_dem', 0), ('inverse_geocode', 0),
                             ('radar_dem', 0), ('geocode', 0), ('azimuth_elevation_angle', 0), ('deramp', 0),
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

    def geocoding(self, dem_folder='', dem_type='SRTM3', force=False, n_jobs=8, blocks=8, parallel=True):

        orig_slices = [self.slices[key] for key in self.slice_names]
        if not force:
            orig_slices = [slice for slice in orig_slices if slice.process_control['geocode'] != '1']
            if len(orig_slices) == 0:
                return

        lines = [int(slice.processes['crop']['Data_lines']) / blocks + 1 for slice in orig_slices]

        # Geocoding of all slices of the image.
        im = SrtmDownload(dem_folder, 'gertmulder', 'Radar2016', resolution=dem_type, n_processes=n_jobs)

        for slice in orig_slices:
            slice.clean_memory()
            im(slice)

        # First create a DEM and do the geocoding for the slice file.
        start_time = time.time()

        dem_folders = []
        for n in range(len(orig_slices)):
            dem_folders.append(dem_folder)

        for slice in orig_slices:
            slice = InverseGeocode.create_output_files(slice)

        if parallel:
            slices = Parallel(n_jobs=n_jobs)(
                delayed(create_dem)(slice, dem_folder) for slice, dem_folder in zip(orig_slices, dem_folders))
            orig_slices = copy.deepcopy(slices)
            slices = Parallel(n_jobs=n_jobs)(delayed(inverse_geocode)(slice, 0, 0) for slice in orig_slices)
        else:
            slices = []
            for slice, dem_folder in zip(orig_slices, dem_folders):

                slice = create_dem(slice, dem_folder)
                slices.append(inverse_geocode(slice, 0, 0))

        # Prepare geocoding output
        for slice in slices:
            sample, interval, buffer, coors, in_coors, out_coors = FindCoordinates.interval_coors(slice)
            Geocode.add_meta_data(slice, sample, coors, interval, buffer)
            Geocode.create_output_files(slice, sample)
            RadarDem.add_meta_data(slice, sample, coors, interval, buffer)
            RadarDem.create_output_files(slice, sample)
            AzimuthElevationAngle.add_meta_data(slice, sample, coors, interval, buffer, scatterer=True, orbit=True)
            AzimuthElevationAngle.create_output_files(slice, sample, scatterer=True, orbit=True)
            slice.clean_memory()

        slice_mat = []
        lines_mat = []
        blocks_mat = []
        for n in range(blocks):
            slice_mat.extend(copy.copy(slices))
            lines_mat.extend(lines)
            blocks_mat.extend(list(np.ones(len(slices)).astype(np.int32) * n))

        interval = [1, 1]
        buffer = [0, 0]

        if parallel:
            slice_m = Parallel(n_jobs=n_jobs)(delayed(geocoding)(slice, block, lines, interval, buffer) for slice, block, lines in zip(slice_mat, blocks_mat, lines_mat))
        else:
            slice_m = []
            for slice, block, lines in zip(slice_mat, blocks_mat, lines_mat):
                slice_m.append(geocoding(slice, block, lines, interval, buffer))

        slices = slice_m[:len(slices)]
        for key, slice in zip(self.slice_names, slices):
            self.slices[key] = slice

        # Time needed to do geocoding of slice image.
        print("--- %s seconds for geocoding ---" % (time.time() - start_time))


    def sparse_geocoding(self, dem_folder='', dem_type='SRTM3', multilook_coarse='', multilook_fine='', offset='',
                         force=False, n_jobs=8, lines=100, parallel=True):
        # Create a sparse geocoded grid. This is used to resemble the multilooked output grid for the interferograms and
        # weather data.

        im = ImageData(self.res_file, 'single')

        # Create general .res file and read.
        str_ml, multilook_fine, offset, ml_shape, fine_lines, fine_pixels, \
        str_int_coarse, interval_coarse, buffer_coarse, offset_coarse, coarse_shape, coarse_lines, coarse_pixels, \
        str_int_fine, interval_fine, buffer_fine, offset_fine, fine_shape \
            = FindCoordinates.interval_multilook_coors(im, 0, 0, 0, multilook_coarse=multilook_coarse,
                                                       multilook_fine=multilook_fine, offset=offset)

        # Geocoding of all slices of the image.
        down = SrtmDownload(dem_folder, 'gertmulder', 'Radar2016', resolution=dem_type, n_processes=n_jobs)
        down(im)
        srtm, srtm_lines = create_dem_lines(im, dem_folder)

        # First create a DEM and do the geocoding for the slice file.
        start_time = time.time()

        # Prepare inverse geocoding
        InverseGeocode.add_meta_data(im)
        InverseGeocode.create_output_files(im)

        # Get information on the needed interval and buffer for the weather data.
        blocks = srtm_lines / lines + 1
        im_list = [copy.copy(im) for n in range(blocks)]
        blocks_list = range(blocks)

        if parallel:
            slices = Parallel(n_jobs=n_jobs)(
                delayed(inverse_geocode)(slice, block, lines) for slice, block in
                                              zip(im_list, blocks_list))
        else:
            slices = []
            for slice, block in zip(im_list, blocks_list):
                slices.append(inverse_geocode(slice, block, lines))

        im = ImageData(self.res_file, 'single')

        # Prepare geocoding output coarse
        geo = RadarDem(im, interval=interval_coarse, buffer=buffer_coarse)
        Geocode.add_meta_data(im, str_int_coarse, [geo.lines, geo.pixels], interval_coarse, buffer_coarse)
        Geocode.create_output_files(im, str_int_coarse)
        RadarDem.add_meta_data(im, str_int_coarse, [geo.lines, geo.pixels], interval_coarse, buffer_coarse)
        RadarDem.create_output_files(im, str_int_coarse)
        AzimuthElevationAngle.add_meta_data(im, str_int_coarse, [geo.lines, geo.pixels], interval_coarse, buffer_coarse, scatterer=True, orbit=True)
        AzimuthElevationAngle.create_output_files(im, str_int_coarse, scatterer=True, orbit=True)
        im.clean_memory()
        geo = RadarDem(im, interval=interval_fine, buffer=buffer_fine)
        Geocode.add_meta_data(im, str_int_fine, [geo.lines, geo.pixels], interval_fine, buffer_fine)
        Geocode.create_output_files(im, str_int_fine)
        RadarDem.add_meta_data(im, str_int_fine, [geo.lines, geo.pixels], interval_fine, buffer_fine)
        RadarDem.create_output_files(im, str_int_fine)
        AzimuthElevationAngle.add_meta_data(im, str_int_fine, [geo.lines, geo.pixels], interval_fine, buffer_fine, scatterer=True, orbit=True)
        AzimuthElevationAngle.create_output_files(im, str_int_fine, scatterer=True, orbit=True)
        im.write()
        im.clean_memory()

        im = ImageData(self.res_file, 'single')

        # First the coarse grid
        blocks = len(coarse_lines) / lines + 1
        im_list = [copy.copy(im) for n in range(blocks)]
        blocks_list = range(blocks)

        if parallel:
            slices = Parallel(n_jobs=n_jobs)(delayed(geocoding)(slice, block, lines, interval_coarse, buffer_coarse) for slice, block in
                                              zip(im_list, blocks_list))
        else:
            slices = []
            for slice, block in zip(im_list, blocks_list):
                slices.append(geocoding(slice, block, lines, interval_coarse, buffer_coarse))

        del slices
        im = ImageData(self.res_file, 'single')

        # Then the fine grid
        blocks = len(fine_lines) / lines + 1
        im_list = [copy.copy(im) for n in range(blocks)]
        blocks_list = range(blocks)

        if parallel:
            slices = Parallel(n_jobs=n_jobs)(delayed(geocoding)(slice, block, lines, interval_fine, buffer_fine)
                                              for slice, block in zip(im_list, blocks_list))
        else:
            slices = []
            for slice, block in zip(im_list, blocks_list):
                slices.append(geocoding(slice, block, lines, interval_fine, buffer_fine))

        del slices
        ImageData.__init__(self, self.res_file, res_type='single')

        # Time needed to do geocoding of slice image.
        print("--- %s seconds for sparse geocoding ---" % (time.time() - start_time))

    def resample(self, master, force=False, n_jobs=8, blocks=8, parallel=True):
        # Resampling of this image based on a slice image.

        keys = list(set(self.slice_names) & set(master.slice_names))
        if not force:
            keys = [key for key in keys if self.slices[key].process_control['resample'] != '1']
            if len(keys) == 0:
                return

        slaves = [self.slices[key] for key in keys]
        masters = [master.slices[key] for key in keys]

        lines = [int(master.processes['crop']['Data_lines']) / blocks + 1 for master in masters]

        # Processed slices.
        # Then coregister, deramp, resample, reramp and correct for topographic and earth reference phase
        start_time = time.time()

        for slave, master in zip(slaves, masters):
            # Preallocate the resampled data.
            print('Preallocate data for resampling of ' + slave.res_path)

            EarthTopoPhase.add_meta_data(master, slave)
            EarthTopoPhase.create_output_files(slave)
            slave.clean_memory()
            master.clean_memory()

        master_mat = []
        slave_mat = []
        lines_mat = []
        blocks_mat = []
        for n in range(blocks):
            master_mat.extend(copy.copy(masters))
            slave_mat.extend(copy.copy(slaves))
            lines_mat.extend(lines)
            blocks_mat.extend(list(np.ones(len(masters)).astype(np.int32) * n))

        if parallel:

            # pool = mp.Pool(n_jobs)
            # res_files = pool.map(resampling, master_mat, slave_mat, blocks_mat, lines_mat)

            res_files = Parallel(n_jobs=n_jobs)(
               delayed(resampling)(master, slave, block, lines) for master, slave, block, lines in zip(master_mat, slave_mat, blocks_mat, lines_mat))
        else:
            res_files = []
            for master, slave, block, lines in zip(master_mat, slave_mat, blocks_mat, lines_mat):
                res_files.append(resampling(master, slave, block, lines))

        slave_mat = [res[1] for res in res_files]
        slaves = slave_mat[:len(slaves)]

        for key, slice in zip(keys, slaves):
            self.slices[key] = slice

        print("--- %s seconds for resampling ---" % (time.time() - start_time))


    def interferogram(self, slave, multilook='', oversampling='', offset_burst='', offset_image='', force=False, n_jobs=8, parallel=True):
        # Creation of interferogram and coherence images with a certain multilooking factor.
        # This includes concatenation of the slices.

        keys = list(set(self.slice_names) & set(slave.slice_names))

        masters = [self.slices[key] for key in keys]
        slaves = [slave.slices[key] for key in keys]

        if multilook == '':
            multilook = [[5, 20]]
        elif isinstance(multilook[0], int):
            multilook = [multilook]
        if offset_burst == '':
            offset_burst = [0, 0]
        if offset_image == '':
            offset_image = [0, 0]
        if oversampling == '':
            oversampling = [[1, 1]]
        elif isinstance(oversampling[0], int):
            oversampling = [oversampling]

        ml_strings = [FindCoordinates.multilook_str(ml, ovr, offset_image)[0] for ml, ovr in zip(multilook, oversampling)]

        ifg_main = ''

        if not force:
            ifg = Interfero.create_meta_data(masters[0], slaves[0])
            if os.path.exists(os.path.join(os.path.dirname(ifg.folder), 'info.res')):
                ifg_main = ImageData(os.path.join(os.path.dirname(ifg.folder), 'info.res'), 'interferogram')

                for ml_str in ml_strings:
                    if 'Data' + ml_str not in ifg_main.data_files['interferogram'].keys():
                        break
                return

        start_time = time.time()
        new_ifgs = [Interfero.create_meta_data(master, slave) for master, slave in zip(masters, slaves)]

        concat_ifgs = [Concatenate(new_ifgs, multilook=ml, oversampling=ovr, offset_burst=offset_burst).offsets for ml, ovr in
                       zip(multilook, oversampling)]
        n_ifg = len(concat_ifgs[0])
        n_ml = len(concat_ifgs)
        concat_ifgs = [[concat_ifgs[j][i] for j in range(n_ml)] for i in range(n_ifg)]

        if parallel:
            ifgs = Parallel(n_jobs=n_jobs)(delayed(create_ifg)(master, slave, ifg, multilook, oversampling, offset)
                                     for master, slave, ifg, offset in zip(masters, slaves, new_ifgs, concat_ifgs))
        else:
            ifgs = []
            for master, slave, ifg, off in zip(masters, slaves, new_ifgs, concat_ifgs):
                ifgs.append(create_ifg(master, slave, ifg, multilook, oversampling, off))

        for ml, ovr, ml_str in zip(multilook, oversampling, ml_strings):
            im = Concatenate(ifgs, ifg_main, multilook=ml, oversampling=ovr, offset_burst=offset_burst)
            im('interferogram', 'Data', in_data='memory')
            im('coherence', 'Data', in_data='memory')
            im.concat.write()
            im.concat.image_create_disk('interferogram', 'Data' + ml_str)
            im.concat.image_memory_to_disk('interferogram', 'Data' + ml_str)
            im.concat.image_create_disk('coherence', 'Data' + ml_str)
            im.concat.image_memory_to_disk('coherence', 'Data' + ml_str)
            im.concat.clean_memory()

            ifg_main = im.concat

        print("--- %s seconds for creating interferogram ---" % (time.time() - start_time))

        return im.concat
