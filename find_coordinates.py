# This class bundles different methods to find the right coordinates or sampling names for different functions.

import numpy as np

class FindCoordinates():

    @staticmethod
    def interval_str(interval, buffer):

        if len(interval) != 2:
            if interval == '':
                interval = [1, 1]
                int_str = ''
            else:
                print('Interval should be a list with 2 values [interval lines, interval pixels]')
                return
        elif list(interval) == [1, 1]:
            int_str = ''
        else:
            int_str = 'int_' + str(interval[0]) + '_' + str(interval[1])

        if len(buffer) != 2:
            if buffer == '':
                buffer = [0, 0]
                buf_str = ''
            else:
                print('Buffer should be a list with 2 values [interval lines, interval pixels]')
                return
        elif list(buffer) == [0, 0]:
            buf_str = ''
        else:
            buf_str = 'buf_' + str(buffer[0]) + '_' + str(buffer[1])

        sample = ''
        if int_str:
            sample = sample + '_' + int_str
        if buf_str:
            sample = sample + '_' + buf_str

        return sample, interval, buffer

    @staticmethod
    def multilook_str(multilook, oversampling, offset):

        if len(multilook) != 2 or multilook == [1, 1]:
            multilook = [5, 20]
            int_str = ''
        else:
            int_str = 'ml_' + str(multilook[0]) + '_' + str(multilook[1])
        if len(oversampling) != 2 or oversampling == [1, 1]:
            oversampling = [1, 1]
            ovr_str = ''
        else:
            if multilook[0] % oversampling[0] == 0 and multilook[1] % oversampling[1] == 0:
                ovr_str = 'ovr_' + str(oversampling[0]) + '_' + str(oversampling[1])
            else:
                print(
                    'Rest of multilook factor divided by oversampling should be zero! Reset oversampling to [1,1]')
                oversampling = [1, 1]
                ovr_str = ''
        if len(offset) != 2 or offset == [0, 0]:
            offset = [0, 0]
            buf_str = ''
        else:
            buf_str = 'off_' + str(offset[0]) + '_' + str(offset[1])

        sample = ''

        if int_str:
            sample = sample + '_' + int_str
        if buf_str:
            sample = sample + '_' + buf_str
        if ovr_str:
            sample = sample + '_' + ovr_str

        return sample, multilook, oversampling, offset

    @staticmethod
    def interval_coors(meta, s_lin=0, s_pix=0, lines=0, interval='', buffer=''):

        sample, interval, buffer = FindCoordinates.interval_str(interval, buffer)

        # First load and create the input data
        orig_s_lin = meta.data_limits['crop']['Data'][0] - buffer[0] - 1
        orig_s_pix = meta.data_limits['crop']['Data'][1] - buffer[1] - 1
        orig_lines = (meta.data_sizes['crop']['Data'][0] + buffer[0] * 2)
        orig_pixels = (meta.data_sizes['crop']['Data'][1] + buffer[1] * 2)

        # Find the coordinate of the first pixel, based on which the new_line and new_pixel are calculated.
        first_line = orig_s_lin
        first_pixel = orig_s_pix
        lines_tot = (first_line + np.arange(orig_lines / interval[0] + 2) * interval[0])
        pixels_tot = (first_pixel + np.arange(orig_pixels / interval[1] + 2) * interval[1])

        shape = [len(lines_tot), len(pixels_tot)]
        if lines != 0:
            shape[0] = lines
        e_lin = s_lin + shape[0]
        e_pix = s_pix + shape[1]

        lines = lines_tot[s_lin:e_lin]
        pixels = pixels_tot[s_pix:e_pix]

        in_coors = [np.min(lines), np.min(pixels), [len(lines), len(pixels)]]
        out_coors = [s_lin, s_pix, [len(lines), len(pixels)]]

        return sample, interval, buffer, [lines_tot, pixels_tot], in_coors, out_coors

    @staticmethod
    def interval_multilook_coors(meta, s_lin=0, s_pix=0, lines=0, multilook='', multilook_coarse='', multilook_fine='', offset=''):
        # The input includes a coarse and fine grid. They are used for:
        #   - coarse: the grid we use to find the ray tracing values for weather models
        #   - fine: the grid we use to interpolate to.
        #   For the coarse grid we should do the full geocoding including elevation and azimuth angles, for the fine
        #   grid only the height is needed.
        #   For the s_lin, s_pix and lines variables we always refer to the fine grid.
        #   Also note that to be able to interpolate from the coarse to the fine grid we will always need a larger extend
        #   of the coarse grid. This is implemented in the last past.
        #
        #   A further important step is that the implemented grids are actually based on a multilooked grid, which should
        #   be transformed to an interval grid here with the postings in the center of every multilooking grid cell.
        #
        #   The output of this function is therefore:
        #   1. The needed interval and buffer for the coarse grid + first line/pixel and size needed from such a file.
        #   2. The needed interval and buffer for the fine grid + the offset that should be applied to go from an
        #      interval grid to an multilooked grid (dependent on offset but if offset is 0, this will be 1)
        #   3. The multilook values. (Only usefull if we fall back to default values)
        #
        #   When implementing the ray-tracing of a weather data model follow these three steps to be sure to get your
        #   your script running.
        #   1. Find or decide on the multilooking factor you want to work on
        #   2. Use this script to find the corresponding fine and coarse intervals/buffers.
        #   3. Run the geocoding for these sparse grids.
        #   4. Run the script to calculate APS's

        if not multilook_fine and multilook_coarse:
            multilook_coarse = [16, 64]
        elif multilook_coarse and not multilook_fine:
            multilook_fine = [4, 16]
        elif multilook and not multilook_fine:
            multilook_fine = multilook
        elif not multilook and not multilook_fine and not multilook_coarse:
            print('Define either the multilook or the combined fine and coarse multilook factors')
        if not offset:
            offset = [0, 0]

        str_ml = FindCoordinates.multilook_str(multilook_fine, oversampling=[1,1], offset=offset)[0]

        # First load and create the input data
        orig_s_lin = meta.data_limits['crop']['Data'][0] - 1
        orig_s_pix = meta.data_limits['crop']['Data'][1] - 1
        orig_lines = meta.data_sizes['crop']['Data'][0]
        orig_pixels = meta.data_sizes['crop']['Data'][1]

        # Redefine az/ra
        az_fine = multilook_fine[0]
        ra_fine = multilook_fine[1]
        fine_s_lin = orig_s_lin + offset[0] + az_fine / 2
        fine_s_pix = orig_s_pix + offset[1] + ra_fine / 2

        # Find the coordinates and buffer for the input dem grid
        lin_off = (fine_s_lin - orig_s_lin) / az_fine + np.minimum(1, (fine_s_lin - orig_s_lin) % az_fine)
        pix_off = (fine_s_pix - orig_s_pix) / ra_fine + np.minimum(1, (fine_s_pix - orig_s_pix) % ra_fine)
        offset_fine = [lin_off, pix_off]
        buffer_fine = -np.array([(fine_s_lin - lin_off * az_fine) - orig_s_lin, (fine_s_pix - pix_off * ra_fine) - orig_s_pix])
        interval_fine = multilook_fine

        fine_n_lines = (orig_lines - offset[0]) / az_fine
        fine_n_pixels = (orig_pixels - offset[1]) / ra_fine
        ml_shape = [fine_n_lines, fine_n_pixels]

        if lines == 0:
            fine_lines = np.arange(fine_n_lines)[s_lin:] * az_fine + fine_s_lin
        else:
            fine_lines = np.arange(fine_n_lines)[s_lin:s_lin + lines] * az_fine + fine_s_lin
        fine_pixels = np.arange(fine_n_pixels)[s_pix:] * ra_fine + fine_s_pix
        fine_shape = [len(fine_lines), len(fine_pixels)]

        # Fine grid naming
        str_int_fine = FindCoordinates.interval_str(interval_fine, buffer_fine)[0]

        # If we also need a coarse grid for APS estimates from ecmwf or harmonie, also coordinates for a coarse grid are
        # calculated.
        if multilook_coarse:
            az_coarse = multilook_coarse[0]
            ra_coarse = multilook_coarse[1]
            coarse_s_lin = orig_s_lin + offset[0] + az_coarse / 2
            coarse_s_pix = orig_s_pix + offset[1] + ra_coarse / 2

            # Find the coordinates and buffer for the input dem grid
            lin_off = (coarse_s_lin - orig_s_lin) / az_coarse + np.minimum(1, (coarse_s_lin - orig_s_lin) % az_coarse)
            pix_off = (coarse_s_pix - orig_s_pix) / ra_coarse + np.minimum(1, (coarse_s_pix - orig_s_pix) % ra_coarse)
            offset_coarse = [lin_off, pix_off]
            buffer_coarse = -np.array([(coarse_s_lin - lin_off * az_coarse) - orig_s_lin, (coarse_s_pix - pix_off * ra_coarse) - orig_s_pix])
            interval_coarse = multilook_coarse

            # Number of pixels
            coarse_lines = np.arange((orig_lines + buffer_coarse[0] * 2) / az_coarse + 2) * az_coarse + orig_s_lin - buffer_coarse[0]
            coarse_pixels = np.arange((orig_pixels + buffer_coarse[1] * 2) / ra_coarse + 2) * ra_coarse + orig_s_pix - buffer_coarse[1]

            coarse_first_lin = np.max(np.argwhere(coarse_lines < fine_lines[0]))
            coarse_first_pix = np.max(np.argwhere(coarse_pixels < fine_pixels[0]))
            coarse_shape = [np.min(np.argwhere(coarse_lines > fine_lines[-1])) - coarse_first_lin + 1,
                            np.min(np.argwhere(coarse_pixels > fine_pixels[-1])) - coarse_first_pix + 1]
            coarse_lines = coarse_lines[coarse_first_lin:coarse_first_lin + coarse_shape[0]]
            coarse_pixels = coarse_pixels[coarse_first_pix:coarse_first_pix + coarse_shape[1]]

            # Coarse grid naming
            str_int_coarse = FindCoordinates.interval_str(interval_coarse, buffer_coarse)[0]

            return str_ml, multilook_fine, offset, ml_shape, fine_lines, fine_pixels, \
                str_int_coarse, interval_coarse, buffer_coarse, offset_coarse, coarse_shape, coarse_lines, coarse_pixels, \
                str_int_fine,   interval_fine,   buffer_fine, offset_fine, fine_shape

        else:
            return str_ml, multilook_fine, offset, ml_shape, fine_lines, fine_pixels, \
                str_int_fine, interval_fine, buffer_fine, offset_fine, fine_shape

    @staticmethod
    def multilook_coors(ifg='', master='', slave='', s_lin=0, s_pix=0, lines=0, multilook='', oversampling='', offset=''):

        meta_data = ''
        if ifg:
            if 'coreg_readfiles' in ifg.processes.keys():
                meta_data = ifg
        if master and not meta_data:
            if 'coreg_readfiles' in master.processes.keys():
                meta_data = master
        if slave and not meta_data:
            if 'coreg_readfiles' in slave.processes.keys():
                meta_data = slave
        if not meta_data:
            print('Processing steps are missing in either ifg, master or slave image')
            return

        sample, multilook, oversampling, offset = FindCoordinates.multilook_str(multilook, oversampling, offset)

        az = multilook[0] / oversampling[0]
        ra = multilook[1] / oversampling[1]
        ovr_lin = float(multilook[0] / 2.0) - (float(multilook[0]) / float(oversampling[0]) / 2)
        ovr_pix = float(multilook[1] / 2.0) - (float(multilook[1]) / float(oversampling[1]) / 2)

        # Extend offset to maintain needed buffer.
        if ovr_lin - float(int(ovr_lin)) > 0.1 or ovr_pix - float(int(ovr_pix)) > 0.1:
            print('Warning. Averaging window for oversampling shifted half a pixel in azimuth or range direction.')
            print('To prevent make sure that oversampling value is odd or multilook/oversampling is even')
            ovr_lin = int(np.ceil(ovr_lin))
            ovr_pix = int(np.ceil(ovr_pix))
        else:
            ovr_lin = int(ovr_lin)
            ovr_pix = int(ovr_pix)

        # Check if offset is large enough
        if offset[0] < ovr_lin:
            print('Offset in lines not large enough to do oversampling (larger window needed.)')
            offset[0] = offset[0] + int(np.ceil(float(offset[0] - ovr_lin) / az))
        if offset[1] < ovr_pix:
            print('Offset in pixels not large enough to do oversampling (larger window needed.)')
            offset[1] = offset[1] + int(np.ceil(float(offset[1] - ovr_pix) / ra))

        # Input pixel
        in_s_lin = s_lin * az + offset[0] - ovr_lin
        in_s_pix = s_pix * ra + offset[1] - ovr_pix

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        shape = meta_data.data_sizes['coreg_crop']['Data']
        out_shape = [(shape[0] - (in_s_lin + ovr_lin + offset[0])) / az, (shape[1] - (in_s_pix + ovr_pix + offset[1])) / ra]

        if lines != 0:
            out_shape[0] = np.minimum(out_shape[0], lines)

        # shape times pixel size + the overlapping area needed of multilooking.
        in_shape = np.array(out_shape) * np.array([az, ra]) + (np.array(multilook) - np.array([az, ra]))
        sample = FindCoordinates.multilook_str(multilook, oversampling, offset)[0]

        return sample, multilook, oversampling, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, out_shape]
