# This class bundles different methods to find the right coordinates or sampling names for different functions.

import numpy as np


class SparseCoordinates():

    @staticmethod
    def multilook_str(multilook, oversample, offset):

        if len(multilook) != 2 or multilook == [1, 1]:
            multilook = [1, 1]
            int_str = ''
        else:
            int_str = 'ml_' + str(multilook[0]) + '_' + str(multilook[1])
        if len(oversample) != 2 or oversample == [1, 1]:
            oversample = [1, 1]
            ovr_str = ''
        else:
            if multilook[0] % oversample[0] == 0 and multilook[1] % oversample[1] == 0:
                ovr_str = 'ovr_' + str(oversample[0]) + '_' + str(oversample[1])
            else:
                print('Rest of multilook factor divided by oversample should be zero! Reset oversample to [1,1]')
                oversample = [1, 1]
                ovr_str = ''
        if len(offset) != 2 or offset == [0, 0]:
            offset = [0, 0]
            buf_str = ''
        else:
            if offset[0] < 0:
                str_1 = 'm' + str(np.abs(offset[0]))
            else:
                str_1 = str(offset[0])
            if offset[1] < 0:
                str_2 = 'm' + str(np.abs(offset[1]))
            else:
                str_2 = str(offset[1])

            buf_str = 'off_' + str_1 + '_' + str_2

        sample = ''

        if int_str:
            sample = sample + '_' + int_str
        if buf_str:
            sample = sample + '_' + buf_str
        if ovr_str:
            sample = sample + '_' + ovr_str

        return sample, multilook, oversample, offset

    @staticmethod
    def multilook_coors(in_shape, s_lin=0, s_pix=0, lines=0, first_line=0, first_pixel=0, multilook='', oversample='', offset='', interval=False):

        sample, multilook, oversample, offset = SparseCoordinates.multilook_str(multilook, oversample, offset)

        if multilook[0] % oversample[0] != 0 or multilook[1] / oversample[1] == 0:
            print('oversample does not fit in multilooking window. Make sure that multilooking/oversample is an'
                  'integer number! Aborting...')
            return

        az = multilook[0] // oversample[0]
        ra = multilook[1] // oversample[1]
        ovr_lin = float(multilook[0] / 2.0) - (float(multilook[0]) / float(oversample[0]) / 2)
        ovr_pix = float(multilook[1] / 2.0) - (float(multilook[1]) / float(oversample[1]) / 2)

        # Extend offset to maintain needed buffer.
        if ovr_lin % 1 > 0.1 or ovr_pix % 1 > 0.1:
            print('Warning. Averaging window for oversample shifted half a pixel in azimuth or range direction.')
            print('To prevent make sure that oversample value is odd or multilook/oversample is even')

        if ovr_lin % 1 > 0.1:
            ovr_lin = int(np.ceil(ovr_lin))
        else:
            ovr_lin = int(np.round(ovr_lin))
        if ovr_pix % 1 > 0.1:
            ovr_pix = int(np.ceil(ovr_pix))
        else:
            ovr_pix = int(np.round(ovr_pix))

        # Check if offset is large enough
        """
        if not interval:
            if offset[0] < ovr_lin or offset[1] < ovr_pix:
                if offset[0] < ovr_lin:
                    print('Offset in lines not large enough to do oversample (larger window needed.)')
                    offset[0] = offset[0] + int(np.ceil(float(offset[0] - ovr_lin) // az))
                if offset[1] < ovr_pix:
                    print('Offset in pixels not large enough to do oversample (larger window needed.)')
                    offset[1] = offset[1] + int(np.ceil(float(offset[1] - ovr_pix) // ra))

                # We will have to find a new sample string to!
                sample, multilook, oversample, offset = SparseCoordinates.multilook_str(multilook, oversample, offset)
        """

        # Adjust the first line/pixel based on first_line / first_pixel
        if first_line != 0:
            offset[0] += (first_line + offset[0]) % az
            in_s_lin = s_lin * az + offset[0] - ovr_lin + first_line
        else:
            in_s_lin = s_lin * az + offset[0] - ovr_lin

        if first_pixel != 0:
            offset[1] += (first_pixel + offset[1]) % ra
            in_s_pix = s_pix * ra + offset[1] - ovr_pix + first_pixel
        else:
            in_s_pix = s_pix * ra + offset[1] - ovr_pix

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        out_shape = [(in_shape[0] - (in_s_lin + ovr_lin + offset[0])) // az, (in_shape[1] - (in_s_pix + ovr_pix + offset[1])) // ra]

        if lines != 0:
            out_shape[0] = np.minimum(out_shape[0], lines)

        # shape times pixel size + the overlapping area needed of multilooking.
        in_shape = np.array(out_shape) * np.array([az, ra]) + (np.array(multilook) - np.array([az, ra]))
        sample = SparseCoordinates.multilook_str(multilook, oversample, offset)[0]

        return sample, multilook, oversample, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, out_shape]

    @staticmethod
    def multilook_lines(in_shape, s_lin=0, s_pix=0, lines=0, first_line=0, first_pixel=0, multilook='', oversample='', offset='', interval=False):

        sample, multilook, oversample, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, out_shape] = \
            SparseCoordinates.multilook_coors(in_shape, s_lin, s_pix, lines, first_line, first_pixel, multilook, oversample, offset, interval)

        lines_in = in_s_lin + np.arange(in_shape[0])
        pixels_in = in_s_pix + np.arange(in_shape[1])

        lines_out = s_lin + np.arange(out_shape[0]) * (multilook[0] // oversample[0]) + offset[0]
        pixels_out = s_pix + np.arange(out_shape[1]) * (multilook[1] // oversample[1]) + offset[1]

        return sample, multilook, oversample, offset, [lines_in, pixels_in], [lines_out, pixels_out]

    @staticmethod
    def find_slices_offset(in_shape, first_line=0, first_pixel=0, multilook='', oversample='', offset='', slices_start='', slice_offset=''):
        # This function finds the needed offset of a slices starting at the slices_start coordinates to be exactly in
        # line with the overall image. The given slice_offset is a minimal slices offset (for example in the case of
        # empty lines after resampling or disalignment of master and slave)

        sample, multilook, oversample, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, out_shape] = \
            SparseCoordinates.multilook_coors(in_shape, 0, 0, 0, first_line, first_pixel, multilook, oversample, offset)

        # In case we only need coordinates for one slice.
        if isinstance(slices_start[0], int):
            slices_start = [slices_start]
        new_slices_offset = []

        for slice_start in slices_start:
            eff_ml = np.array([(multilook[0] // oversample[0]), (multilook[1] // oversample[1])])
            slice_first = np.array(slice_start) + np.array(slice_offset)

            # Now check whether the overlap due to oversampling is large enough if this configuration is used.
            # Otherwise shift a pixel.
            ovr_lin = float(multilook[0] / 2.0) - (float(multilook[0]) / float(oversample[0]) / 2)
            ovr_pix = float(multilook[1] / 2.0) - (float(multilook[1]) / float(oversample[1]) / 2)

            in_offset = np.array([int(slice_first[0] + ovr_lin - in_s_lin), int(slice_first[1] + ovr_pix - in_s_pix)])
            out_offset = in_offset // eff_ml + (in_offset % eff_ml > 0)

            # Now calculate the offset for this slide
            new_slices_offset.append([in_s_lin + out_offset[0] * eff_ml[0] - slice_start[0],
                                      in_s_pix + out_offset[1] * eff_ml[1] - slice_start[1]])

        return new_slices_offset

    @staticmethod
    def interval_coors(in_shape, s_lin=0, s_pix=0, lines=0, first_line=0, first_pixel=0, multilook='', oversample='', offset='', interval=True):

        sample, multilook, oversample, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, shape] = \
            SparseCoordinates.multilook_coors(in_shape, s_lin, s_pix, lines, first_line, first_pixel, multilook, oversample, offset, interval)

        ml_shift = (np.array(multilook).astype(np.float32) - np.array([1,1])) // 2

        if ml_shift[0] % 1 > 0.1 or ml_shift[1] % 1 > 0.1:
            # print('Warning. Interval coordinates do not start with an integer number.')

            s_lin = in_s_lin + ml_shift[0]
            s_pix = in_s_pix + ml_shift[1]
        else:
            s_lin = int(in_s_lin + ml_shift[0])
            s_pix = int(in_s_pix + ml_shift[1])

        return sample, multilook, oversample, offset, [s_lin, s_pix, shape]

    @staticmethod
    def interval_lines(in_shape, s_lin=0, s_pix=0, lines=0, first_line=0, first_pixel=0, multilook='', oversample='', offset='', interval=True):

        sample, multilook, oversample, offset, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, out_shape] = \
            SparseCoordinates.multilook_coors(in_shape, s_lin, s_pix, lines, first_line, first_pixel, multilook, oversample, offset, interval)

        ml_shift = (np.array(multilook).astype(np.float32) - np.array([1, 1])) // 2

        if ml_shift[0] % 1 > 0.1 or ml_shift[1] % 1 > 0.1:
            # print('Warning. Interval coordinates do not have integer numbers.')

            lines = in_s_lin + np.arange(out_shape[0]) * (multilook[0] // oversample[0]) + ml_shift[0]
            pixels = in_s_pix + np.arange(out_shape[1]) * (multilook[1] // oversample[1]) + ml_shift[1]

        else:
            lines = in_s_lin + np.arange(out_shape[0]) * (multilook[0] // oversample[0]) + int(ml_shift[0])
            pixels = in_s_pix + np.arange(out_shape[1]) * (multilook[1] // oversample[1]) + int(ml_shift[1])

        return sample, multilook, oversample, offset, [lines, pixels]
