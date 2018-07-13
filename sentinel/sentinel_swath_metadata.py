# This function performs an xml query on a provided xml file.
from doris_processing.image_data import ImageData

import collections
from lxml import etree
import os
from datetime import datetime
from datetime import timedelta
from shapely import geometry
import numpy as np
import copy


class CreateSwathXmlRes():
    # This function creates a resfile (or all the information needed to create one for a sentinel swath and corresponding
    # bursts. This includes the precise or restituted orbits.

    def __init__(self, swath_xml='', swath_etree=''):
        
        # xml data
        if swath_etree:
            if not isinstance(swath_etree, etree._ElementTree):
                print('swath_etree should be an instance of the etree class!')
                return
            else:
                self.etree = swath_etree
        else:
            self.etree = ''
        
        if isinstance(swath_xml, str) and not swath_etree: 
            if not os.path.exists(swath_xml):
                print('xml file does not exist')
                return
            
        self.swath_xml = swath_xml
        
        # Initialize the xml swath inputs
        self.swath_header = dict()          # Header of resfile
        self.swath_readfiles = dict()       # Dummy readfiles for swath, used as base for the burst readfiles
        self.swath_orbit = dict()           # Swath orbit information
        self.swath_xml_update = dict()
        self.burst_xml_dat = dict()         # Extra data read from xml to get burst information

        # Swath coverage
        self.swath_coverage = []
        self.swath_coors = []

        # Initialize the coverage of swath and bursts
        self.burst_coverage = []
        self.burst_center = []
        self.burst_coors = []
        self.burst_center_coors = []

        # Initialize the different resfile steps
        self.burst_readfiles = []       # The readfile parts for different bursts
        self.burst_crop = []
        self.burst_res_files = []

        # Final results for burst .res files
        self.burst_meta = []        # Final resfiles for burst files.

    def __call__(self, orbit_class):
        # When we call the function perform all steps and create the burst .res files
        self.read_xml()
        self.create_swath_header()
        self.create_swath_orbit(orbit_class)
        self.burst_swath_coverage()
        self.create_burst_readfiles()
        self.create_burst_crop()

        # In the last step we combine header, readfiles, orbit and crop to create burst resfiles
        self.burst_meta = []

        for readfiles, crop, coverage in zip(self.burst_readfiles, self.burst_crop, self.burst_coverage):
            resfile = ImageData('', 'single')
            resfile.insert(readfiles, 'readfiles')
            resfile.insert(self.swath_orbit, 'orbits')
            resfile.insert(crop, 'crop')
            resfile.header = self.swath_header
            resfile.geometry = coverage
            self.burst_meta.append(resfile)

        return self.burst_meta

    def burst_coverage(self):
        # When we are only interested in the coverage of the individual bursts. This is needed when we want to check
        # how different bursts overlap.

        self.read_xml()
        self.burst_swath_coverage()

        return self.burst_center, self.burst_center_coors

    def read_xml(self):

        if not self.etree:
            self.etree = etree.parse(self.swath_xml)

        self.swath_readfiles = collections.OrderedDict([
            ('Volume_file'                                  , 'blank'),
            ('Volume_ID'                                    , './/adsHeader/missionDataTakeId'),
            ('Volume_identifier'                            , 'blank'),
            ('Volume_set_identifier'                        , 'blank'),
            ('Number of records in ref. file'               , 'blank'),
            ('SAR_PROCESSOR'                                , 'needs_initial_processing'),
            ('SWATH'                                        , './/adsHeader/swath'),
            ('PASS'                                         , './/generalAnnotation/productInformation/pass'),
            ('IMAGE_MODE'                                   , './/adsHeader/mode'),
            ('polarisation'                                 , './/adsHeader/polarisation'),
            ('Product type specifier'                       , './/adsHeader/missionId'),
            ('Logical volume generating facility'           , 'blank'),
            ('Location and date/time of product creation'   , 'blank'),
            ('Number_of_lines_Swath'                        , './/imageAnnotation/imageInformation/numberOfLines'),
            ('number_of_pixels_Swath'                       , './/imageAnnotation/imageInformation/numberOfSamples'),
            ('rangePixelSpacing'                            , './/imageAnnotation/imageInformation/rangePixelSpacing'),
            ('azimuthPixelSpacing'                          , './/imageAnnotation/imageInformation/azimuthPixelSpacing'),
            ('total_Burst'                                  , 'needs_initial_processing'),
            ('Burst_number_index'                           , 'burst_specific'),
            ('RADAR_FREQUENCY (HZ)'                         , './/generalAnnotation/productInformation/radarFrequency'),
            ('Scene identification'                         , 'needs_initial_processing'),
            ('Scene location'                               , 'needs_initial_processing'),
            ('Sensor platform mission identifer'            , './/adsHeader/missionId'),
            ('Scene_center_heading'                         , './/generalAnnotation/productInformation/platformHeading'),
            ('Scene_centre_latitude'                        , 'burst_specific'),
            ('Scene_centre_longitude'                       , 'burst_specific'),
            ('Radar_wavelength (m)'                         , 'needs_initial_processing'),
            ('Azimuth_steering_rate (deg/s)'                , './/generalAnnotation/productInformation/azimuthSteeringRate'),
            ('Pulse_Repetition_Frequency_raw_data(TOPSAR)'  , './/generalAnnotation/downlinkInformationList/downlinkInformation/prf'),
            ('First_pixel_azimuth_time (UTC)'               , 'burst_specific'),
            ('Pulse_Repetition_Frequency (computed, Hz)'    , './/imageAnnotation/imageInformation/azimuthFrequency'),
            ('Azimuth_time_interval (s)'                    , './/imageAnnotation/imageInformation/azimuthTimeInterval'),
            ('Total_azimuth_band_width (Hz)'                , './/imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing/totalBandwidth'),
            ('Weighting_azimuth'                           , './/imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing/windowType'),
            ('Range_time_to_first_pixel (2way) (ms)'        , 'needs_initial_processing'),
            ('Range_sampling_rate (computed, MHz)'          , 'needs_initial_processing'),
            ('Total_range_band_width (MHz)'                 , 'needs_initial_processing'),
            ('Weighting_range'                              , './/imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/windowType'),
            ('DC_reference_azimuth_time'                    , 'burst_specific'),
            ('DC_reference_range_time'                      , 'burst_specific'),
            ('Xtrack_f_DC_constant (Hz, early edge)'        , 'burst_specific'),
            ('Xtrack_f_DC_linear (Hz/s, early edge)'        , 'burst_specific'),
            ('Xtrack_f_DC_quadratic (Hz/s/s, early edge)'   , 'burst_specific'),
            ('FM_reference_azimuth_time'                    , 'burst_specific'),
            ('FM_reference_range_time'                      , 'burst_specific'),
            ('FM_polynomial_constant_coeff (Hz, early edge)', 'burst_specific'),
            ('FM_polynomial_linear_coeff (Hz/s, early edge)', 'burst_specific'),
            ('FM_polynomial_quadratic_coeff (Hz/s/s, early edge)', 'burst_specific'),
            ('Datafile'                                     , 'needs_initial_processing'),
            ('Dataformat'                                   , 'burst_specific'),
            ('Number_of_lines_original'                     , 'burst_specific'),
            ('Number_of_pixels_original'                    , 'burst_specific')
        ])

        self.swath_xml_update = collections.OrderedDict([
            ('heading', './/generalAnnotation/productInformation/platformHeading'),
            ('rangeRSR', './/generalAnnotation/productInformation/rangeSamplingRate'),
            ('rangeBW','.//imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/processingBandwidth'),
            ('rangeTimePix', './/imageAnnotation/imageInformation/slantRangeTime'),
            ('orbitABS', './/adsHeader/absoluteOrbitNumber'),
            ('scenePol', './/adsHeader/polarisation'),
            ('sceneMode', './/adsHeader/mode'),
            ('imageLines', './/swathTiming/linesPerBurst'),
            ('imagePixels', './/swathTiming/samplesPerBurst'),
            ('Swath_startTime', './/adsHeader/startTime'),
            ('Swath_stopTime', './/adsHeader/stopTime'),
            ('linesPerBurst', './/swathTiming/linesPerBurst'),
            ('samplesPerBurst', './/swathTiming/samplesPerBurst'),
        ])

        self.burst_xml_dat = collections.OrderedDict([
            ('firstValidSample'                     , './/swathTiming/burstList/burst/firstValidSample'),
            ('lastValidSample'                      , './/swathTiming/burstList/burst/lastValidSample'),
            ('sceneCenLine_number'                  , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/line'),
            ('sceneCenPixel_number'                 , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/pixel'),
            ('sceneCenLat'                          , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/latitude'),
            ('sceneCenLon'                          , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/longitude'),
            ('height'                               , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/height'),
            ('azimuthTime'                          , './/geolocationGrid/geolocationGridPointList/geolocationGridPoint/azimuthTime'),
            ('sceneRecords'                         , './/imageDataInfo/imageRaster/numberOfRows'),
            ('orbitTime'                            , './/generalAnnotation/orbitList/orbit/time'),
            ('orbitX'                               , './/generalAnnotation/orbitList/orbit/position/x'),
            ('orbitY'                               , './/generalAnnotation/orbitList/orbit/position/y'),
            ('orbitZ'                               , './/generalAnnotation/orbitList/orbit/position/z'),
            ('velX'                                 , './/generalAnnotation/orbitList/orbit/velocity/x'),
            ('velY'                                 , './/generalAnnotation/orbitList/orbit/velocity/y'),
            ('velZ'                                 , './/generalAnnotation/orbitList/orbit/velocity/z'),
            ('azimuthTimeStart'                     , './/swathTiming/burstList/burst/azimuthTime'),
            ('doppler_azimuth_Time'                 , './/dopplerCentroid/dcEstimateList/dcEstimate/azimuthTime'),
            ('doppler_range_Time'                   , './/dopplerCentroid/dcEstimateList/dcEstimate/t0'),
            ('dopplerCoeff'                         , './/dopplerCentroid/dcEstimateList/dcEstimate/dataDcPolynomial'),
            ('azimuthFmRate_reference_Azimuth_time' , './/generalAnnotation/azimuthFmRateList/azimuthFmRate/azimuthTime'),
            ('azimuthFmRate_reference_Range_time'   , './/generalAnnotation/azimuthFmRateList/azimuthFmRate/t0'),
            ('azimuthFmRatePolynomial'              , './/generalAnnotation/azimuthFmRateList/azimuthFmRate/azimuthFmRatePolynomial')
        ])

        # Now find the variables of self.swath_readfiles and self.swath_xml_update in the xml data.
        for key in self.swath_readfiles.keys():
            if self.swath_readfiles[key].startswith('.//'):
                self.swath_readfiles[key] = self.etree.find(self.swath_readfiles[key]).text

        for key in self.swath_xml_update.keys():
            if self.swath_xml_update[key].startswith('.//'):
                self.swath_xml_update[key] = self.etree.find(self.swath_xml_update[key]).text

        for key in self.burst_xml_dat.keys():
            if self.burst_xml_dat[key].startswith('.//'):
                self.burst_xml_dat[key] = [n.text for n in self.etree.findall(self.burst_xml_dat[key])]

        dates = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.burst_xml_dat['orbitTime']]
        self.burst_xml_dat['orbitTime'] = [s.second + s.hour * 3600 + s.minute * 60 + s.microsecond / 1000000.0 for s in dates]

        # Finally do some first calculations to get standardized values.
        self.swath_readfiles['SAR_PROCESSOR'] = 'Sentinel-' + self.swath_readfiles['Sensor platform mission identifer'][-2:]
        self.swath_readfiles['total_Burst'] = str(len(self.burst_xml_dat['azimuthTimeStart']))
        self.swath_readfiles['Scene identification'] = 'Orbit: '+ self.swath_xml_update['orbitABS']
        self.swath_readfiles['Scene location'] = 'lat: ' + self.burst_xml_dat['sceneCenLat'][0] + ' lon:' + self.burst_xml_dat['sceneCenLon'][0]
        self.swath_readfiles['Radar_wavelength (m)'] = "{:.9f}".format(299792458.0/float(self.swath_readfiles['RADAR_FREQUENCY (HZ)']))
        self.swath_readfiles['Range_time_to_first_pixel (2way) (ms)'] = "{:.15f}".format(float(self.swath_xml_update['rangeTimePix'])*1000)
        self.swath_readfiles['Range_sampling_rate (computed, MHz)'] = "{:.9f}".format(float(self.swath_xml_update['rangeRSR'])/1000000)
        self.swath_readfiles['Total_range_band_width (MHz)'] = "{:.9f}".format(float(self.swath_xml_update['rangeBW'])/1000000)
        self.swath_readfiles['Datafile'] = os.path.join(os.path.dirname(os.path.dirname(self.swath_xml)),
                                           'measurement', os.path.basename(self.swath_xml)[:-4] + '.tiff')
        self.swath_readfiles['Dataformat'] = 'tiff'
        self.swath_readfiles['Number_of_lines_original'] = self.swath_xml_update['imageLines']
        self.swath_readfiles['Number_of_pixels_original'] = self.swath_xml_update['imagePixels']
        self.swath_readfiles['Number_of_lines_burst'] = self.swath_xml_update['linesPerBurst']
        self.swath_readfiles['Number_of_pixels_burst'] = self.swath_xml_update['samplesPerBurst']
        self.swath_readfiles['First_pixel (w.r.t. tiff_image)'] = '1'
        self.swath_readfiles['Last_pixel (w.r.t. tiff_image)'] = self.swath_xml_update['samplesPerBurst']

    def create_swath_header(self):

        self.swath_header = collections.OrderedDict()

        self.swath_header['row_1'] = ['===============================================\n']
        self.swath_header['MASTER RESULTFILE:'] = ''
        self.swath_header['Created by'] = 'Doris TU Delft'
        self.swath_header['row_2'] = 'Doris (Delft o-o Radar Interferometric Software)'
        self.swath_header['Version'] = 'Version (2015) (For TOPSAR)'
        self.swath_header['FFTW library'] = 'used'
        self.swath_header['VECLIB library'] = 'not used'
        self.swath_header['LAPACK library'] = 'not used'
        self.swath_header['Compiled at'] = 'XXXXXXXX'
        self.swath_header['By GUN gcc'] = 'XXXXXXXX'
        self.swath_header['row_3'] = ['===============================================\n']

    def create_swath_orbit(self, orbit_class):
        # This function utilizes the orbit_read script to read precise orbit files and export them to the resfile format.
        # Additionally it removes the burst_datapoints part, as it is not needed anymore.

        orbit_time = datetime.strptime(self.burst_xml_dat['azimuthTimeStart'][0], '%Y-%m-%dT%H:%M:%S.%f')
        sat = self.swath_readfiles['SAR_PROCESSOR'][-2:]
        orbit_dat = orbit_class.interpolate_orbit(orbit_time, 'POE', satellite=sat)

        # If there are no precise orbit files available switch back to .xml file information
        if orbit_dat == False:
            orbit_dat = self.burst_xml_dat

        self.swath_orbit = collections.OrderedDict()

        self.swath_orbit['row_0'] = ['t(s)', 'X(m)', 'Y(m)', 'Z(m)', 'velX(m)', 'velY(m)', 'velZ(m)']
        t = orbit_dat['orbitTime']
        x = orbit_dat['orbitX']
        y = orbit_dat['orbitY']
        z = orbit_dat['orbitZ']
        vel_x = orbit_dat['velX']
        vel_y = orbit_dat['velY']
        vel_z = orbit_dat['velZ']
        self.swath_orbit['NUMBER_OF_DATAPOINTS'] = str(len(t))

        # Save the rows
        for n in range(len(t)):
            self.swath_orbit['row_' + str(n + 1)] = ["{:.6f}".format(t[n]),
                                                     "{:.7f}".format(float(x[n])),
                                                     "{:.7f}".format(float(y[n])),
                                                     "{:.7f}".format(float(z[n])),
                                                     "{:.7f}".format(float(vel_x[n])),
                                                     "{:.7f}".format(float(vel_y[n])),
                                                     "{:.7f}".format(float(vel_z[n]))]

    def burst_swath_coverage(self):
        # This function returns the lat, lon of the corners of all bursts in this swath. If polygon is True also the poly
        # gons are generated.

        # Now calculate the centre pixels of individual bursts.
        line_nums = np.array([int(n) for n in self.burst_xml_dat['sceneCenLine_number']])
        size = (len(np.unique(line_nums)), line_nums.size / len(np.unique(line_nums)))
        line_nums = np.reshape(line_nums, size)

        lat = np.reshape([float(n) for n in self.burst_xml_dat['sceneCenLat']], size)
        lon = np.reshape([float(n) for n in self.burst_xml_dat['sceneCenLon']], size)
        az_times = np.reshape([datetime.strptime(n,'%Y-%m-%dT%H:%M:%S.%f') for n in self.burst_xml_dat['azimuthTime']], size)
        az_steps = timedelta(seconds = float(self.swath_readfiles['Azimuth_time_interval (s)']))
        az_start_time = az_times[0, :] - az_steps * line_nums[0, :]

        # Calculate line numbers
        az_diff = (az_times - az_start_time[None, :])
        line_nums = np.reshape(np.round(np.array([n.seconds + n.microseconds/1000000.0 for n in np.ravel(az_diff)]) \
                       / (az_steps.microseconds / 1000000.0)), size).astype(np.int32)

        self.burst_coors = []
        self.burst_coverage = []
        self.burst_center = []
        self.burst_center_coors = []

        # Now calculate the polygons for the different bursts
        for n in range(size[0] - 1):
            self.burst_coors.append([[lon[n, 0], lat[n, 0]],          [lon[n, -1], lat[n, -1]],
                                     [lon[n+1, -1], lat[n+1, -1]],    [lon[n+1, 0], lat[n+1, 0]]])
            self.burst_coverage.append(geometry.Polygon(self.burst_coors[n]))

            self.burst_center_coors.append([(lon[n, 0] + lon[n+1, 0] + lon[n, -1] + lon[n+1, -1]) / 4,
                                            (lat[n, 0] + lat[n+1, 0] + lat[n, -1] + lat[n+1, -1]) / 4])
            self.burst_center.append(geometry.Point(self.burst_center_coors[n]))

        self.swath_coors = [[lon[0, 0], lat[0, 0]],    [lon[0, -1], lat[0, -1]],
                           [lon[-1, -1], lat[-1, -1]], [lon[-1, 0], lat[-1, 0]]]
        self.swath_coverage = geometry.Polygon(self.swath_coors)

    def create_burst_readfiles(self):
        # First copy swath metadata for burst and create a georef dict which stores information about the geo reference of
        # the burst.

        self.burst_readfiles = []

        # Time steps for different parameters
        doppler_times = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                    self.burst_xml_dat['doppler_azimuth_Time']])
        frequency_times = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                      self.burst_xml_dat['azimuthFmRate_reference_Azimuth_time']])
        burst_start_time = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                       self.burst_xml_dat['azimuthTimeStart']])

        for n in range(int(self.swath_readfiles['total_Burst'])):

            readfiles = copy.deepcopy(self.swath_readfiles)

            readfiles['Burst_number_index'] = str(n + 1)

            # First find coordinates of center and optionally the corners
            readfiles['Scene_centre_longitude'] = str(self.burst_center_coors[n][0])
            readfiles['Scene_centre_latitude'] = str(self.burst_center_coors[n][1])
            readfiles['Scene_ul_corner_latitude'] = str(self.burst_coors[n][0][1])
            readfiles['Scene_ur_corner_latitude'] = str(self.burst_coors[n][1][1])
            readfiles['Scene_lr_corner_latitude'] = str(self.burst_coors[n][2][1])
            readfiles['Scene_ll_corner_latitude'] = str(self.burst_coors[n][3][1])
            readfiles['Scene_ul_corner_longitude'] = str(self.burst_coors[n][0][0])
            readfiles['Scene_ur_corner_longitude'] = str(self.burst_coors[n][1][0])
            readfiles['Scene_lr_corner_longitude'] = str(self.burst_coors[n][2][0])
            readfiles['Scene_ll_corner_longitude'] = str(self.burst_coors[n][3][0])

            # Find doppler centroid frequency and azimuth reference time
            readfiles['First_pixel_azimuth_time (UTC)'] = burst_start_time[n].strftime('%Y-%m-%dT%H:%M:%S.%f')

            # First index after start burst for doppler and azimuth
            doppler_id = np.where(doppler_times > burst_start_time[n])[0][0]
            frequency_id = np.where(frequency_times > burst_start_time[n])[0][0]

            # Assign DC values to metadata
            parameter = self.burst_xml_dat['dopplerCoeff'][doppler_id].split()
            readfiles['DC_reference_azimuth_time'] = doppler_times[doppler_id].strftime('%Y-%m-%dT%H:%M:%S.%f')
            readfiles['DC_reference_range_time'] = self.burst_xml_dat['doppler_range_Time'][doppler_id]
            readfiles['Xtrack_f_DC_constant (Hz, early edge)'] = parameter[0]
            readfiles['Xtrack_f_DC_linear (Hz/s, early edge)'] = parameter[1]
            readfiles['Xtrack_f_DC_quadratic (Hz/s/s, early edge)'] = parameter[2]

            # Assign FM values to metadata
            parameter = self.burst_xml_dat['azimuthFmRatePolynomial'][frequency_id].split()
            readfiles['FM_reference_azimuth_time'] = frequency_times[frequency_id].strftime('%Y-%m-%dT%H:%M:%S.%f')
            readfiles['FM_reference_range_time'] = self.burst_xml_dat['azimuthFmRate_reference_Range_time'][frequency_id]
            readfiles['FM_polynomial_constant_coeff (Hz, early edge)'] = parameter[0]
            readfiles['FM_polynomial_linear_coeff (Hz/s, early edge)'] = parameter[1]
            readfiles['FM_polynomial_quadratic_coeff (Hz/s/s, early edge)'] = parameter[2]

            # Line coordinates in tiff file
            burst_lines = int(self.swath_readfiles['Number_of_lines_burst'])
            readfiles['First_line (w.r.t. tiff_image)'] = str(1 + n * burst_lines)
            readfiles['Last_line (w.r.t. tiff_image)'] = str((n + 1) * burst_lines)

            self.burst_readfiles.append(readfiles)

    def create_burst_crop(self):
        # This function returns a description of the crop files part, which defines how the burst is cropped out of the
        # swath data. This function is generally called when the .res and raw data is written to the datastack and uses one
        # of the outputs from the burst_readfiles

        self.burst_crop = []

        for n in range(int(self.swath_readfiles['total_Burst'])):
            # Line and pixel coordinates in .tiff file (We do the crop directly here to prevent confusion)
            last_sample = np.array([int(x) for x in self.burst_xml_dat['lastValidSample'][n].split()])
            first_sample = np.array([int(x) for x in self.burst_xml_dat['firstValidSample'][n].split()])

            first_line = np.argmax(np.diff(np.array(first_sample))) + 1
            last_line = np.argmin(np.diff(np.array(first_sample)))
            first_pixel = np.min(first_sample[first_sample != -1])
            last_pixel = np.min(last_sample[first_sample != -1])

            crop = collections.OrderedDict()
            crop['Data_output_file'] = 'crop.raw'
            crop['Data_output_format'] = 'complex_int'

            start_line = int(self.burst_readfiles[n]['First_line (w.r.t. tiff_image)'])

            crop['Data_lines'] = str(last_line - first_line + 1)
            crop['Data_pixels'] = str(last_pixel - first_pixel + 1)
            crop['Data_first_pixel'] = str(first_pixel + 1)
            crop['Data_first_line'] = str(first_line + 1)
            crop['Data_first_line (w.r.t. tiff_image)'] = str(start_line + first_line)
            crop['Data_last_line (w.r.t. tiff_image)'] = str(start_line + last_line)

            self.burst_crop.append(crop)
