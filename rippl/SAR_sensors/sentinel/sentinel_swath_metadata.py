# This function performs an xml query on a provided xml file.
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

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
        self.swath_readfiles = dict()       # Dummy readfile.py for swath, used as base for the burst readfile.py
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

    def __call__(self, orbit_class, adjust_date=False):
        # When we call the function perform all steps and create the burst .res files
        self.read_xml()
        self.burst_swath_coverage()
        self.create_burst_readfiles(adjust_date=adjust_date)
        self.create_swath_orbit(orbit_class, adjust_date=adjust_date)
        self.create_burst_crop()

        # In the last step we combine header, readfile.py, orbit and crop to create burst resfiles
        self.burst_meta = []

        for readfiles, crop, coverage in zip(self.burst_readfiles, self.burst_crop, self.burst_coverage):
            resfile = ImageProcessingData('')
            resfile.meta.add_readfile(readfiles)
            resfile.meta.add_orbit(self.swath_orbit)
            resfile.meta.create_header()
            resfile.add_process(crop)
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
            ('SAR_processor'                                , 'needs_initial_processing'),
            ('Swath'                                        , './/adsHeader/swath'),
            ('Pass'                                         , './/generalAnnotation/productInformation/pass'),
            ('Image_mode'                                   , './/adsHeader/mode'),
            ('Polarisation'                                 , './/adsHeader/polarisation'),
            ('Product type specifier'                       , './/adsHeader/missionId'),
            ('Number_of_lines_swath'                        , './/imageAnnotation/imageInformation/numberOfLines'),
            ('Number_of_pixels_swath'                       , './/imageAnnotation/imageInformation/numberOfSamples'),
            ('Range_pixel_spacing'                          , './/imageAnnotation/imageInformation/rangePixelSpacing'),
            ('Azimuth_pixel_spacing'                        , './/imageAnnotation/imageInformation/azimuthPixelSpacing'),
            ('Number_of_bursts'                             , 'needs_initial_processing'),
            ('Burst_number_index'                           , 'burst_specific'),
            ('Radar_frequency (Hz)'                         , './/generalAnnotation/productInformation/radarFrequency'),
            ('Scene_identification'                         , 'needs_initial_processing'),
            ('Sensor_platform_mission_identifer'            , './/adsHeader/missionId'),
            ('Scene_center_heading'                         , './/generalAnnotation/productInformation/platformHeading'),
            ('Scene_center_latitude'                        , 'burst_specific'),
            ('Scene_center_longitude'                       , 'burst_specific'),
            ('Radar_wavelength (m)'                         , 'needs_initial_processing'),
            ('Azimuth_steering_rate (deg/s)'                , './/generalAnnotation/productInformation/azimuthSteeringRate'),
            ('Pulse_repetition_frequency_raw_data (TOPSAR)' , './/generalAnnotation/downlinkInformationList/downlinkInformation/prf'),
            ('Orig_first_pixel_azimuth_time (UTC)'          , 'burst_specific'),
            ('First_pixel_azimuth_time (UTC)'               , 'burst_specific'),
            ('First_pixel_azimuth_time (UTC, TOPSAR)'       , 'burst_specific'),
            ('Pulse_repetition_frequency (computed, Hz)'    , './/imageAnnotation/imageInformation/azimuthFrequency'),
            ('Azimuth_time_interval (s)'                    , './/imageAnnotation/imageInformation/azimuthTimeInterval'),
            ('Total_azimuth_band_width (Hz)'                , './/imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing/totalBandwidth'),
            ('Weighting_azimuth'                            , './/imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing/windowType'),
            ('Orig_range_time_to_first_pixel (2way) (ms)'   , 'needs_initial_processing'),
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
            ('Dataformat'                                   , 'burst_specific')
        ])

        self.swath_xml_update = collections.OrderedDict([
            ('Heading', './/generalAnnotation/productInformation/platformHeading'),
            ('Range_sampling_rate', './/generalAnnotation/productInformation/rangeSamplingRate'),
            ('Range_bandwidth','.//imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/processingBandwidth'),
            ('Slant_range_time', './/imageAnnotation/imageInformation/slantRangeTime'),
            ('Orbit_number', './/adsHeader/absoluteOrbitNumber'),
            ('Polarisation', './/adsHeader/polarisation'),
            ('Acquisition_mode', './/adsHeader/mode'),
            ('Image_lines', './/swathTiming/linesPerBurst'),
            ('Image_pixels', './/swathTiming/samplesPerBurst'),
            ('Swath_start_time', './/adsHeader/startTime'),
            ('Swath_stop_time', './/adsHeader/stopTime'),
            ('Lines_per_burst', './/swathTiming/linesPerBurst'),
            ('Samples_per_burst', './/swathTiming/samplesPerBurst'),
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
            ('azimuthTimeStart (TOPSAR)'            , './/swathTiming/burstList/burst/sensingTime'),
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

        for key in ['Number_of_lines_swath', 'Number_of_pixels_swath']:
            self.swath_readfiles[key] = int(self.swath_readfiles[key])
        for key in ['Range_pixel_spacing', 'Azimuth_pixel_spacing', 'Radar_frequency (Hz)',
                    'Pulse_repetition_frequency_raw_data (TOPSAR)', 'Pulse_repetition_frequency (computed, Hz)',
                    'Total_azimuth_band_width (Hz)', 'Azimuth_time_interval (s)', 'Azimuth_steering_rate (deg/s)',
                    'Scene_center_heading']:
            self.swath_readfiles[key] = float(self.swath_readfiles[key])

        dates = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in self.burst_xml_dat['orbitTime']]
        self.burst_xml_dat['orbitTime'] = [s.second + s.hour * 3600 + s.minute * 60 + s.microsecond / 1000000.0 for s in dates]

        # Finally do some first calculations to get standardized values.
        self.swath_readfiles['Swath'] = int(self.swath_readfiles['Swath'][-1])
        self.swath_readfiles['SAR_processor'] = 'Sentinel-' + self.swath_readfiles['Sensor_platform_mission_identifer'][-2:]
        self.swath_readfiles['Number_of_bursts'] = len(self.burst_xml_dat['azimuthTimeStart'])
        self.swath_readfiles['Scene_identification'] = 'Orbit: '+ self.swath_xml_update['Orbit_number']
        self.swath_readfiles['Radar_wavelength (m)'] = 299792458.0/float(self.swath_readfiles['Radar_frequency (Hz)'])
        self.swath_readfiles['Range_time_to_first_pixel (2way) (ms)'] = 'None'
        self.swath_readfiles['Orig_range_time_to_first_pixel (2way) (ms)'] = float(self.swath_xml_update['Slant_range_time']) * 1000
        self.swath_readfiles['Range_sampling_rate (computed, MHz)'] = float(self.swath_xml_update['Range_sampling_rate'])/1000000
        self.swath_readfiles['Total_range_band_width (MHz)'] = float(self.swath_xml_update['Range_bandwidth'])/1000000
        self.swath_readfiles['Datafile'] = os.path.join(os.path.dirname(os.path.dirname(self.swath_xml)),
                                           'measurement', os.path.basename(self.swath_xml)[:-4] + '.tiff')
        self.swath_readfiles['Dataformat'] = 'tiff'
        self.swath_readfiles['Number_of_lines'] = int(self.swath_xml_update['Lines_per_burst'])
        self.swath_readfiles['Number_of_pixels'] = int(self.swath_xml_update['Samples_per_burst'])
        self.swath_readfiles['First_pixel (w.r.t. tiff_image)'] = 0
        self.swath_readfiles['Last_pixel (w.r.t. tiff_image)'] = int(self.swath_xml_update['Samples_per_burst']) - 1

    def create_swath_orbit(self, orbit_class, adjust_date=False):
        # This function utilizes the orbit_read script to read precise orbit files and export them to the resfile format.
        # Additionally it removes the burst_datapoints part, as it is not needed anymore.

        orbit_time = datetime.strptime(self.burst_xml_dat['azimuthTimeStart'][0], '%Y-%m-%dT%H:%M:%S.%f')
        sat = self.swath_readfiles['SAR_processor'][-2:]

        orbit_type = 'precise'
        orbit_dat = orbit_class.interpolate_orbit(orbit_time, orbit_type, satellite=sat, adjust_date=adjust_date)
        if orbit_dat == False:
            orbit_type = 'restituted'
            orbit_dat = orbit_class.interpolate_orbit(orbit_time, orbit_type, satellite=sat, adjust_date=adjust_date)
        # If there are no precise orbit files available switch back to .xml file information
        if orbit_dat == False:
            orbit_type = 'xml_data'
            orbit_dat = self.burst_xml_dat
            if isinstance(adjust_date, datetime):
                orbit_dat.t = [(datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') - adjust_date).total_seconds() for d in orbit_dat.t]
            else:
                orbit_dat.t = [(datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') - datetime.strptime(d[:10], '%Y-%m-%d')).total_seconds() for d in orbit_dat.t]

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

        # Create orbit object
        self.swath_orbit = Orbit()
        self.swath_orbit.create_orbit(t, x, y, z, vel_x, vel_y, vel_z, satellite=self.swath_readfiles['SAR_processor'],
                                      type=orbit_type,
                                      date=self.burst_readfiles[0].json_dict['Orig_first_pixel_azimuth_time (UTC)'][:10])

    def burst_swath_coverage(self):
        # This function returns the lat, lon of the corners of all bursts in this swath. If polygon is True also the poly
        # gons are generated.

        # Now calculate the center pixels of individual bursts.
        line_nums = np.array([int(n) for n in self.burst_xml_dat['sceneCenLine_number']])
        size = (len(np.unique(line_nums)), line_nums.size // len(np.unique(line_nums)))

        lat = np.reshape([float(n) for n in self.burst_xml_dat['sceneCenLat']], size)
        lon = np.reshape([float(n) for n in self.burst_xml_dat['sceneCenLon']], size)

        self.burst_coors = []
        self.burst_coverage = []
        self.burst_center = []
        self.burst_center_coors = []

        # Now calculate the polygons for the different bursts
        for n in np.arange(size[0] - 1):
            self.burst_coors.append([[lon[n, 0], lat[n, 0]],          [lon[n, -1], lat[n, -1]],
                                     [lon[n+1, -1], lat[n+1, -1]],    [lon[n+1, 0], lat[n+1, 0]]])
            self.burst_coverage.append(geometry.Polygon(self.burst_coors[n]))

            self.burst_center_coors.append([(lon[n, 0] + lon[n+1, 0] + lon[n, -1] + lon[n+1, -1]) / 4,
                                            (lat[n, 0] + lat[n+1, 0] + lat[n, -1] + lat[n+1, -1]) / 4])
            self.burst_center.append(geometry.Point(self.burst_center_coors[n]))

        self.swath_coors = [[lon[0, 0], lat[0, 0]],    [lon[0, -1], lat[0, -1]],
                           [lon[-1, -1], lat[-1, -1]], [lon[-1, 0], lat[-1, 0]]]
        self.swath_coverage = geometry.Polygon(self.swath_coors)

    def create_burst_readfiles(self, adjust_date=False):
        # First copy swath meta_data for burst and create a georef dict which stores information about the geo reference of
        # the burst.

        self.burst_readfiles = []

        # Time steps for different parameters
        doppler_times = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                    self.burst_xml_dat['doppler_azimuth_Time']])
        frequency_times = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                      self.burst_xml_dat['azimuthFmRate_reference_Azimuth_time']])
        burst_start_time = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                       self.burst_xml_dat['azimuthTimeStart']])
        burst_start_time_topsar = np.asarray([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in
                                       self.burst_xml_dat['azimuthTimeStart (TOPSAR)']])

        for n in np.arange(self.swath_readfiles['Number_of_bursts']):

            readfiles = copy.deepcopy(self.swath_readfiles)

            readfiles['Burst_number_index'] = int(n + 1)
            readfiles['Slice'] = 'True'

            # Line coordinates in tiff file
            burst_lines = int(self.swath_readfiles['Number_of_lines'])
            readfiles['First_line (w.r.t. tiff_image)'] = int(n * burst_lines)
            readfiles['Last_line (w.r.t. tiff_image)'] = int((n + 1) * burst_lines - 1)
            readfiles['Orig_first_line'] = 0
            readfiles['Orig_first_pixel'] = 0
            readfiles['First_line'] = 0
            readfiles['First_pixel'] = 0

            # First find coordinates of center and optionally the corners
            readfiles['Scene_center_longitude'] = float(self.burst_center_coors[n][0])
            readfiles['Scene_center_latitude'] = float(self.burst_center_coors[n][1])
            readfiles['Scene_center_line'] = int(self.swath_readfiles['Number_of_lines'] / 2)
            readfiles['Scene_center_pixel'] = int(self.swath_readfiles['Number_of_pixels'] / 2)
            readfiles['Scene_ul_corner_latitude'] = float(self.burst_coors[n][0][1])
            readfiles['Scene_ur_corner_latitude'] = float(self.burst_coors[n][1][1])
            readfiles['Scene_lr_corner_latitude'] = float(self.burst_coors[n][2][1])
            readfiles['Scene_ll_corner_latitude'] = float(self.burst_coors[n][3][1])
            readfiles['Scene_ul_corner_longitude'] = float(self.burst_coors[n][0][0])
            readfiles['Scene_ur_corner_longitude'] = float(self.burst_coors[n][1][0])
            readfiles['Scene_lr_corner_longitude'] = float(self.burst_coors[n][2][0])
            readfiles['Scene_ll_corner_longitude'] = float(self.burst_coors[n][3][0])

            # Find doppler centroid frequency and azimuth reference time
            readfiles['First_pixel_azimuth_time (UTC, TOPSAR)'] = burst_start_time_topsar[n].strftime('%Y-%m-%dT%H:%M:%S.%f')
            readfiles['First_pixel_azimuth_time (UTC)'] = 'None'
            readfiles['Orig_first_pixel_azimuth_time (UTC)'] = burst_start_time[n].strftime('%Y-%m-%dT%H:%M:%S.%f')

            # First index after start burst for doppler and azimuth
            doppler_id = np.where(doppler_times > burst_start_time[n])[0][0]
            frequency_id = np.where(frequency_times > burst_start_time[n])[0][0]

            # Assign DC values to meta_data
            parameter = self.burst_xml_dat['dopplerCoeff'][doppler_id].split()
            readfiles['DC_reference_azimuth_time'] = doppler_times[doppler_id].strftime('%Y-%m-%dT%H:%M:%S.%f')
            readfiles['DC_reference_range_time'] = float(self.burst_xml_dat['doppler_range_Time'][doppler_id])
            readfiles['Xtrack_f_DC_constant (Hz, early edge)'] = float(parameter[0])
            readfiles['Xtrack_f_DC_linear (Hz/s, early edge)'] = float(parameter[1])
            readfiles['Xtrack_f_DC_quadratic (Hz/s/s, early edge)'] = float(parameter[2])

            # Assign FM values to meta_data
            parameter = self.burst_xml_dat['azimuthFmRatePolynomial'][frequency_id].split()
            readfiles['FM_reference_azimuth_time'] = frequency_times[frequency_id].strftime('%Y-%m-%dT%H:%M:%S.%f')
            readfiles['FM_reference_range_time'] = float(self.burst_xml_dat['azimuthFmRate_reference_Range_time'][frequency_id])
            readfiles['FM_polynomial_constant_coeff (Hz, early edge)'] = float(parameter[0])
            readfiles['FM_polynomial_linear_coeff (Hz/s, early edge)'] = float(parameter[1])
            readfiles['FM_polynomial_quadratic_coeff (Hz/s/s, early edge)'] = float(parameter[2])

            readfile_burst = Readfile(json_data=readfiles, adjust_date=adjust_date)

            self.burst_readfiles.append(readfile_burst)

    def create_burst_crop(self):
        # This function returns a description of the crop files part, which defines how the burst is cropped out of the
        # swath data. This function is generally called when the .res and raw data is written to the datastack and uses one
        # of the outputs from the burst_readfiles

        self.burst_crop = []

        for n in np.arange(int(self.swath_readfiles['Number_of_bursts'])):
            # Line and pixel coordinates in .tiff file (We do the crop directly here to prevent confusion)
            last_sample = np.array([int(x) for x in self.burst_xml_dat['lastValidSample'][n].split()])
            first_sample = np.array([int(x) for x in self.burst_xml_dat['firstValidSample'][n].split()])

            first_line = np.argmax(np.diff(np.array(first_sample))) + 1
            last_line = np.argmin(np.diff(np.array(first_sample)))
            first_pixel = np.min(first_sample[first_sample != -1]) - 1
            last_pixel = np.min(last_sample[first_sample != -1]) - 1

            coordinates = CoordinateSystem()
            coordinates.load_readfile(self.burst_readfiles[n])
            coordinates.create_radar_coordinates([1, 1], [1, 1])
            coordinates.slice = True
            coordinates.first_pixel = first_pixel
            coordinates.first_line = first_line
            coordinates.shape = [last_line - first_line, last_pixel - first_pixel]
            crop = ProcessData('', 'crop', coordinates, polarisation=self.swath_readfiles['Polarisation'])
            crop.add_process_image('crop', 'complex_int', coordinates.shape)

            self.burst_crop.append(crop)
