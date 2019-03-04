# This function creates a list of processing steps to load for processing. This is done gather and catergorize all
# steps in one scheme. Be sure that you add your step here after creating a new one to allow the use of multiprocessing
# for that step.

# For every new step:
# 1. Load the step at the imports
# 2. Be sure that the new step defines his dependencies and memory use in the processing_info function.
# 3. Run this script and evaluate the resulting processing scheme.

import inspect

# Import the different steps:

# Individual images > coregistering / geocoding / resampling
from orbit_dem_functions.srtm_download import SrtmDownload
from processing_steps.import_dem import CreateSrtmDem, CreateExternalDem
from processing_steps.radar_dem import RadarDem
from processing_steps.geocode import Geocode
from processing_steps.inverse_geocode import InverseGeocode
from processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
from processing_steps.geometrical_coreg import GeometricalCoreg
# from processing_steps.correlation_coreg import CorrelationWindows
from processing_steps.deramping_reramping import Deramp, Reramp
from processing_steps.square_amplitude import SquareAmplitude
from processing_steps.amplitude import Amplitude
from processing_steps.conversion_grid import ConversionGrid
from processing_steps.resample import Resample
from processing_steps.earth_topo_phase import EarthTopoPhase
from processing_steps.baseline import Baseline
from processing_steps.height_to_phase import HeightToPhase

from processing_steps.ecmwf_era5_aps import EcmwfEra5Aps
from processing_steps.ecmwf_oper_aps import EcmwfOperAps
from processing_steps.ecmwf_interim_aps import EcmwfInterimAps
from processing_steps.harmonie_aps import HarmonieAps
from processing_steps.harmonie_interferogram import HarmonieInterferogram

# Interferograms > interferogram creation, coherence, filters, corrections.
from processing_steps.interfero import Interfero
from processing_steps.coherence import Coherence
from processing_steps.multilook import Multilook
from processing_steps.unwrap import Unwrap

# Concatenation. This is a special step because it moves from slices to full images and does not belong to a certain
# processing algorithm specifically.
from processing_steps.concatenate import Concatenate

# Slice overlaps > For images in TOPS mode we can get information about overlapping regions.
# Nothing yet...

# Network functions. These functions only work with a network of images.
# No functions yet.

# -> Add your new import statement here

class ProcessingList():

    def __init__(self):

        # Add to dict with processing steps:
        processing = dict()
        processing['srtm_download'] = SrtmDownload
        processing['import_DEM'] = CreateSrtmDem
        processing['create_external_DEM'] = CreateExternalDem
        processing['inverse_geocode'] = InverseGeocode
        processing['radar_DEM'] = RadarDem
        processing['geocode'] = Geocode
        processing['azimuth_elevation_angle'] = AzimuthElevationAngle
        processing['geometrical_coreg'] = GeometricalCoreg

        # processing['correlation_coreg'] = CorrelationWindows
        processing['deramp'] = Deramp
        processing['resample'] = Resample
        processing['reramp'] = Reramp
        processing['earth_topo_phase'] = EarthTopoPhase
        processing['baseline'] = Baseline
        processing['height_to_phase'] = HeightToPhase
        processing['conversion_grid'] = ConversionGrid
        processing['square_amplitude'] = SquareAmplitude
        processing['amplitude'] = Amplitude

        processing['interferogram'] = Interfero
        processing['coherence'] = Coherence
        processing['multilook'] = Multilook
        processing['unwrap'] = Unwrap

        processing['harmonie_aps'] = HarmonieAps
        processing['harmonie_interferogram'] = HarmonieInterferogram
        processing['ecmwf_era5_aps'] = EcmwfEra5Aps
        processing['ecmwf_oper_aps'] = EcmwfOperAps
        processing['ecmwf_interim_aps'] = EcmwfInterimAps

        processing['concatenate'] = Concatenate

        self.processing_steps = []
        self.processing = processing
        self.processing_inputs = dict()
        self.get_function_input()

    def get_function_input(self):

        for key in self.processing.keys():
            self.processing_inputs[key] = dict()

            # Get the functions
            functions = inspect.getmembers(self.processing[key], predicate=inspect.isfunction)
            methods = inspect.getmembers(self.processing[key], predicate=inspect.ismethod)

            # Get the input variable names
            for function in functions:
                self.processing_inputs[key][function[0]] = inspect.getargspec(function[1])[0]
            for method in methods:
                self.processing_inputs[key][method[0]] = inspect.getargspec(method[1])[0]

    def get_function_steps(self):
        # Here we split all the different functions in 3 sub categories (1 coreg master, 2 slave info, 3 ifg info)

        coreg_steps = []
        slave_steps = []
        ifg_steps = []
