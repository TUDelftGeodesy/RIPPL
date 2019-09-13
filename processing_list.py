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
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.processing_steps_old.import_dem import CreateSrtmDem, CreateExternalDem
from rippl.processing_steps_old.radar_dem import RadarDem
from rippl.processing_steps_old.geocode import Geocode
from rippl.processing_steps_old.inverse_geocode import InverseGeocode
from rippl.processing_steps_old.azimuth_elevation_angle import AzimuthElevationAngle
from rippl.processing_steps_old.geometrical_coreg import GeometricalCoreg
# from rippl.processing_steps_old.correlation_coreg import CorrelationWindows
from rippl.processing_steps_old.deramping_reramping import Deramp, Reramp
from rippl.processing_steps_old.square_amplitude import SquareAmplitude
from rippl.processing_steps_old.amplitude import Amplitude
from rippl.processing_steps_old.conversion_grid import ConversionGrid
from rippl.processing_steps_old.resample import Resample
from rippl.processing_steps_old.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps_old.baseline import Baseline
from rippl.processing_steps_old.height_to_phase import HeightToPhase

from rippl.processing_steps_old.ecmwf_era5_aps import EcmwfEra5Aps
from rippl.processing_steps_old.ecmwf_oper_aps import EcmwfOperAps
from rippl.processing_steps_old.harmonie_aps import HarmonieAps
from rippl.processing_steps_old.harmonie_interferogram import HarmonieInterferogram

# Geocoding, dem generation of geographic/projected grids
from rippl.processing_steps_old.coor_geocode import CoorGeocode
from rippl.processing_steps_old.coor_dem import CoorDem
from rippl.processing_steps_old.projection_coor import ProjectionCoor

# Sparsing and masking
from rippl.processing_steps_old.create_point_data import CreatePointData
from rippl.processing_steps_old.create_mask import CreateMask
from rippl.processing_steps_old.sparse_grid import SparseGrid
from rippl.processing_steps_old.mask_grid import MaskGrid

# Interferograms > interferogram creation, coherence, filters, corrections.
from rippl.processing_steps_old.interfero import Interfero
from rippl.processing_steps_old.coherence import Coherence
from rippl.processing_steps_old.multilook import Multilook
from rippl.processing_steps_old.unwrap import Unwrap

# Concatenation. This is a special step because it moves from slices to full images and does not belong to a certain
# processing algorithm specifically.
from rippl.processing_steps_old.concatenate import Concatenate

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
        processing['import_dem'] = CreateSrtmDem
        processing['create_external_dem'] = CreateExternalDem
        processing['inverse_geocode'] = InverseGeocode
        processing['radar_dem'] = RadarDem
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

        processing['projection_coor'] = ProjectionCoor
        processing['coor_dem'] = CoorDem
        processing['coor_geocode'] = CoorGeocode

        processing['sparse_grid'] = SparseGrid
        processing['mask_grid'] = MaskGrid
        processing['point_data'] = CreatePointData
        processing['create_mask'] = CreateMask

        processing['interferogram'] = Interfero
        processing['coherence'] = Coherence
        processing['multilook'] = Multilook
        processing['unwrap'] = Unwrap

        processing['harmonie_aps'] = HarmonieAps
        processing['harmonie_interferogram'] = HarmonieInterferogram
        processing['ecmwf_era5_aps'] = EcmwfEra5Aps
        processing['ecmwf_oper_aps'] = EcmwfOperAps

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
