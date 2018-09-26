# This function creates a list of processing steps to load for processing. This is done gather and catergorize all
# steps in one scheme. Be sure that you add your step here after creating a new one to allow the use of multiprocessing
# for that step.

# For every new step:
# 1. Load the step at the imports
# 2. Be sure that the new step defines his dependencies and memory use in the processing_info function.
# 3. Run this script and evaluate the resulting processing scheme.

# Import the different steps:

# Individual images > coregistering / geocoding / resampling
from orbit_dem_functions.srtm_download import SrtmDownload
from processing_steps.import_dem import CreateSrtmDem, CreateExternalDem
from processing_steps.radar_dem import RadarDem
from processing_steps.geocode import Geocode
from processing_steps.inverse_geocode import InverseGeocode
from processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
from processing_steps.geometrical_coreg import GeometricalCoreg
from processing_steps.correlation_coreg import CorrelationWindows
from processing_steps.deramping_reramping import Deramp, Reramp
from processing_steps.resample import Resample
from processing_steps.earth_topo_phase import EarthTopoPhase

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

def import_processing_list():

    # Add to dict with processing steps:
    processing = dict()
    processing['srtm_download'] = SrtmDownload
    processing['create_srtm_dem'] = CreateSrtmDem
    processing['create_external_dem'] = CreateExternalDem
    processing['inverse_geocode'] = InverseGeocode
    processing['radar_dem'] = RadarDem
    processing['geocode'] = Geocode
    processing['azimuth_elevation_angle'] = AzimuthElevationAngle
    processing['geometrical_coreg'] = GeometricalCoreg
    processing['correlation_coreg'] = CorrelationWindows
    processing['deramp'] = Deramp
    processing['resample'] = Resample
    processing['reramp'] = Reramp
    processing['earth_topo_phase'] = EarthTopoPhase

    processing['interfero'] = Interfero
    processing['coherence'] = Coherence
    processing['multilook'] = Multilook
    processing['unwrap'] = Unwrap

    processing['concatenate'] = Concatenate
