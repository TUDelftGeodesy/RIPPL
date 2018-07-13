# This test file generates a simple interferogram based on the developed functions.

from doris_processing.image_metadata import ImageMetadata
from doris_processing.image_data import ImageData
from doris_processing.stack import Stack

# Load processing steps
from doris_processing.processing_steps.geocode import Geocode
from doris_processing.processing_steps.deramping_reramping import Deramp, Reramp
from doris_processing.processing_steps.geometrical_coreg import GeometricalCoreg
from doris_processing.processing_steps.resample import Resample
from doris_processing.processing_steps.earth_topo_phase import EarthTopoPhase
from doris_processing.processing_steps.interfero import Interferogram

# First check whether the data and metadata function.
filename = '/media/gert/Data/radar_datastacks/delft_asc_t088/stack/20170221/swath_2/burst_2/ifgs.res'
type = 'ifgs'
burst_ifgs = ImageData(filename, type)
filename = '/media/gert/Data/radar_datastacks/delft_asc_t088/stack/20170221/swath_2/burst_2/slave.res'
type = 'single'
burst_slave = ImageData(filename, type)
filename = '/media/gert/Data/radar_datastacks/delft_asc_t088/stack/20170221/swath_2/burst_2/master.res'
type = 'single'
burst_master = ImageData(filename, type)

burst_slave.image_disk_to_memory('resample', 1000, 5000, (400, 100))

# Check whether we can load a stack of data.
stack_folder = '/media/gert/Data/radar_datastacks/new_stack/'
dem_database = '/media/gert/Data/DEM/dem_processing'

datastack = Stack(stack_folder)
datastack.read_master_slice_list()
datastack.read_stack()

# Perform the first processing steps for one burst only
date_1 = datastack.image_dates[0]
date_2 = datastack.image_dates[1]
name = datastack.images[date_1].slice_names[0]

# Load master and slave
master = datastack.images[date_1].slices[name]
slave = datastack.images[date_1].slices[name]

Geocode()
