# This test file generates a simple interferogram based on the developed functions.

from image_metadata import ImageMetadata
from image_data import ImageData
from stack import Stack

# Load processing steps
from processing_steps.geocode import Geocode
from processing_steps.deramping_reramping import Deramp, Reramp
from processing_steps.geometrical_coreg import GeometricalCoreg
from processing_steps.resample import Resample
from processing_steps.earth_topo_phase import EarthTopoPhase
from processing_steps.interfero import Interfero

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
dem_database = '/media/gert/Data/DEM/DEM_processing'

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
