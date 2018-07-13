# The main goal of this function is to connect different functions to create a pipeline.
# The pipeline code will follow the following steps:
#   - List all needed input and output of the requested processing_steps
#   - Order the different processing_steps in such a way that we will never miss one of the outputs
#   - Create a list of inputs (data on disk), intermediate results (in memory) and outputs (data on disk)
#   - From these list of inputs and outputs we define when they should be loaded in memory
#   Final result of the former will be a list with loading data, processing data, unloading data (either removal or saving to disk)
#   - Results can be altered to create output of certain intermediate steps if needed
#
# If the processing_steps do not fit together the pipeline creator will throw an error
# If the whole process itself is ok the input image is checked whether the needed input files are there. If not the
# function will throw an error.
#
# The final created function

# image meta data
from image_data import ImageData






class create_pipeline(ImageData):


    def __init__(self, functions, ):


    def define_function_order(self):


    def link_input_image(self):


    def load_processing_functions(self):





