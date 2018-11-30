import warnings
import os
import collections
import locale
from shapely import geometry
import numpy as np
from coordinate_system import CoordinateSystem
from six import string_types


class ImageMetadata(object):
    # This class hold metadata of a doris datafile and processing chain and is capable of reading from and writing to a
    # .res file used by the doris software.

    def __init__(self, filename='', res_type='', warn=False):
        # Initialize variables

        # Check whether method to read months is set to us
        if locale.getlocale() != ('en_US', 'UTF-8'):
            locale.setlocale(locale.LC_ALL, 'en_US.utf8')

        # Filename of resfile and type (single, interferogram)
        self.res_path = []
        self.res_type = ''
        self.warn = warn

        # Processes, process_control and header of resfile
        self.processes = collections.OrderedDict()
        self.processing_folder = []
        self.orig_data_folder = []
        self.process_data = {}
        self.process_control = {}
        self.process_timestamp = {}
        self.process_time = {}
        self.header = {}

        # Initialize some basic variables. Mainly about geometry and size
        self.polygon = []           # Polygon of image
        self.point = []             # Centerpoint of image
        self.lat_lim = []           # latitude limits
        self.lon_lim = []           # longitude limits
        self.size = []              # Size of image
        self.first_pixel = []       # First pixel in total image
        self.first_line = []        # First line in total image

        #####################################################

        # Create a ResData object (single/interferogram)
        if res_type not in ['single','interferogram'] and not filename:
            warnings.warn('Define if results data is slave, master or interferogram')
            return
        else:
            self.res_type = res_type
        if filename:
            if not os.path.exists(filename):
                if warn:
                    warnings.warn('This filename does not exist: ' + filename)
            else:
                self.res_path = filename
                self.res_read()
        else:
            self.process_control = ImageMetadata.get_process_control(res_type)

    @staticmethod
    def get_process_control(res_type='single'):

        if res_type == 'single':
            process_control = collections.OrderedDict([('readfiles', '0'), ('orbits', '0'), ('crop', '0'),
                                                        ('import_DEM', '0'), ('inverse_geocode', '0'),
                                                        ('radar_DEM', '0'), ('geocode', '0'),
                                                        ('coor_conversion', '0'),
                                                        ('azimuth_elevation_angle', '0'), ('deramp', '0'),
                                                        ('sim_amplitude', '0'), ('coreg_readfiles', '0'),
                                                        ('coreg_orbits', '0'), ('coreg_crop', '0'),
                                                        ('geometrical_coreg', '0'), ('square_amplitude', '0'),
                                                        ('amplitude', '0'),
                                                        ('correl_coreg', '0'), ('combined_coreg', '0'),
                                                        ('master_timing', '0'), ('oversample', '0'),
                                                        ('resample', '0'), ('reramp', '0'), ('height_to_phase', '0'),
                                                        ('earth_topo_phase', '0'), ('filt_azi', '0'), ('baseline', '0'),
                                                        ('filt_range', '0'), ('harmonie_aps', '0'), ('ecmwf_aps', '0'),
                                                        ('structure_function', '0'), ('split_spectrum', '0')])
        elif res_type == 'interferogram':
            process_control = collections.OrderedDict([('coreg_readfiles', '0'), ('coreg_orbits', '0'),
                                                        ('coreg_crop', '0'), ('image_stitch', '0'),
                                                        ('interferogram', '0'),
                                                        ('split_spectrum', '0'), ('ESD', '0'),
                                                        ('correl_tracking', '0'), ('coherence', '0'),
                                                        ('filtphase', '0'), ('unwrap', '0'),
                                                        ('combined_tracking', '0'), ('NWP_phase', '0'),
                                                        ('structure_function', '0')])
        else:
            print('res_type should either be single or interferogram')
            return

        return process_control

    def res_read(self):
        self.meta_reader()
        self.process_reader()
        self.geometry()

    def geometry(self):

        if self.res_type == 'single':
            dat = ''
        elif self.res_type == 'interferogram':
            dat = 'coreg_'
        else:
            return

        if dat + 'readfiles' in self.process_control.keys():
            # Maybe in some files these data does not exist...
            try:
                lines = int(self.processes[dat + 'crop']['crop_lines'])
                pixels = int(self.processes[dat + 'crop']['crop_pixels'])
                lin_min = int(self.processes[dat + 'crop']['crop_first_line'])
                pix_min = int(self.processes[dat + 'crop']['crop_first_pixel'])

                ul = (float(self.processes[dat + 'readfiles']['Scene_ul_corner_latitude']), float(self.processes[dat + 'readfiles']['Scene_ul_corner_longitude']))
                ll = (float(self.processes[dat + 'readfiles']['Scene_ll_corner_latitude']), float(self.processes[dat + 'readfiles']['Scene_ll_corner_longitude']))
                lr = (float(self.processes[dat + 'readfiles']['Scene_lr_corner_latitude']), float(self.processes[dat + 'readfiles']['Scene_lr_corner_longitude']))
                ur = (float(self.processes[dat + 'readfiles']['Scene_ur_corner_latitude']), float(self.processes[dat + 'readfiles']['Scene_ur_corner_longitude']))

                self.polygon = geometry.Polygon([ul, ur, lr, ll])
                self.point = geometry.Point(((ul[0] + ur[0] + lr[0] + ll[0]) / 4, (ul[1] + ur[1] + lr[1] + ll[1]) / 4))
                self.lat_lim = [np.min([ul[0], ur[0], lr[0], ll[0]]), np.max([ul[0], ur[0], lr[0], ll[0]])]
                self.lon_lim = [np.min([ul[1], ur[1], lr[1], ll[1]]), np.max([ul[1], ur[1], lr[1], ll[1]])]
                self.size = (lines, pixels)
                self.first_pixel = pix_min
                self.first_line = lin_min
            except:
                if self.warn:
                    print('Geometry cannot be loaded')

    def read_res_coordinates(self, step, file_types=''):
        # Read the coordinates from a .res file step. This can be used to detect existing input from for example a DEM

        # First check the different file_types
        step_dat = self.processes[step]
        pos_file_types = [dat[:-12] for dat in step_dat.keys() if dat.endswith('_output_file')]

        if not file_types:
            file_types = pos_file_types
        else:
            for file_type in file_types:
                if file_type not in pos_file_types:
                    print('file type ' + file_type + ' does not exist!')
                    file_types.remove(file_type)

        coordinates_list = []

        for file_type in file_types:

            len_type = len(file_type)
            type_info = dict()
            for dat in step_dat.keys():
                if dat.startswith(file_type):
                    type_info[dat[len_type + 1:]] = step_dat[dat]

            dat_coors = CoordinateSystem()

            if 'multilook_azimuth' in type_info:
                multilook = [int(type_info['multilook_azimuth']), int(type_info['multilook_range'])]
                oversample = [int(type_info['oversample_azimuth']), int(type_info['oversample_range'])]
                offset = [int(type_info['offset_azimuth']), int(type_info['offset_range'])]

                dat_coors.create_radar_coordinates(multilook=multilook, oversample=oversample, offset=offset)

            elif 'lat0' in type_info:
                ellipse_type = type_info['ellipse_type']
                lat0 = float(type_info['lat0'])
                lon0 = float(type_info['lon0'])
                dlat = float(type_info['dlat'])
                dlon = float(type_info['dlon'])

                dat_coors.create_geographic(dlat=dlat, dlon=dlon, ellipse_type=ellipse_type, lat0=lat0, lon0=lon0)

            elif 'projection_type' in type_info:
                projection_type = type_info['projection_type']
                ellipse_type = type_info['ellipse_type']
                proj4_str = type_info['proj4_str']
                x0 = float(type_info['x0'])
                y0 = float(type_info['y0'])
                dx = float(type_info['dx'])
                dy = float(type_info['dy'])

                dat_coors.create_projection(dx=dx, dy=dy, x0=x0, y0=y0, proj4_str=proj4_str,
                                            ellipse_type=ellipse_type, projection_type=projection_type)

            dat_coors.shape = [int(type_info['lines']), int(type_info['pixels'])]
            dat_coors.first_line = int(type_info['first_line'])
            dat_coors.first_pixel = int(type_info['first_pixel'])
            meta_name = os.path.basename(os.path.dirname(self.res_path))
            dat_coors.res_path = self.res_path
            if meta_name.startswith('slice'):
                dat_coors.meta_name = meta_name
            else:
                dat_coors.meta_name = 'full'

            if 'readfiles' in self.processes.keys():
                dat_coors.slice = self.processes['readfiles']['slice'] == 'True'
            else:
                dat_coors.slice = self.processes['coreg_readfiles']['slice'] == 'True'
            coordinates_list.append(dat_coors)

        return coordinates_list

    def meta_reader(self):
        # This function
        with open(self.res_path) as resfile:
            splitter = ':'
            temp = collections.OrderedDict()
            row = 0
            for line in resfile:
                try:
                    ## Filter out rubbish
                    if line == '\n':
                        continue
                    elif 'Start_process_control' in line:
                        self.header = temp
                        temp = collections.OrderedDict()
                    elif 'End_process_control' in line:
                        self.process_control = temp
                        break
                    elif splitter in line and line[0] is not '|' and line[0] is not '\t' :
                        # Split line if possible and add to dictionary
                        l_split = line.split(splitter)
                        temp[l_split[0].strip()] = l_split[1].strip()
                    else:
                        name = 'row_' + str(row)
                        row += 1
                        temp[name] = [line]

                except:
                    print('Error occurred at line: ' + line)

    def process_reader(self,processes = ''):
        # This function reads random processes based on standard buildup of processes in res files.
        # leader_datapoints can be one of the processes, although it will not appear in the process_control in a .res file
        # If loc is true, it will only return the locations where different processes start.

        if not processes:
            processes = list(self.process_control.keys())

        processes.append('leader_datapoints')
        process = ''

        with open(self.res_path) as resfile:
            # Start at row zero and with empty list
            temp = collections.OrderedDict()
            row = 0
            line_no = -1
            timestamp = False
            timestamp_line = 0
            for line in resfile:
                try:
                    line_no += 1
                    # Filter out rubbish
                    if '|'in line[0]:
                        continue
                    elif '**' in line:
                        continue
                    elif line == '\n':
                        continue

                    # Check if timestamp
                    if ' *===========' in line:
                        # First line of time stamp
                        temp = collections.OrderedDict()
                        timestamp = True
                        row = 0
                        continue
                    elif ' *-----------' in line:
                        timestamp = False
                        timestamp_data = temp
                        timestamp_line = line_no + 5
                        continue

                    # Check if process
                    if '*' in line[0]:
                        if line.replace('*_Start_', '').split(':')[0].strip() in processes:
                            process = line.replace('*_Start_', '').split(':')[0].strip()
                            temp = collections.OrderedDict()
                            row = 0; space = [0]; space_r = [0,0,0,0,0,0,0,0]

                            # Finally save the timestamp if it exists
                            if line_no == timestamp_line:
                                self.process_timestamp[process] = timestamp_data
                            else:
                                self.process_timestamp[process] = ''

                        elif line.replace('* End_', '').split(':')[0] == process:
                            self.processes[process] = temp
                            temp = collections.OrderedDict()
                            process = ''
                        continue

                    # Save line
                    if timestamp is True:
                        # Save rows in timestamp
                        row_name = 'row_' + str(row)
                        temp[row_name] = line
                        if row == 1:
                            self.process_time[process] = line.split(':', 1)[1].strip()
                        row += 1
                    elif process:
                        # If we are in a process output line
                        # Split line using ':' , '=' or spaces (tables)
                        # variable space and space row define the future spacing in every processing step in a res file.

                        if process == 'coarse_orbits':
                            # Add some code for a strange exception in coarse_orbits
                            if '//' in line:
                                temp[line.split()[0]] = line.split()[1:]
                            else:
                                l_split = line.replace('=',':').split(':')
                                temp[l_split[0].strip()] = l_split[1].strip()

                        elif ':' in line:
                            l_split = line.split(':',1)
                            temp[l_split[0].strip()] = l_split[1].strip()
                        else:
                            # If the line does not contain a : it is likely a table.
                            l_split = line.replace('\t',' ').split()
                            row_name = 'row_' + str(row)
                            temp[row_name] = [l_split[i].strip() for i in np.arange(len(l_split))]
                            row += 1

                except:
                    print('Error occurred at line: ' + line)

    def process_spacing(self,process=''):

        spacing = 0
        table_spacing = [0,0,0,0,0,0,0]

        dat = self.processes[process]

        for key in dat.keys():
            spacing = max(len(key) + 8, spacing)

            if key.startswith('row'):
                n=0
                for val in self.processes[process][key]:
                    table_spacing[n] = max(len(val) + 3, table_spacing[n])
                    n += 1
        spacing = [spacing]

        return spacing, table_spacing

    def del_process(self,process=''):
        # function deletes one or multiple processes from the corresponding res file

        if isinstance(process, string_types): # one process
            if not process in self.process_control.keys():
                warnings.warn('The requested process does not exist! (or processes are not read jet, use self.process_reader): ' + str(process))
                return
        elif isinstance(process, list): # If we use a list
            for proc in process:
                if not proc in self.process_control.keys():
                    warnings.warn('The requested process does not exist! (or processes are not read jet, use self.process_reader): ' + str(proc))
                    return
        else:
            warnings.warn('process should contain either a string of one process or a list of multiple processes: ' + str(process))

        # Now remove the process and write the file again.
        if isinstance(process, string_types): # Only one process should be removed
            self.process_control[process] = '0'
            del self.processes[process]
        else:
            for proc in process:
                self.process_control[proc] = '0'
                del self.processes[proc]

    def write(self, new_filename='', warn=True):
        # Here all the available information acquired is written to a new resfile. Generally if information is manually
        # added or removed and the file should be created or created again. (For example the readfiles for Sentinel 1
        # which are not added yet..)

        if not new_filename and not self.res_path:
            warnings.warn('Please specify filename: ' + str(new_filename))
            return
        elif not new_filename:
            new_filename = self.res_path
        if not self.process_control or not self.processes and warn:
            warnings.warn('Every result file needs at least a process control and one process to make any sense: ' + str(new_filename))

        # Open file and write header, process control and processes
        self.res_path = new_filename
        if not os.path.exists(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))
        f = open(new_filename,"w")

        # Write the header:
        if self.header:
            spacing = [40]
            for key in self.header.keys():
                if 'row' in key:       # If it is just a string
                    f.write(self.header[key][0])
                else:                   # If the key should included
                    f.write((key + ':').ljust(spacing[0]) + self.header[key] + '\n')

        # Write the process control
        for i in np.arange(3):
            f.write('\n')
        f.write('Start_process_control\n')
        for process in self.process_control.keys():
            if process != 'leader_datapoints':  # leader_datapoints is left out in process control
                f.write((process + ':\t\t') + str(self.process_control[process]) + '\n')
        f.write('End_process_control\n')

        # Then loop through all the processes
        for process in [p for p in self.processes.keys()]:
            # First check for a timestamp and add it if needed.
            if self.process_timestamp[process]:
                for i in np.arange(2):
                    f.write('\n')
                f.write('   *====================================================================* \n')
                for key in self.process_timestamp[process].keys():
                    f.write(self.process_timestamp[process][key])
                f.write('   *--------------------------------------------------------------------* \n')

            # Then write the process itself
            if process == 'coarse_orbits':
                spacing = [45]
                spacing_row = [15,10,15]
            else:
                spacing, spacing_row = self.process_spacing(process)
            data = self.processes[process]

            for i in np.arange(3):
                f.write('\n')
            f.write('******************************************************************* \n')
            f.write('*_Start_' + process + ':\n')
            f.write('******************************************************************* \n')

            for line_key in self.processes[process].keys():
                if 'row' in line_key:  # If it is a table of consists of several different parts
                    line = ''.join([(' ' + data[line_key][i]).replace(' -','-').ljust(spacing_row[i]) for i in np.arange(len(data[line_key]))])
                    f.write(line + '\n')
                elif process == 'coarse_orbits':  # the coarse orbits output is different from the others.
                    if 'Control point' in line_key: # Special case coarse orbits...
                        f.write((line_key + ' =').ljust(spacing[0]) + str(self.processes[process][line_key]) + '\n')
                    elif not isinstance(data[line_key], string_types): # Another special case
                        f.write(line_key.ljust(spacing_row[0]) + (data[line_key][0]).ljust(spacing_row[1]) +
                                data[line_key][1].ljust(spacing_row[2]) + ' '.join(data[line_key][2:]) + '\n')
                    elif isinstance(data[line_key], string_types): # Handle as in normal cases
                        f.write((line_key + ':').ljust(spacing[0]) + str(self.processes[process][line_key]) + '\n')
                else: # If it consists out of two parts
                    f.write((line_key + ':').ljust(spacing[0]) + str(self.processes[process][line_key]) + '\n')

            f.write('******************************************************************* \n')
            f.write('* End_' + process + ':_NORMAL\n')
            f.write('******************************************************************* \n')
        f.close()

        # Read the locations in the new file
        self.process_reader()

    def insert(self, data, process, variable=''):
        # This function inserts a variable or a process which does not exist at the moment
        processes = list(self.process_control.keys())
        processes.extend(['header', 'leader_datapoints'])

        if process not in processes:
            warnings.warn('This process does not exist for this datatype: ' + str(process))
            print('We will add it anyway')
            self.process_control[process] = '0'

        # If a full process is added
        if not variable:
            if self.process_control[process] == '1':
                # print('The ' + str(process) + ' process already exists. Data will be updated')
                self.processes[process] = data
            elif self.process_control[process] == '0':
                self.process_control[process] = '1'
                self.processes[process] = data
                self.process_timestamp[process] = ''

        # A variable is added
        if variable:
            if variable in self.processes[process].keys():
                warnings.warn('Variable ' + variable + ' in process ' + process + ' already exists. Metadata will be '
                                                                                  'updated')
            self.processes[process][variable] = data

    def delete(self, process, variable=''):
        # This function deletes a variable or a process which does exist at the moment
        processes = self.process_control.keys()
        processes.extend(['header','leader_datapoints'])

        if process not in processes:
            warnings.warn('This process does not exist for this datatype: ' + str(process))
            return

        # If a full process is deleted
        if not variable:
            if self.process_control[process] == '0':
                warnings.warn('This process does not exist: ' + str(process))
                return
            elif self.process_control[process] == '1':
                self.process_control[process] = '0'
                del self.processes[process]
                del self.process_timestamp[process]

        # A variable is deleted
        if variable:
            if not variable in self.processes[process].keys():
                warnings.warn('This variable does not exist: ' + str(variable))
                return
            else:
                del self.processes[process][variable]
