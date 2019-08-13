# Function to run parallel package. For further details inspect the pipeline class.
import os
import numpy as np
from rippl.meta_data.image_data import ImageData
import gc


def run_parallel(dat):
    # First split the functions and variables
    functions = dat['function']
    succes = True

    for func in functions:

        if dat['meta']:

            var = dat['meta_var'][0]
            var_names = dat['meta_var_name'][0]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in np.arange(len(var))]
                eval_str = 'func.add_meta_data(' + ','.join(func_str) + ')'
                print(func)
                eval(eval_str)

            dat['meta_var'].pop(0)
            dat['meta_var_name'].pop(0)

        if dat['create']:

            var = dat['create_var'][0]
            var_names = dat['create_var_name'][0]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in np.arange(len(var))]
                eval_str = 'func.create_output_files(' + ','.join(func_str) + ')'
                eval(eval_str)

            dat['create_var'].pop(0)
            dat['create_var_name'].pop(0)

        if dat['proc']:

            var = dat['proc_var'][0]
            var_names = dat['proc_var_name'][0]

            if 'ifg_meta' in var_names:
                meta_name = os.path.dirname(var[var_names.index('ifg_meta')].res_path)
            else:
                meta_name = os.path.dirname(var[var_names.index('meta')].res_path)

            if len(os.path.basename(meta_name)) == 8 or len(os.path.basename(meta_name)) == 17:
                meta_name = 'image ' + os.path.basename(meta_name)
            else:
                meta_name = os.path.basename(meta_name) + ' from rippl.image ' + os.path.basename(os.path.dirname(meta_name))

            if 's_lin' in var_names and 'lines' in var_names:
                s_lin = var[var_names.index('s_lin')]
                lines = var[var_names.index('lines')]
                print('Started processing ' + func.__name__ + ' for ' + meta_name + ' from line ' + str(s_lin) +
                      ' till ' + str(s_lin + lines -1))
            else:
                print('Started processing ' + func.__name__ + ' for ' + meta_name)

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in np.arange(len(var))]
                eval_str = 'func(' + ','.join(func_str) + ')'
                proc_func = eval(eval_str)

                # Run the function created by the eval() string
                succes = proc_func()

            if 's_lin' in var_names and 'lines' in var_names:
                s_lin = var[var_names.index('s_lin')]
                lines = var[var_names.index('lines')]
                print('Finished processing ' + func.__name__ + ' for ' + meta_name + ' from line ' + str(s_lin) +
                      ' till ' + str(s_lin + lines -1))
            else:
                print('Finished processing ' + func.__name__ + ' for ' + meta_name)

            dat['proc_var'].pop(0)
            dat['proc_var_name'].pop(0)

            gc.collect()

            # If
            if not succes:
                break

        if dat['save']:
            var = dat['save_var'][0]
            var_names = dat['save_var_name'][0]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in np.arange(len(var))]
                eval_str = 'func.save_to_disk(' + ','.join(func_str) + ')'
                eval(eval_str)

            dat['save_var'].pop(0)
            dat['save_var_name'].pop(0)

        if dat['clear_mem']:
            var_list = dat['clear_mem_var'][0]
            var_names_list = dat['clear_mem_var_name'][0]

            for var, var_names in zip(var_list, var_names_list):

                if len(var_names) > 0:
                    meta = var[0]

                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in np.arange(1, len(var_names))]
                    eval_str = 'meta.clean_memory(' + ','.join(func_str) + ')'
                    eval(eval_str)

            dat['clear_mem_var'].pop(0)
            dat['clear_mem_var_name'].pop(0)

        # Finally clean the memmap files (can cause memory problems when copying these objects)
        for im in list(dat['res_dat'].keys()):
            for im_type in list(dat['res_dat'][im].keys()):
                if isinstance(dat['res_dat'][im][im_type], ImageData):
                    dat['res_dat'][im][im_type].clean_memmap_files()

    if succes:
        return dat['res_dat']
    else:
        return False
