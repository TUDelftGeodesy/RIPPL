# Function to run parallel package. For further details inspect the pipeline class.
import os


def run_parallel(dat):
    # First split the functions and variables
    functions = dat['function']

    for func, n in zip(functions, range(len(functions))):

        if dat['meta']:

            var = dat['meta_var'][n]
            var_names = dat['meta_var_name'][n]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                eval_str = 'func.add_meta_data(' + ','.join(func_str) + ')'
                exec(eval_str)

        if dat['create']:

            var = dat['create_var'][n]
            var_names = dat['create_var_name'][n]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                eval_str = 'func.create_output_files(' + ','.join(func_str) + ')'
                exec(eval_str)

            # Finally clean the memmap files (can cause memory problems when copying these objects)
            memmap_data_vars = [var[var_names.index(meta_str)] for meta_str in var_names if
                                meta_str in ['meta', 'master_meta', 'cmaster_meta', 'ifg_meta']]
            for meta in memmap_data_vars:
                meta.clean_memmap_files()

        if dat['proc']:

            var = dat['proc_var'][n]
            var_names = dat['proc_var_name'][n]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                eval_str = 'proc_func = func(' + ','.join(func_str) + ')'
                exec(eval_str)

            # Run the function created by the eval() string
            proc_func()

            if 's_lin' in var_names and 'lines' in var_names and 'meta' in var_names:

                meta_name = os.path.dirname(var[var_names.index('meta')].res_path)
                if len(os.path.basename(meta_name)) == 8:
                    meta_name = 'image ' + os.path.basename(meta_name)
                else:
                    meta_name = os.path.basename(meta_name) + ' from image ' + os.path.basename(os.path.dirname(meta_name))
                s_lin = var[var_names.index('s_lin')]
                lines = var[var_names.index('lines')]
                print('Finished processing ' + func.__name__ + ' for ' + meta_name + ' from line ' + str(s_lin) +
                      ' till ' + str(s_lin + lines -1))

            elif 'meta' in var_names:

                meta_name = os.path.dirname(var[var_names.index('meta')].res_path)
                if len(os.path.basename(meta_name)) == 8:
                    meta_name = 'image ' + os.path.basename(meta_name)
                else:
                    meta_name = os.path.basename(meta_name) + ' from image ' + os.path.basename(
                        os.path.dirname(meta_name))
                print('Finished processing ' + func.__name__ + ' for ' + meta_name)

            # Finally clean the memmap files (can cause memory problems when copying these objects)
            memmap_data_vars = [var[var_names.index(meta_str)] for meta_str in var_names if
                                meta_str in ['meta', 'master_meta', 'cmaster_meta', 'ifg_meta']]
            for meta in memmap_data_vars:
                meta.clean_memmap_files()

        if dat['save']:
            var = dat['save_var'][n]
            var_names = dat['save_var_name'][n]

            if len(var_names) > 0:
                # Because the number of variables can vary we use the eval functions.
                func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                eval_str = 'func.save_to_disk(' + ','.join(func_str) + ')'
                exec(eval_str)

            # Finally clean the memmap files (can cause memory problems when copying these objects)
            memmap_data_vars = [var[var_names.index(meta_str)] for meta_str in var_names if
                                meta_str in ['meta', 'master_meta', 'cmaster_meta', 'ifg_meta']]
            for meta in memmap_data_vars:
                meta.clean_memmap_files()

        if dat['clear_mem']:
            var_list = dat['clear_mem_var'][n]
            var_names_list = dat['clear_mem_var_name'][n]

            for var, var_names in zip(var_list, var_names_list):

                if len(var_names) > 0:
                    meta = var[0]

                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(1, len(var_names))]
                    eval_str = 'meta.clean_memory(' + ','.join(func_str) + ')'
                    exec(eval_str)

    return dat['res_dat']
