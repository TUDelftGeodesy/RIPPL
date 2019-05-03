from rippl.NWP_simulations.ECMWF.ecmwf_mars_request import MarsRequest


def parallel_download(input):
    data_type = input['data_type']
    dataset_class = input['dataset_class']
    t_list = input['t_list']
    bb_str = input['bb_str']
    grid = input['grid']
    level_list = input['level_list']
    dataset = input['dataset']

    date = input['date']
    target = input['target']

    download = MarsRequest(data_type, dataset_class, t_list, bb_str, grid, level_list, dataset)
    download(date, target)
