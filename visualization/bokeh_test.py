import numpy as np
from bokeh.plotting import figure
from bokeh.models.widgets import Select, RangeSlider
from bokeh.models import LinearColorMapper
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
import copy

import sys
sys.path.extend(['/home/gert/Software/Gert'])

from stack import Stack

# Read stack
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

start_date = '2017-11-02'
end_date = '2017-11-20'
master_date = '2017-11-09'

stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037'

# Nothing will be loaded to memory! Only using memmaps here.
max_pixel_num = 2000000     # The maximum number of pixels an image can hold.

stack = Stack(stack_folder)
stack.read_master_slice_list()
stack.read_stack(start_date, end_date)

# Process stack
image_keys = stack.images.keys()
ifg_keys = stack.interferograms.keys()
possible_dates = list(set(image_keys) | set([ifg[:8] for ifg in ifg_keys]) | set([ifg[9:] for ifg in ifg_keys]))

# Helper variable for updates dropdown.
#source = ColumnDataSource(data=dict(entry=[],value=[]))
#callbacks = CustomJS(args=dict(source=source), code="""
#                        var f = cb_obj.get('value');
#                        var data = source.get('data')
#                        s2.set('options', source.data['entry']);
#                        s2.trigger('change');
#                        """)
#    source.data['entry'] = ifgs.keys()
#    source.trigger('data', source.data, source.data)

# Get the different result options and create dropdown.

def find_ifgs():

    date_1_key = date_1.value
    ifgs = dict()
    for ifg in ifg_keys:
        if date_1_key == ifg[:8]:
            ifgs[ifg[9:]] = ifg
        elif date_1_key == ifg[9:]:
            ifgs[ifg[:8]] = ifg

    return ifgs

def find_image(slice=False):

    date_1_key = date_1.value
    date_2_key = date_2.value
    if date_2_key:
        if date_1_key + '_' + date_2_key in ifg_keys:
            key = date_1_key + '_' + date_2_key
        elif date_2_key + '_' + date_1_key in ifg_keys:
            key = date_2_key + '_' + date_1_key
    else:
        key = date_1_key

    if len(key) == 8:
        image = stack.images[key]
    else:
        image = stack.interferograms[key]

    if slice:
        image = image.slices[slice_list.value]

    return key, image

def reload_slices():

    key, image = find_image()

    slices = image.slices.keys()
    if slice_list.value in slices:
        slices.remove(slice_list.value)
        slice_str = [slice_list.value, '']
    else:
        slice_str = ['']

    slice_str.extend(slices)
    slice_list.options = slice_str

def reload_steps():

    key, image = find_image()

    steps = []
    if slice_list.value == '':
        for step in image.data_disk.keys():
            if len(step) != 0:
                steps.append(step)
    else:
        for step in image.slices[slice_list.value].data_disk.keys():
            if len(step) != 0:
                steps.append(step)

    if step_list.value in steps:
        steps.remove(step_list.value)
        step_str = [step_list.value, '']
    else:
        step_str = ['']

    step_str.extend(steps)
    step_list.options = step_str

def reload_data_types():

    key, image = find_image()

    data_types = ['']
    if slice_list.value == '':
        if step_list.value:
            data_types = image.data_disk[step_list.value].keys()
    else:
        if step_list.value:
            data_types = image.slices[slice_list.value].data_disk[step_list.value].keys()

    if data_type_list.value in data_types:
        data_types.remove(data_type_list.value)
        data_type_str = [data_type_list.value, '']
    else:
        data_type_str = ['']

    data_type_str.extend(data_types)
    data_type_list.options = data_type_str

def update_first_date(attr, old, new):
    ifgs = find_ifgs()
    ops = ['']
    ops.extend(ifgs.keys())
    date_2.options = ops

    reload_slices()
    reload_steps()
    reload_data_types()
    update()

def update_second_date(attr, old, new):

    reload_slices()
    reload_steps()
    reload_data_types()
    update()

def update_slice(attr, old, new):

    reload_steps()
    reload_data_types()
    update()

def update_step(attr, old, new):

    reload_data_types()
    update()

def update_data_type(attr, old, new):

    if slice_list.value:
        key, image = find_image(slice=True)
    else:
        key, image = find_image(slice=False)
    steps = step_list.options

    if step_list.value in steps:
        if data_type_list.value in image.data_files[step_list.value].keys():
            data_type = image.data_types[step_list.value][data_type_list.value]
        else:
            return
    else:
        return

    if step_list.value in ['structure_function']:
        plot_type.options = ['mean bandwith', 'mean directions']
        plot_type.value = ['mean bandwith']

    elif data_type in ['complex_int', 'complex_short', 'complex_real4']:
        plot_type.options = ['phase', 'log10 amplitude', 'mixed']
        plot_type.value = 'phase'

    elif data_type in ['real8', 'real4', 'real2', 'int8', 'int16', 'int32', 'int64']:
        plot_type.options = ['linear', 'log10']
        plot_type.value = 'linear'

    update()

def update_plot_type(attr, old, new):

    update(find_limits=True)

def update_slider(attr, old, new):

    update()

def update(find_limits=False):

    if slice_list.value:
        key, image = find_image(slice=True)
    else:
        key, image = find_image(slice=False)

    steps = []
    for step in image.data_disk.keys():
        if len(step) != 0:
            steps.append(step)

    if step_list.value in steps and len(step_list.value) > 0:
        if data_type_list.value in image.data_disk[step_list.value].keys() and len(data_type_list.value) > 0:

            if not image.check_loaded(step_list.value, loc='disk', file_type=data_type_list.value, warn=False):
                if not image.read_data_memmap(step_list.value, data_type_list.value):
                    return

            data = image.data_disk[step_list.value][data_type_list.value]
            data_size = image.data_sizes[step_list.value][data_type_list.value]
            data_type = image.data_types[step_list.value][data_type_list.value]

            data_pix = data_size[0] * data_size[1]

            if data_pix > max_pixel_num:

                i = 2
                while data_pix / i**2 > max_pixel_num:
                    i += 1

                data = copy.deepcopy(data[::i, ::i])

            # Convert numpy value if needed
            if data_type == 'complex_int':
                data = data.view(np.int16).astype('float32', subok=False).view(np.complex64)
            elif data_type == 'complex_short':
                data = data.view(np.float16).astype('float32', subok=False).view(np.complex64)

            if plot_type.value == '':
                return

            data[np.abs(data) == 0] = np.nan

            if plot_type.value == 'phase':
                data = np.angle(data)
            elif plot_type.value == 'mixed':
                amp = np.log10(np.abs(data))
                data = np.angle(data)
            elif plot_type.value == 'log10 amplitude':
                data = np.log10(np.abs(data))
            elif plot_type.value == 'log10':
                np.log10(data)

            if find_limits:
                [range.start, range.end] = [np.nanpercentile(data, 1), np.nanpercentile(data, 99)]

            color_mapper = LinearColorMapper(palette="Spectral11", low=range.start, high=range.end, nan_color='white')
            plot.image(image=[data], x=0, y=0, dw=10, dh=10, color_mapper=color_mapper)
            # plot.image(image=[data], x=0, y=0, dw=10, dh=10, color_mapper=color_mapper)

# Create the image and interferogram dropdown.
date_1 = Select(title="Date_1", options=[''] + sorted(possible_dates), value="")
date_1.on_change("value", update_first_date)
date_2 = Select(title="Date_2 for ifg", value='')
date_2.on_change("value", update_second_date)
slice_list = Select(title="Slice", value='')
slice_list.on_change("value", update_slice)
step_list = Select(title="Processing step", value='')
step_list.on_change("value", update_step)
data_type_list = Select(title="Output type", value='')
data_type_list.on_change('value', update_data_type)

# Image type
plot_type = Select(value = '', options=['', 'phase', 'wrapped_phase', 'amplitude'])
plot_type.on_change('value', update_plot_type)

# Range slider
range = RangeSlider(start=-np.pi, end=np.pi, value=(-np.pi, np.pi), step=.1, title="Stuff")
range.on_change('value', update_slider)

# Plot
inputs = widgetbox(date_1, date_2, slice_list, step_list, data_type_list, plot_type, range)

plot = figure(x_range=(0, 10), y_range=(0, 10),
              tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"

# show(im_layout)

# A bit more complicated but nice way to create structure functions using polygon select.
def interactive_structure_function():
    print('Not working!')


