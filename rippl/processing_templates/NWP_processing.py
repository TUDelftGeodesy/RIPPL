# Import rippl template
from rippl.processing_templates.InSAR_processing import InSAR_Processing

# Metadata general processing
from rippl.pipeline import Pipeline

# Import processing functions
from rippl.processing_steps.NWP_delay import NWPDelay
from rippl.processing_steps.NWP_ifg import NWPInterferogram


class NWP_Processing(InSAR_Processing):
    
    def calculate_aps(self, ml_name_ray_tracing='aps', ml_name='aps', time_delays=[], time_step=5,
                      time_correction=False, geometry_correction=True,
                      split_signal=True, model_name='era5', model_level_type='pressure_levels',
                      latlim=[-90, 90], lonlim=[-180, 180], spline_type='linear'):

        self.reload_stack()
        aps_pipeline = Pipeline(pixel_no=0, processes=self.processes, chunk_orientation='chunks',
                                processing_type='secondary_slc', image_type='full', stack=self, include_reference=True)
        aps_pipeline.add_processing_step(NWPDelay(out_coor=ml_name, in_coor=ml_name_ray_tracing,
                                                  time_delays=time_delays, time_step=time_step, time_correction=time_correction,
                                                  model_name=model_name, model_level_type=model_level_type,
                                                  geometry_correction=geometry_correction, split_signal=split_signal,
                                                  latlim=latlim, lonlim=lonlim, spline_type=spline_type), True)
        aps_pipeline()

    def calculate_ifg_aps(self, ml_name='aps', geometry_correction=True, split_signal=True, model_name='era5',
                          latlim=[-90, 90], lonlim=[-180, 180]):

        self.reload_stack()
        aps_ifg = Pipeline(pixel_no=0, processes=self.processes, processing_type='ifg', image_type='full',
                                       stack=self)
        aps_ifg.add_processing_step(NWPInterferogram(out_coor=ml_name, in_coor=ml_name,
                                                     model_name=model_name,
                                                     geometry_correction=geometry_correction, split_signal=split_signal)
                                    , True)
        aps_ifg()
