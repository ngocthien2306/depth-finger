# python python/depth_reprojection.py --projector-width 2160 --projector-height 3820 --calib "data/new_calib_v5.yaml" --bias "event_data/finger_v1.bias" --input "event_data/finger_v1.raw" --z-near 0.3 --z-far 0.6 --no-frame-dropping False --camera-perspective

# python python/depth_reprojection.py --calib "data/new_calib_v5.yaml" --bias "../data/biases_v2.bias" --input "event_data/error_01.raw" --no-frame-dropping False --camera-perspective


import cv2
from metavision_sdk_ui import EventLoop
import metavision_hal as mv_hal

from metavision_core.event_io import EventsIterator

from bias_events_iterator import NonBufferedBiasEventsIterator, BiasEventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, Window

from bias_events_iterator import NonBufferedBiasEventsIterator
from depth_reprojection_processor import DepthReprojectionProcessor, RuntimeParams

import click
import sys
from biases import Biases, load_bias_file
import time

import json
import socket
import struct
import os
from glob import glob


def project_events(ev_processor, file_input):
    bias_file = '../data/biases_v2.bias'    
    # file_path = '../../event_data/type_68cm.raw'
    EV_PACKETS_PER_FRAME = 1
    delta_t = 1e6 / 60 / EV_PACKETS_PER_FRAME
    mv_iterator = NonBufferedBiasEventsIterator(input_filename=file_input, delta_t=delta_t, bias_file=bias_file)
    
    height, width = mv_iterator.get_size()
    print("Dimensions:", width, height)
    loop_durations = []

    for evs in mv_iterator:
        start_time = time.perf_counter()  # Start measuring time for the loop

        # Directly measure the time for the main loop without using StatsPrinter
        main_loop_start = time.perf_counter()  # Start timing the main loop
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()
        if not len(evs):
            continue

        ev_processor.process_events(evs)
        # send_data(sock, ADDR, text_position)

        if ev_processor.should_close():
            sys.exit(0)
        
       

def main(file_input):
    f_name = os.path.basename(file_input).split('.')[0]
    params = RuntimeParams(
        camera_width=1280, 
        camera_height=720,        
        projector_width=1080,
        projector_height=1920,       
        projector_fps=60,        
        z_near=0.1,
        z_far=0.68,                
        calib='../data/calib_68cm_v11.yaml',
        no_frame_dropping=False,
        camera_perspective=True,
        projector_time_map=None,        
    )

    # sock, ADDR = setup_socket()

    # calib='/home/farchan/Downloads/X-maps-prev/X-maps/data/nebra_evk3.0/X-maps_calibration_8_5mm.yaml',
    with DepthReprojectionProcessor(params=params) as ev_processor:
        while True:
            project_events(ev_processor, file_input)
            ev_processor.reset()
  
if __name__ == "__main__":
    # file_path = 'D:/data_20241122/68cm/'
    # list_file =  glob(file_path + "*.raw")
    # print(list_file)
    # file = list_file[3]
    file = r"C:\Users\pclab321\Documents\metavision\recordings\Zack_R_00.raw"
    main(file)