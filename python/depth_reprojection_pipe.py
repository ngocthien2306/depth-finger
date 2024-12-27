from typing import Any, Callable

from metavision_sdk_core import PolarityFilterAlgorithm, RoiFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

from trigger_finder import RobustTriggerFinder
# from stats_printer import StatsPrinter, SingleTimer
from stats_printer import StatsPrinter
from cam_proj_calibration import CamProjMaps, CamProjCalibrationParams
from x_maps_disparity import XMapsDisparity
from proj_time_map import ProjectorTimeMap
from disp_to_depth import DisparityToDepth
from timing_watchdog import TimingWatchdog
from event_buf_pool import EventBufPool
from frame_event_filter import FrameEventFilterProcessor

from dataclasses import dataclass, field

import cv2
import numpy as np

# import matplotlib.pyplot as plt
import time

from contextlib import contextmanager

import socket
import threading
import struct
import json


# @dataclass
# class DepthReprojectionPipe:
#     params: "RuntimeParams"
#     stats_printer: StatsPrinter
#     frame_callback: Callable

#     pos_filter = PolarityFilterAlgorithm(1)

#     # TODO revisit: does this have an effect on latency?
#     act_filter: ActivityNoiseFilterAlgorithm = field(init=False)

#     pos_events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()
#     # act_events_buf = None

#     calib_maps: CamProjMaps = field(init=False)

#     trigger_finder: RobustTriggerFinder = field(init=False)

#     ev_filter_proc = FrameEventFilterProcessor()

#     x_maps_disp: XMapsDisparity = field(init=False)
#     disp_to_depth: DisparityToDepth = field(init=False)

#     watchdog: TimingWatchdog = field(init=False)

#     pool = EventBufPool()    

##
@contextmanager
def timed_operation(message: str):
    """Helper function to time operations."""
    start_time = time.perf_counter()
    yield  # This allows the block of code to run
    elapsed_time = time.perf_counter() - start_time
    print(f"{message}: {elapsed_time:.6f} seconds")

@dataclass
class DepthReprojectionPipe:
    params: "RuntimeParams"
    stats_printer: StatsPrinter
    frame_callback: Callable

    roi_filter = RoiFilterAlgorithm(x0=350, y0=120, x1=750, y1=650, output_relative_coordinates=False)

    pos_filter = PolarityFilterAlgorithm(1)

    # TODO revisit: does this have an effect on latency?
    act_filter: ActivityNoiseFilterAlgorithm = field(init=False)

    roi_filter_buf = RoiFilterAlgorithm.get_empty_output_buffer()

    pos_events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()
    # act_events_buf = None

    calib_params: CamProjCalibrationParams = field(init=False)
    calib_maps: CamProjMaps = field(init=False)

    trigger_finder: RobustTriggerFinder = field(init=False)

    ev_filter_proc = FrameEventFilterProcessor()

    x_maps_disp: XMapsDisparity = field(init=False)
    disp_to_depth: DisparityToDepth = field(init=False)

    watchdog: TimingWatchdog = field(init=False)

    pool = EventBufPool()

    def __post_init__(self):
        # self.setup_socket()
        self.act_filter = ActivityNoiseFilterAlgorithm(
            self.params.camera_width, self.params.camera_height, int(1e6 / self.params.projector_fps)
        )

        with timed_operation("Setting up calibration"):
            self.calib_params = CamProjCalibrationParams.from_yaml(
                self.params.calib,
                self.params.camera_width,
                self.params.camera_height,
                self.params.projector_width,
                self.params.projector_height,
            )
            self.calib_maps = CamProjMaps(self.calib_params)

        with timed_operation("Setting up projector time map"):
            if self.params.projector_time_map is not None:
                proj_time_map = ProjectorTimeMap.from_file(self.params.projector_time_map)
            else:
                proj_time_map = ProjectorTimeMap.from_calib(self.calib_params, self.calib_maps)

        with timed_operation("Setting up projector X-map"):
            self.x_maps_disp = XMapsDisparity(
                calib_params=self.calib_params,
                cam_proj_maps=self.calib_maps,
                proj_time_map_rect=proj_time_map.projector_time_map_rectified,
            )

        with timed_operation("Setting up disparity to depth"):
            self.disp_to_depth = DisparityToDepth(
                stats=self.stats_printer,
                calib_params=self.calib_params,
                calib_maps=self.calib_maps,
                z_near=self.params.z_near,
                z_far=self.params.z_far,
            )

    

        self.trigger_finder = RobustTriggerFinder(
            projector_fps=self.params.projector_fps,
            stats=self.stats_printer,
            pool=self.pool,
            frame_callback=self.process_ev_frame,
        )

        self.watchdog = TimingWatchdog(stats_printer=self.stats_printer, projector_fps=self.params.projector_fps)

    # -------------------------------- pipe --------------------------------------  
    def reload_calib(self) :
        self.calib_params = CamProjCalibrationParams.from_yaml(
                self.params.calib,
                self.params.camera_width,
                self.params.camera_height,
                self.params.projector_width,
                self.params.projector_height,
            )
        self.calib_maps = CamProjMaps(self.calib_params)
      
    def get_calib_bg(self, calib_name="relative_translation"):   
        return self.calib_params.get_calib_("relative_translation")

    def set_calib_bg(self, calib_value):
        self.calib_params.set_calib_bg(calib_value)
        self.calib_maps = CamProjMaps(self.calib_params)

    def reset_depth_diff(self):
        self.disp_to_depth.reset_depth_diff()
    # ------------------------------- socket -------------------------------------
    # def setup_socket(self):
    #     CONFIG_FILE = "../localhost/config.json"
    #     # CONFIG_FILE = "localhost/config.json"
    #     with open(CONFIG_FILE) as f:
    #         config = json.load(f)

    #     self.ADDR:"socket._Address" = (config["addr"], config["port"])
    #     self.STRUCT:struct.Struct = struct.Struct(config["struct"])

    #     # Create a UDP socket
    #     self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     if hasattr(socket, "SO_REUSEPORT"):
    #         self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    # def send_data(self, position):          
    #     packed_data = self.STRUCT.pack(*position)
    #     self.sock.sendto(packed_data, self.ADDR)  
    # --------------------------------------------------------------------

    def get_proj_region(self):
        return self.disp_to_depth.get_proj_region()

    def process_events(self, evs, pos=None, finger_valid=False):
        if self.watchdog.is_processing_behind(evs) and self.params.should_drop_frames:
            # if pos is not None:                
            #     # if pos[3] < 70:
            #     if(finger_valid):
            #         print(f"================= send data [drop] ==================> {pos}")
            #         self.send_data(pos)
            #     else:
            #         self.send_data((0,0,0,-1))
            self.trigger_finder.drop_frame()

        # Apply ROI event
        self.roi_filter.process_events(evs, self.roi_filter_buf)
        self.pos_filter.process_events(self.roi_filter_buf, self.pos_events_buf)

        # self.pos_filter.process_events(evs, self.pos_events_buf)

        act_out_buf = self.pool.get_buf()
        self.act_filter.process_events(self.pos_events_buf, act_out_buf)

        self.trigger_finder.process_events(act_out_buf)

    def process_ev_frame(self, evs):
        """Callback from the trigger finder, evs contain the events of the current frame"""
        # generate_frame(evs, frame)
        # window.show_async(frame)

        with self.stats_printer.measure_time("ev rect"):
            # get rectified event coordinates
            ev_x_rect_i16, ev_y_rect_i16 = self.calib_maps.rectify_cam_coords_i16(evs)

        with self.stats_printer.measure_time("frame ev filter"):
            filtered_evs = self.ev_filter_proc.filter_events(evs, ev_x_rect_i16)
            self.stats_printer.add_metric("frame evs filtered out [%]", 100 - len(filtered_evs) / len(evs) * 100)

            # redo the rectification, because we don't know which events were filtered out
            # TODO perf y coords aren't used in the filtering, are computed twice
            if len(filtered_evs) < len(evs):
                ev_x_rect_i16, ev_y_rect_i16 = self.calib_maps.rectify_cam_coords_i16(filtered_evs)

            evs = filtered_evs

        with self.stats_printer.measure_time("x-maps disp"):
            ev_disparity_f32, inlier_mask = self.x_maps_disp.compute_event_disparity(
                events=evs,
                ev_x_rect_i16=ev_x_rect_i16,
                ev_y_rect_i16=ev_y_rect_i16,
            )

        if self.params.camera_perspective:
            with self.stats_printer.measure_time("disp map"):
                disp_map = self.calib_maps.compute_disp_map_camera_view(
                    events=evs, inlier_mask=inlier_mask, ev_disparity_f32=ev_disparity_f32
                )
        else:
            with self.stats_printer.measure_time("disp map"):
                disp_map = self.calib_maps.compute_disp_map_projector_view(
                    ev_x_rect_i16=ev_x_rect_i16,
                    ev_y_rect_i16=ev_y_rect_i16,
                    inlier_mask=inlier_mask,
                    ev_disparity_f32=ev_disparity_f32,
                )
            with self.stats_printer.measure_time("remap disp"):
                disp_map = self.disp_to_depth.remap_rectified_disp_map_to_proj(disp_map)

        with self.stats_printer.measure_time("disp2rgb"):
            depth_map, depth_nomap = self.disp_to_depth.colorize_depth_from_disp(disp_map)            

        self.frame_callback(depth_map, depth_nomap)

    def select_next_frame_event_filter(self):
        new_filter = self.ev_filter_proc.select_next_filter()
        self.stats_printer.log(f"Selected event filter: {new_filter}")

    def reset(self):
        self.watchdog.reset()
        self.trigger_finder.reset()
