from typing import Any, Callable, Optional

from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

from depth_reprojection_pipe import DepthReprojectionPipe
from stats_printer import Occurences, StatsPrinter

from dataclasses import dataclass, field

from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
import time
import cv2
import numpy as np
import collections
import socket
import threading
import struct
import json
# from python.finger_detect.hand_detection_tracked import load_model
from finger_detect.hand_detection import load_model

from CPTouchCalibrate import CPTouchCalibrate

# from stroke import StrokeSmoothing

import math

import pyautogui

pyautogui.FAILSAFE = False

@dataclass
class RuntimeParams:
    camera_width: int
    camera_height: int
    projector_width: int
    projector_height: int
    projector_fps: int
    z_near: float
    z_far: float
    calib: str
    projector_time_map: str
    no_frame_dropping: bool
    camera_perspective: bool    

    @property
    def should_drop_frames(self):
        return not self.no_frame_dropping

USE_FAKE_WINDOW = False 

class FakeWindow:
    def should_close(self):
        return False

    def show_async(self, img):
        # In a fake window, we won't actually display anything
        pass

    def set_keyboard_callback(self, cb):
        # Fake window does not handle keyboard events
        pass


@dataclass
class DepthReprojectionProcessor:
    params: RuntimeParams

    stats_printer: StatsPrinter = StatsPrinter()

    _pipe: DepthReprojectionPipe = field(init=False)
    _window: BaseWindow = field(init=False)

    # Add attributes for FPS calculation
    _last_time: float = field(default_factory=time.time, init=False)
    _frame_count: int = field(default=0, init=False)
    _fps: float = field(default=0.0, init=False)


    # _smoother: StrokeSmoothing = field(init=False)
    _depth_arr = []
    _depth_len = 10    
    _point_arr = []
    # _point_arr = {}
    _evs_frame_idx = 0
    _bg_val = None    
    _prev_point = None
    _prev_point_ori = None
    _prev_point_dict = {}
    _finger_valid = False
    _region = [(457, 130), (700, 553)]
    _bg_calib = False
    # _bg_calib = True
    _bg_calib_range = (915,925)


    _finger_min = 5555
    _finger_max = 0
    _finger_depth_array = []    
    _thres_max = 38 #70
    _point_thres = 5 #10

    _should_reset = False
    _mouse_press_thres = 20




    def __init__(self, params: RuntimeParams):
        self.params = params
        # self.setup_socket()
        # self._finger_detection = load_model(r"./finger_detect/best_40_68_cm_afternoon_26_11.onnx")
        self._finger_detection = load_model(r"./finger_detect/TU_best.onnx")
        # self._finger_detection = load_model(r"./finger_detect/best68cm_new.onnx")
        self.startCalibrate = False
        self.touchCalibrate = CPTouchCalibrate()
        # self._smoother = StrokeSmoothing()

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

    # Method to send data in a thread
    # def send_data_in_thread(self, p):
    #     # Create a new thread to call the send_data function
    #     threading.Thread(target=self.send_data, args=(p,), daemon=True).start()

    def __exit__(self, *exc_info):
        # self.sock.close()  
        self.stats_printer.print_stats()
        return False

    def should_reset(self):              
        return self._should_reset

    def should_close(self):
        return self._window.should_close()
    
    def position_mapping(self, x_ori, y_ori):
            (_x_min, _y_min), (_x_max, _y_max) = self._region
            y_map = (x_ori/(_x_max-_x_min)) * 1080
            y_map = 1080 - np.clip(y_map, 0, 1080)
            x_map = (y_ori/(_y_max-_y_min)) * 1920
            x_map = np.clip(x_map, 0, 1920)
            return x_map, y_map

    def show_async(self, depth_map, depth_nomap):
        central_pos = None
        rect_pos = (0,0,0,0)
        # FPS calculation
        frame_count = 0
        start_time = time.time()        

        
        
        self.draw_text(depth_map, f"thres:{self._point_thres}", org=(10, 350), front_scale=0.7, color=(0,0,0), thickness=1)
        self.draw_text(depth_map, f"bg_val: {self._bg_val}", org=(10, 400), front_scale=0.7, color=(0,0,0), thickness=1)
        self.draw_text(depth_map, f"region: {self._region}", org=(10, 450), front_scale=0.7, color=(0,0,0), thickness=1)
        


        # ------------ finger detection  -----------------------

        d_min = np.min(depth_nomap)

        # display 12 points i plane        
        # [(371, 135), (702, 539)]
        (_x_min, _y_min), (_x_max, _y_max) = self._region
        rx, ry, rw, rh =  [_x_min, _y_min, _x_max-_x_min, _y_max-_y_min]
        margin = 10
        # depth_map = self._finger_detection(depth_map)
        d_max = np.max(depth_nomap)
        # ------------------- check bg value ----------------------
        
        # ------------------- check bg value ----------------------
        calib_adjust = 0.0001
        
        if self._bg_calib == False:                

            avg_bg_calib = 0
            avg_bg_calib += depth_nomap[ry+margin, rx+margin]
            avg_bg_calib += depth_nomap[ry+margin, rx+rw//2]
            avg_bg_calib += depth_nomap[ry+margin, rx+rw-margin]

            avg_bg_calib += depth_nomap[ry+rh//3, rx+margin]
            avg_bg_calib += depth_nomap[ry+rh//3, rx+rw//2]
            avg_bg_calib += depth_nomap[ry+rh//3, rx+rw-margin]

            avg_bg_calib += depth_nomap[ry+2*rh//3, rx+margin]
            avg_bg_calib += depth_nomap[ry+2*rh//3, rx+rw//2]
            avg_bg_calib += depth_nomap[ry+2*rh//3, rx+rw-margin]

            avg_bg_calib += depth_nomap[ry+rh-margin, rx+margin]
            avg_bg_calib += depth_nomap[ry+rh-margin, rx+rw//2]
            avg_bg_calib += depth_nomap[ry+rh-margin, rx+rw-margin]

            avg_bg_calib /= 12
                        
            self.draw_text(depth_map, f"bg depth avg: {avg_bg_calib}", org=(10, 500), front_scale=0.5)

            calib_bg = self._pipe.get_calib_bg()
            self.draw_text(depth_map, f"calib bg: {calib_bg}", org=(10, 550), front_scale=0.5)            
            self._bg_val = np.max(depth_nomap[_y_min:_y_max, _x_min:_x_max])
            
            if (avg_bg_calib > self._bg_calib_range[1]) or (self._bg_val > self._bg_calib_range[1]):
                calib_bg[0][0] -= calib_adjust
                self._pipe.set_calib_bg(calib_bg)
                self._pipe.reset_depth_diff()
            elif (avg_bg_calib < self._bg_calib_range[0]) or (self._bg_val < self._bg_calib_range[0]):
                calib_bg[0][0] += calib_adjust
                self._pipe.set_calib_bg(calib_bg) 
                self._pipe.reset_depth_diff()
            else:
                self._bg_calib = True                    
                    

        
        if self._bg_val is None:           
            (_x_min, _y_min), (_x_max, _y_max) = self._region            
            self._bg_val = np.max(depth_nomap[_y_min:_y_max, _x_min:_x_max])

        # cv2.rectangle(depth_map, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 4)        
        
        # point_11 = round(self._bg_val-depth_nomap[ry+margin, rx+margin])#/(d_max - d_min)*62+1, 1)        
        # # point_11 = round((point_11-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_11}", org=(rx+margin, ry+margin), front_scale=0.5)
        # cv2.circle(depth_map, [rx+margin, ry+margin], 2, (0, 255, 255), 3)
        # point_12 = round(self._bg_val-depth_nomap[ry+margin, rx+rw//2])#/(d_max - d_min)*62+1, 1)        
        # # point_12 = round((point_12-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_12}", org=(rx+rw//2, ry+margin), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw//2, ry+margin], 2, (0, 255, 255), 3)
        # point_13 = round(self._bg_val-depth_nomap[ry+margin, rx+rw-margin])#/(d_max - d_min)*62+1, 1)        
        # # point_13 = round((point_13-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_13}", org=(rx+rw-margin, ry+margin), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw-margin, ry+margin], 2, (0, 255, 255), 3)

        # point_21 = round(self._bg_val-depth_nomap[ry+rh//3, rx+margin])#/(d_max - d_min)*62+1, 1)        
        # # point_21 = round((point_21-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_21}", org=(rx+margin, ry+rh//3), front_scale=0.5)        
        # cv2.circle(depth_map, [rx+margin, ry+rh//3], 2, (0, 255, 255), 3)
        # point_22 = round(self._bg_val-depth_nomap[ry+rh//3, rx+rw//2])#/(d_max - d_min)*62+1, 1)        
        # # point_22 = round((point_22-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_22}", org=(rx+rw//2, ry+rh//3), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw//2, ry+rh//3], 2, (0, 255, 255), 3)
        # point_23 = round(self._bg_val-depth_nomap[ry+rh//3, rx+rw-margin])#/(d_max - d_min)*62+1, 1)        
        # # point_23 = round((point_23-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_23}", org=(rx+rw-margin, ry+rh//3), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw-margin, ry+rh//3], 2, (0, 255, 255), 3)

        # point_31 = round(self._bg_val-depth_nomap[ry+2*rh//3, rx+margin])#/(d_max - d_min)*62+1, 1)        
        # # point_31 = round((point_31-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_31}", org=(rx+margin, ry+2*rh//3), front_scale=0.5)    
        # cv2.circle(depth_map, [rx+margin, ry+2*rh//3], 2, (0, 255, 255), 3)
        # point_32 = round(self._bg_val-depth_nomap[ry+2*rh//3, rx+rw//2])#/(d_max - d_min)*62+1, 1)        
        # # point_32 = round((point_32-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_32}", org=(rx+rw//2, ry+2*rh//3), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw//2, ry+2*rh//3], 2, (0, 255, 255), 3)
        # point_33 = round(self._bg_val-depth_nomap[ry+2*rh//3, rx+rw-margin])#/(d_max - d_min)*62+1, 1)        
        # # point_33 = round((point_33-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_33}", org=(rx+rw-margin, ry+2*rh//3), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw-margin, ry+2*rh//3], 2, (0, 255, 255), 3)

        # point_41 = round(self._bg_val-depth_nomap[ry+rh-margin, rx+margin])#/(d_max - d_min)*62+1, 1)        
        # # point_41 = round((point_41-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_41}", org=(rx+margin, ry+rh-margin), front_scale=0.5)    
        # cv2.circle(depth_map, [rx+margin, ry+rh-margin], 2, (0, 255, 255), 3)
        # point_42 = round(self._bg_val-depth_nomap[ry+rh-margin, rx+rw//2])#/(d_max - d_min)*62+1, 1)        
        # # point_42 = round((point_42-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_42}", org=(rx+rw//2, ry+rh-margin), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw//2, ry+rh-margin], 2, (0, 255, 255), 3)
        # point_43 = round(self._bg_val-depth_nomap[ry+rh-margin, rx+rw-margin])#/(d_max - d_min)*62+1, 1)        
        # # point_43 = round((point_43-d_min)/(d_max - d_min)*68, 2)
        # self.draw_text(depth_map, f"{point_43}", org=(rx+rw-margin, ry+rh-margin), front_scale=0.5)
        # cv2.circle(depth_map, [rx+rw-margin, ry+rh-margin], 2, (0, 255, 255), 3)


        
                

        # ------------ finger detection  -----------------------
        # depth_map, list_fingertip = self._finger_detection(depth_map)
        # [(475, 135), (702, 539)]
        # (x_min, y_min), (x_max, y_max) = self._region
        # depth_map[_y_min:_y_max, _x_min:_x_max], list_fingertip = self._finger_detection(depth_map[_y_min:_y_max, _x_min:_x_max])
        
        depth_map[_y_min:_y_max, _x_min:_x_max], list_fingertip, track_id_list = self._finger_detection(depth_map[_y_min:_y_max, _x_min:_x_max])


        # Check sender
        list_points = []
        prev_time = time.perf_counter()
        num_100 = 0

        # Loop through the list of fingertip coordinates
        position = []
        
        # for fi, fger in enumerate(list_fingertip):
        for fi, (fger, id_finger) in enumerate(zip(list_fingertip, track_id_list)):
            self._point_arr.append(fger)
            if len(self._point_arr) > 5:
                self._point_arr.pop(0)
            # print(f"========= finger ========> {self._point_arr}")
            # fger = np.average(np.array(self._point_arr), axis=0).astype(np.int)   

            fx, fy = fger  
            fx += _x_min
            fy += _y_min            

            # depth_value = depth_nomap[fy, fx]

            _thes_shadow = 3
            _thres_shadow_density = 650
            depth_value_mask = depth_nomap[fy-_thes_shadow:fy+_thes_shadow, fx-_thes_shadow:fx+_thes_shadow] > _thres_shadow_density
            if np.sum(depth_value_mask) == 0:
                depth_value = self._bg_val
            else:
                depth_value = np.min(depth_nomap[fy-_thes_shadow:fy+_thes_shadow, fx-_thes_shadow:fx+_thes_shadow][depth_value_mask])  

            d_max = np.max(depth_nomap)
            d_min = np.min(depth_nomap)    

            # depth_value = round(depth_value/(d_max - d_min)*62+1, 1)
            # depth_value = (depth_value-d_min)/(d_max - d_min)*30+2
            depth_reverse = self._bg_val - depth_value
            dep_cm_thresh = 0.251
            depth_value_cm = round(depth_reverse*dep_cm_thresh, 2)
            depth_value_socket = 68 - depth_value_cm            
            # depth_reverse = np.clip(0, np.abs(depth_reverse))

            if depth_reverse > self._finger_max:
                self._finger_max = depth_reverse
            if depth_reverse < self._finger_min:
                self._finger_min = depth_reverse

            # mmm_text = f"Check min-max: {self._finger_min}, {self._finger_max}, {self._bg_val}"
            # self.draw_text(depth_map, mmm_text, org=(10, 300), front_scale=1, color=(0,0,0), thickness=1)

            # self._finger_depth_array.append(depth_reverse)
            # if len(self._finger_depth_array)>10:
            #     self._finger_depth_array.pop(0)
                
            # mmm_text = f"Check min-max: {np.min(self._finger_depth_array)}, {min(np.max(self._finger_depth_array),self._thres_max)}"
            # self.draw_text(depth_map, mmm_text, org=(10, 350), front_scale=1, color=(0,0,0), thickness=1)


            # Draw a circle on the depth map at the fingertip coordinates (red circle)
            
            # ------------------- fix the finger position ---------------------------------
            
            if id_finger not in self._prev_point_dict:
                self._prev_point_dict[id_finger] = [fx, fy, depth_reverse]
            
            point_dist = round(math.sqrt((fx - self._prev_point_dict[id_finger][0])**2 + (fy - self._prev_point_dict[id_finger][1])**2), 3)
            
            if point_dist < self._point_thres:                
                fx = self._prev_point_dict[id_finger][0]
                fy = self._prev_point_dict[id_finger][1]
                        
            self._prev_point_dict[id_finger] = [fx, fy, depth_reverse]                        
            
                        
            
            # ------------------- fix the finger position ---------------------------------
            
            cv2.circle(depth_map, (fx, fy), 5, (0, 255, 0), -1)  # (fy, fx) -> (x, y)
            # Display the depth value and the fx, fy coordinates near the fingertip
            # text = f'Depth: {depth_reverse} | ({fx}, {fy})'  
            text = f'Depth: {depth_reverse} | ({fx-_x_min}, {fy-_y_min}) '  
            cv2.putText(depth_map, text, (fx - 50, fy - 20),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # position.append((fi, fx, fy, depth_value_socket))
            position.append((id_finger, fx-_x_min, fy-_y_min, depth_reverse))
            # position.append((id_finger, fx, fy, depth_value))
            
            # self._prev_point_dict[id_finger]=[fx, fy, depth_reverse]



        # ------------ highest detection  -----------------------
        # depth_map, central_pos, rect_pos = self.detect_region(depth_map, depth_nomap)

        text_position = ""        
        depth_map_with_fps = depth_map

        dict_frame = self.stats_printer.get_local_frame()              
        total_frame = (dict_frame['frame_drop'] + dict_frame['frame_show']) % 60
        # total_frame = dict_frame['frame_show']
        depth_map_with_fps = self.add_fps_to_image(depth_map, f"FPS: {total_frame}")

        

        # Send data        
        # self.send_data(position)

        # if len(position) == 0:
        #     # position = self._send_data
        #     position = [(0,0,0,-1)]
        if len(position) == 0:                        
            # self.send_data((0,0,0,-1))
            self._finger_valid = False
        else:
            self._finger_valid = True        

        if self._finger_valid:
            for p in position:            
                fid, fx, fy, fd = p
                if fid == 0:                
                    fx_map, f_ymap = self.position_mapping(fx, fy)
                    pyautogui.moveTo(fx_map, f_ymap, _pause=False)

                    is_down = fd < self._mouse_press_thres

                    if is_down:
                        pyautogui.mouseDown(fx_map, f_ymap, _pause=False)
                    else:
                        pyautogui.mouseUp(fx_map, f_ymap, _pause=False)

        
        
        # Generate the color bar        
        color_bar = self.create_color_bar_with_values()

        # Combine the depth map and color bar
        combined_image = self.combine_depth_and_colorbar(depth_map_with_fps, color_bar)

        self.draw_text(combined_image, '68 cm-', org=(1180, 40), color=(0, 0, 255), front_scale=0.8)
        self.draw_text(combined_image, '35 cm-', org=(1180, 300), color=(0, 255, 0), front_scale=0.8)
        self.draw_text(combined_image, '5 cm-', org=(1180, 640), color=(255, 0, 0), front_scale=0.8)
        
        # Display the combined image with FPS and color bar
        self._window.show_async(combined_image)

        # Count frames shown in stats
        self.stats_printer.count("frames shown")



    def add_fps_to_image(self, image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                         font_scale=1, color=(255, 0, 0), thickness=2):
        """
        Adds the FPS text to the image using OpenCV.
        """
        return cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


    
    def create_color_bar_with_values(self, bar_width=60, bar_height=125, num_labels=4):
        # Create a gradient from 10 to 40
        gradient = np.linspace(40, 10, bar_height).reshape((bar_height, 1))
        
        # Normalize the gradient to [0, 1]
        normalized_gradient = (gradient - 40) / (10 - 40)

        # Convert the normalized gradient to a color map
        colormap = cv2.applyColorMap((normalized_gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize the color bar
        color_bar = np.repeat(colormap, bar_width, axis=1)

        # Convert to BGR format
        color_bar_bgr = cv2.cvtColor(color_bar, cv2.COLOR_RGB2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45  # Smaller font size
        color = (255, 255, 255)
        thickness = 1

        # Define the label positions and values
        label_values = np.linspace(10, 40, num_labels)

        # Calculate positions for labels
        label_positions = np.linspace(0, bar_height - 1, num_labels, dtype=int)

        for i, pos in enumerate(label_positions):
            depth_value = label_values[i]
            # cv2.putText(color_bar_bgr, f"{depth_value:.2f}", (5, pos + 25), font, font_scale, color, thickness)

        return color_bar_bgr


    def combine_depth_and_colorbar(self, depth_map, color_bar):
        """
        Combines the depth map and color bar side by side.
        """
        # Ensure both images are the same height before concatenation
        depth_map_height, depth_map_width, _ = depth_map.shape
        color_bar = cv2.resize(color_bar, (color_bar.shape[1], depth_map_height))

        # Combine depth map and color bar horizontally
        combined_image = np.hstack((depth_map, color_bar))

        return combined_image
    

    def draw_text(self, frame, text='test', org=(50,50), front_scale=1, color=(255, 0, 0), thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, text, org, font,  front_scale, color, thickness, cv2.LINE_AA)
        return frame

    def detect_region(self, depth_map, depth_nomap, region_thres=50, bg_thres=100, threshold_value=99.5):
        #depth_map = cv2.medianBlur(depth_map, 3)        
         
        gray_img = 255 - depth_nomap

        gray_img = np.where(gray_img >= bg_thres, 0, gray_img)        

        percentile_90 = np.percentile(gray_img, threshold_value)        

        gray_img = cv2.threshold(gray_img, percentile_90, 255, cv2.THRESH_BINARY)[1]     
   

        # Apply connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_img, connectivity=8)

        # print(f"===> {num_labels} | {labels} | {stats} | {centroids}")

        areas = stats[:, cv2.CC_STAT_AREA]
        region_areas = areas[1:]  # Exclude the first element (background)
        
        # print("region_areas", region_areas)
        if len(region_areas) > 0 and np.max(region_areas) > region_thres:
            max_idx = np.argmax(region_areas)
            #print(f"idx: {max_idx}")            

            x = stats[max_idx+1, cv2.CC_STAT_LEFT]
            y = stats[max_idx+1, cv2.CC_STAT_TOP]
            w = stats[max_idx+1, cv2.CC_STAT_WIDTH]
            h = stats[max_idx+1, cv2.CC_STAT_HEIGHT]
            area = stats[max_idx+1, cv2.CC_STAT_AREA]
            centroid = centroids[max_idx+1]
            # cv2.rectangle(depth_map, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(depth_map, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
            return depth_map, centroid, (x, y, w, h)

        return depth_map, None, None


    def __enter__(self):
        
        self._pipe = DepthReprojectionPipe(
            params=self.params, stats_printer=self.stats_printer, frame_callback=self.show_async
        )        
        
        if USE_FAKE_WINDOW:
            self._window = FakeWindow()
        else:
            self._window = MTWindow(
                title="X Maps Depth",
                width=self.params.camera_width if self.params.camera_perspective else self.params.projector_width,
                height=self.params.camera_height if self.params.camera_perspective else self.params.projector_height,
                mode=BaseWindow.RenderMode.BGR,
                open_directly=True,
            )

            self._window_frame = MTWindow(
                title="Frame",
                width=self.params.camera_width if self.params.camera_perspective else self.params.projector_width,
                height=self.params.camera_height if self.params.camera_perspective else self.params.projector_height,
                mode=BaseWindow.RenderMode.BGR,
                open_directly=True,
            )

        self._window.set_keyboard_callback(self.keyboard_cb)
        self._window_frame.set_keyboard_callback(self.keyboard_cb)

        self.periodic_gen = PeriodicFrameGenerationAlgorithm(
            self.params.camera_width,
            self.params.camera_height,
            accumulation_time_us=20000,
            fps=60,
        )

        self.periodic_gen.set_output_callback(self.periodic_cb)

        self.accumulated_events = []
        return self

    # def __exit__(self, *exc_info):
    #     self.stats_printer.print_stats()
    #     return False

    def keyboard_cb(self, key, scancode, action, mods):
        if action != UIAction.RELEASE:
            return
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            self._window.set_close_flag()
        if key == UIKeyEvent.KEY_E:
            self._pipe.select_next_frame_event_filter()
        if key == UIKeyEvent.KEY_S:
            self.stats_printer.toggle_silence()
        if key == UIKeyEvent.KEY_Z:
            self.startCalibrate = True
        if key == UIKeyEvent.KEY_R:            
            # self._should_reset = True
            self._pipe.reload_calib()
            self._bg_calib = False
        if key == UIKeyEvent.KEY_D:
            self._point_thres = 5
        if key == UIKeyEvent.KEY_T:
            self._point_thres = 15        

    def process_events(self, evs):        
        self.stats_printer.print_stats_if_needed()
        self.stats_printer.count("processed evs", len(evs))
        # print(f"================= prev point ==================> {self._prev_point}")
        self._pipe.process_events(evs, self._prev_point, self._finger_valid)
        # self._pipe.process_events(evs, None, self._finger_valid)
        self.stats_printer.print_stats_if_needed()

        if self._pipe.get_proj_region() is not None:
            self._region = self._pipe.get_proj_region()
        # print("================= get_proj_region ===================> ", self._pipe.get_proj_region())

        # if self.startCalibrate:
        #     event_frame = self.create_image_from_events(evs, self.params.camera_width,self.params.camera_height)
        #     result, x, y, w, h = self.touchCalibrate.startMapping(event_frame)
        #     if result == 0:
        #         self.startCalibrate = False

        self.visualize_frame(evs)
        
        # return self._text_position
    def periodic_cb(self, ts, frame):
        self._window_frame.show_async(frame)

    def visualize_frame(self, evs):        
        self.periodic_gen.process_events(evs)

    def reset(self):
        self._pipe.reset()

    
    def make_binary_histo(self, events, img=None, width=304, height=240):    
        if img is None:
            img = 127 * np.ones((height, width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 127
        if events.size:
            assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
            assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

            img[events['y'], events['x'], :] = 255 * events['p'][:, None]
        return img
    
    def create_image_from_events(self, events, width, height):
        img = np.full((height, width, 3), 128, dtype=np.uint8)
        img[events['y'], events['x']] = 255 * events['p'][:, None]
        return img