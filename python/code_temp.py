from dataclasses import dataclass
import dataclasses
import math
import os, sys, time, traceback
from typing import Callable, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import cv2

from openni import openni2

oni_device: openni2.Device
color_stream: openni2.VideoStream
depth_stream: openni2.VideoStream
ir_stream: openni2.VideoStream
def init_openni():
    global oni_device, color_stream, depth_stream, ir_stream
    openni2.initialize(os.environ['OPENNI2_REDIST'])
    uris = openni2.Device.enumerate_uris()
    if not uris:
        raise RuntimeError("Camera not found")
    
    oni_device = openni2.Device.open_file(uris[0])
    oni_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)

    # color_stream = oni_device.create_color_stream()
    # color_stream.start()

    depth_stream = oni_device.create_depth_stream()
    def on_new_depth_frame(_:openni2.VideoStream):
        global depth_stream_readable
        depth_stream_readable = True
    depth_stream.register_new_frame_listener(on_new_depth_frame)
    depth_stream.start()

    ir_stream = oni_device.create_ir_stream()
    ir_stream.start()
    return

def dispose_openni():
    global oni_device, color_stream, depth_stream, ir_stream
    if "depth_stream" in globals() and depth_stream:
        depth_stream.close()
        depth_stream = None
        pass
    if "color_stream" in globals() and color_stream:
        color_stream.close()
        color_stream = None
        pass
    if "ir_stream" in globals() and ir_stream:
        ir_stream.close()
        ir_stream = None
        pass
    if "oni_device" in globals() and oni_device:
        oni_device.close()
        oni_device = None
        pass
    if openni2.is_initialized():
        openni2.unload()
        pass
    return

import mediapipe as mp
from mediapipe.python.solutions.hands import Hands
from mediapipe.tasks.python.core.base_options import (BaseOptions)
from mediapipe.tasks.python.vision.hand_landmarker import (HandLandmarkerOptions, HandLandmarker, HandLandmarkerResult)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (VisionTaskRunningMode)
hand_landmarker: HandLandmarker
def init_mediapipe():
    global hand_landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        result_callback = callback,
        num_hands=1
        )
    hand_landmarker = HandLandmarker.create_from_options(options)
    return

import pyautogui
def init_pyautogui():
    pyautogui.FAILSAFE = False
    return

from CPTouchCalibrate import CPTouchCalibrate
touch_calibator: CPTouchCalibrate
def init_touch_calibator():
    global touch_calibator
    touch_calibator = CPTouchCalibrate()

def init():
    init_pyautogui()
    init_openni()
    init_mediapipe()
    init_touch_calibator()
    return

def dispose():
    dispose_openni()
    return

def read_color_mat() -> cv2.typing.MatLike:
    frame = color_stream.read_frame()
    buffer = frame.get_buffer_as_uint8()
    mat = (np.frombuffer(buffer, dtype=np.uint8)
              .reshape(frame.height, frame.width, 3))
    return mat

def read_depth_mat() -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    global depth_stream_readable
    frame = depth_stream.read_frame()
    depth_stream_readable = False
    buffer = frame.get_buffer_as_uint16()
    raw_mat = (np.frombuffer(buffer, dtype=np.uint16)
              .reshape(frame.height, frame.width, 1))
    mat = cv2.convertScaleAbs(raw_mat, alpha=255.0 / 1024.0)
    mask = mat<(255-60)
    mat[mask] = mat[mask]+60
    mat[mat<=60] = 30
    color_map = cv2.applyColorMap(mat, cv2.COLORMAP_HOT)
    return raw_mat, color_map

def read_ir_mat() -> cv2.typing.MatLike:
    frame = ir_stream.read_frame()
    buffer = frame.get_buffer_as_uint16()
    mat = (np.frombuffer(buffer, dtype=np.uint16)
              .reshape(frame.height, frame.width, 1))
    mat = cv2.convertScaleAbs(mat, alpha=255.0 / 1024.0)
    mat = np.clip(mat + 0, 0, 255)
    mat = cv2.merge([mat, mat, mat])
    return mat

def close_ir_stream():
    global ir_stream
    if "ir_stream" in globals() and ir_stream:
        ir_stream.close()
        ir_stream = None

calibrated_thres_mat: cv2.typing.MatLike
def update_calibrated_thres_mat():
    """read `loop_calibrated_depth_mat` ,  write `calibrated_thres_mat`"""
    global calibrated_thres_mat
    depth_int32 = loop_calibrated_depth_mat.astype(np.int32)
    depth_thres = depth_int32 + loop_current_mode.calibrated_threshold
    depth_clip =  np.clip(depth_thres, 0, 65535)
    depth_uint16 = depth_clip.astype(np.uint16)
    calibrated_thres_mat = depth_uint16
    return

def create_timestamp():
    return round(time.time() * 1000)

@dataclass
class Mode:
    name: str
    thres_down: int
    thres_up: int
    smoothing: int
    def copy(self) -> "Mode":
        return dataclasses.replace(self)

keyboard_mode = Mode("keyboard", 15, 20, 2)
painting_mode = Mode("painting", 20, 30, 2)

loop_running: bool
loop_calibrated: bool
loop_has_been_calibrated: bool
loop_calibrated_depth_mat: cv2.typing.MatLike
loop_current_mode: Mode
loop_shared_mat_dict: Dict[int, Tuple[cv2.typing.MatLike, cv2.typing.MatLike]] = {}
def loop():
    global loop_running, loop_calibrated, loop_has_been_calibrated, loop_calibrated_depth_mat, loop_current_mode, hand_landmarker_result
    init()
    with hand_landmarker:
        loop_running = True
        loop_calibrated = False
        loop_has_been_calibrated = False
        loop_current_mode = painting_mode.copy()
        loop_saved_point = (0,0,0)
        loop_saved_vector = (0,0,0)
        loop_smoothing_pool = []
        loop_is_down = False
        loop_state: str = ""
        while loop_running:
            if not loop_calibrated:
                _A_ir_mat = read_ir_mat()
                loop_calibrated = touch_calibator.startMapping(_A_ir_mat)
                continue
            elif not loop_has_been_calibrated and loop_calibrated:
                # _A_raw_depth_mat, _A_color_depth_mat = read_depth_mat()
                # loop_calibrated_depth_mat = _A_raw_depth_mat.copy()
                # update_calibrated_thres_mat()
                close_ir_stream()
                loop_has_been_calibrated = True
                continue
            
            if depth_stream_readable:
                _B_raw_depth_mat, _B_color_depth_mat = read_depth_mat()
                _B_timestamp = create_timestamp()
                loop_shared_mat_dict[_B_timestamp] = (_B_raw_depth_mat, _B_color_depth_mat)
                _B_mp_image = mp.Image(mp.ImageFormat.SRGB , _B_color_depth_mat)
                hand_landmarker.detect_async(_B_mp_image, _B_timestamp)
            
            if hand_landmarker_result is not None:
                _C_timestamp, results = hand_landmarker_result
                _C_chkp = create_timestamp()
                hand_landmarker_result = None
                _C_raw_depth_mat, _C_color_depth_mat = loop_shared_mat_dict[_C_timestamp]

                if len(results.hand_landmarks) > 0 :
                    hand_landmarks = results.hand_landmarks[0]
                    pt1 = [*hand_landmarks][8]
                    x1, y1 = pt1.x, pt1.y
                    x = np.clip(round(x1*640), 0, 639)
                    y = np.clip(round(y1*480), 0, 479)
                    mask = cv2.circle(np.zeros((480,640, 1), np.uint8), (x, y), 5, (255,255,255), -1, cv2.LINE_4)
                    _, z, _, _ = cv2.minMaxLoc(_C_raw_depth_mat, mask)
                    cv2.circle(_C_color_depth_mat, (x, y), 5, (0,255,0), -1, cv2.LINE_AA)
                    
                    cx, cy = touch_calibator.convertPoints((x, y))


                    if True: # Smoothing
                        if len(loop_smoothing_pool) > 0:
                            lx, ly, lz, _ = loop_smoothing_pool[-1]
                            v = [x-lx, y-ly, z-lz]
                            w = np.sqrt(np.dot(v,v))
                            loop_smoothing_pool = [*loop_smoothing_pool, (cx, cy, z, w)]
                            loop_smoothing_pool = loop_smoothing_pool[(-(loop_current_mode.smoothing)):]
                            xs = np.sum([ix*iw for ix,_,_,iw in loop_smoothing_pool]) +cx * w
                            ys = np.sum([iy*iw for _,iy,_,iw in loop_smoothing_pool])  + cy *w
                            zs = np.sum([iz*iw for _,_,iz,iw in loop_smoothing_pool])  + z * w
                            b = np.sum([iw for _,_,_,iw in loop_smoothing_pool])  + w
                            if b > 0:
                                mx, my, mz = xs / b, ys / b, zs / b
                                mx = np.clip(round(mx), 0, 1919)
                                my = np.clip(round(my), 0, 1079)
                            else:
                                mx, my, mz = x, y, z
                            # cv2.circle(_C_color_depth_mat, (mx, my), 5, (255,255,0), -1, cv2.LINE_AA)
                            pass
                        else:
                            mx, my, mz = x, y, z
                            loop_smoothing_pool = [*loop_smoothing_pool, (x, y, z, 0)]
                        pass
                        # print(cx, cy, mx, my)
                        cx, cy = mx, my
                    lcx, lcy, lcz = loop_saved_point
                    v = [cx-lcx, cy-lcy]
                    d = np.sqrt(np.dot(v,v))
                    if d<=12:
                        cx = lcx
                        cy = lcy
                    loop_saved_point = (cx, cy, z)

                    if loop_is_down:
                        is_down = z < loop_current_mode.thres_up
                    else:
                        is_down = z < loop_current_mode.thres_down

                    if is_down:
                        pyautogui.moveTo(cx, cy, _pause=False)
                        pass
                    if loop_is_down != is_down:
                        if is_down:
                            print("mouse down")
                            current_state = "Mouse Down"
                            pyautogui.mouseDown(cx, cy, _pause=False)
                        if not is_down:
                            print("mouse up")
                            current_state = "Mouse Up"
                            pyautogui.mouseUp(cx, cy, _pause=False)
                            pass
                    loop_is_down = is_down
                    loop_state = current_state
                else:
                    cx = 0
                    cy = 0
                    z = 0
                    diff_dot_vec = None
                    diff_dist = 0
                    current_state = "No Hand"
                    if loop_state != current_state:
                        pyautogui.mouseUp(_pause=False)
                        loop_state = current_state

                
                cv2.rectangle(_C_color_depth_mat, (0,0), (300, 160), (0,0,0), -1)
                cv2.putText(_C_color_depth_mat, f"mode: {loop_current_mode.name}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(_C_color_depth_mat, f"state: {current_state}", (0, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(_C_color_depth_mat, f"loc: {round(cx)}, {round(cy)}, {round(z)}", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(_C_color_depth_mat, f"thres down: {loop_current_mode.thres_down} up: {loop_current_mode.thres_up}", (0, 80), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(_C_color_depth_mat, f"smoothing: {loop_current_mode.smoothing}", (0, 100), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(_C_color_depth_mat, f"t: {_C_chkp - _C_timestamp}", (0, 120), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Depth Color Map', _C_color_depth_mat)
                cv2.setWindowProperty('Depth Color Map', cv2.WND_PROP_TOPMOST, 1.0)
                for k in [k for k in loop_shared_mat_dict.keys() if k < (int(time.time() * 1000) - 5000)]:
                    del loop_shared_mat_dict[k]
            key = cv2.waitKeyEx(1)
            if key == ord('q'):
                loop_running = False
            elif key == ord('Q'):
                loop_running = False
            elif key == 7340032: # f1
                loop_current_mode = painting_mode.copy()
            elif key == 7405568: # f2
                loop_current_mode = keyboard_mode.copy()
            elif key == 2490368: # up
                loop_current_mode.thres_down = np.clip(loop_current_mode.thres_down + 1, 0, 500)
            elif key == 2621440: # down
                loop_current_mode.thres_down = np.clip(loop_current_mode.thres_down - 1, 0, 500)
            elif key == 2424832: # left
                loop_current_mode.thres_up = np.clip(loop_current_mode.thres_up - 1, 0, 500)
            elif key == 2555904: # right
                loop_current_mode.thres_up = np.clip(loop_current_mode.thres_up + 1, 0, 500)
            elif key == 2162688: # page up
                loop_current_mode.smoothing = np.clip(loop_current_mode.smoothing + 1, 1, 10)
            elif key == 2228224: # page down
                loop_current_mode.smoothing = np.clip(loop_current_mode.smoothing - 1, 1, 10)
            pass

            pass # while loop_running
        pass # with hand_landmarker
    return

TSmoothVal = TypeVar('TSmoothVal')
smooth_dict: Dict[str, List[TSmoothVal]] = {}
def smooth(key: str, val: 'TSmoothVal') -> 'TSmoothVal':
    global smooth_dict
    pool = smooth_dict.get(key) or []
    pool = pool[loop_current_mode.smooth_frame:]
    pool = [*pool, val]
    
    return

hand_landmarker_result: Optional[Tuple[int, HandLandmarkerResult]] = None
def callback(results: HandLandmarkerResult, image, timestamp:int):
    global hand_landmarker_result
    hand_landmarker_result = (timestamp, results)
    pass

def main() -> int:
    try:
        loop()
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        dispose()
        pass
        

if __name__ == "__main__":
    sys.exit(main())