import cv2
import numba
import numpy as np
from dataclasses import dataclass

# Set the number of threads
numba.set_num_threads(2)

@numba.jit(nopython=True, parallel=True, cache=True, error_model="numpy")
def clip_normalize_uint8_depth_frame(depth_frame: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """function to clip a depth map to min and max arguments, normalize to [0,255] and change dtype to np.uint8"""
    height, width = depth_frame.shape
    frame = np.zeros((height, width), dtype=np.uint8)
    frame_1000 = np.zeros((height, width), dtype=np.int16)
    min_value, max_value = np.float32(min_value), np.float32(max_value)  # convert min_value and max_value to float32
    range_value = max_value - min_value

    for i in numba.prange(height):
        for j in range(width):
            val = depth_frame[i, j]
            val_1000 = depth_frame[i, j]
            if val != 0:
                val = max(min(val, max_value), min_value)
                val = (val - min_value) / range_value * 255
                val_1000 = max(min(val_1000, max_value), min_value)
                val_1000 = (val_1000 - min_value) / range_value * 1000
            frame[i, j] = np.uint8(val)
            frame_1000[i, j] = np.int16(val_1000)

    # return frame
    return frame, frame_1000

@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_white_mask(frame, norm_frame):
    height, width = norm_frame.shape
    for i in numba.prange(height):
        for j in range(width):
            if norm_frame[i, j] == 0:
                frame[i, j, :] = 255
    return frame


def save_color_map_to_txt(color_frame: np.ndarray, filename: str) -> None:
    """Save the matrix representation of the colorized depth map to a txt file."""
    # Reshape the matrix for easier saving (flattening the color channels)
    reshaped_frame = color_frame.reshape(-1, color_frame.shape[2])  # shape: (height * width, 3)
    
    # Save to a text file
    np.savetxt(filename, reshaped_frame, fmt='%d', delimiter=',', header="R,G,B")



def generate_color_map(norm_frame: np.ndarray) -> None:
    """Generate a colored visualization from the depth map"""
    # cp_normframe = norm_frame.copy() * 2 - 255  # v2
    # cp_normframe_mask = norm_frame < 231 #220
    # cp_normframe[cp_normframe_mask] = cp_normframe[cp_normframe_mask] // 2
    # frame = cv2.applyColorMap(cp_normframe, cv2.COLORMAP_HOT)   # v2

    frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_TURBO)

    # zero depth represents no depth value, to still be able to find depth at that pixel
    # at the next iteration if color map is projected back, undefined depth values are set to white
    # to create new events
    frame = apply_white_mask(frame, norm_frame)

    # Save the color frame matrix to a txt file
    # filename = 'matrix_representation.txt'
    # save_color_map_to_txt(frame, filename)

    return frame






@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True, error_model="numpy")
def disparity_to_depth_rectified(disparity, P1):
    """Function for simplified calculation of depth from disparity.
    This calculation neglects the change in depth caused be the rotation of the rectification.
    If this rotation is small, the error is small."""

    height, width = disparity.shape
    depth = np.zeros((height, width), dtype=np.float32)

    for i in numba.prange(height):
        for j in range(width):
            val = disparity[i, j]
            if val == 0:
                depth[i, j] = 0.0
            else:
                depth[i, j] = max(P1[0, 3] / val, 1e-9)

    return depth


@dataclass
class DisparityToDepth:
    stats: "StatsPrinter"
    calib_params: "CalibParams"
    calib_maps: "CamProjMaps"
    z_near: float
    z_far: float
    DEPTH_DIFF = None
    DEPTH_DIFF_1000 = None
    _REGION_PAD = 10
    PROJ_REGION = None
    INIT_REGION = None
    imageList = []

    dilate_kernel = np.ones((7, 7), dtype=np.uint8)

    def get_proj_region(self):
        return self.PROJ_REGION

    def remap_rectified_disp_map_to_proj(self, rectified_disp_map):
        # if projector view is active, dilate pixels
        # projector view is the depth maps from the projectors perspective and with the projectors resolution.
        # Two problems, first, the resolution of the depth map is lower than the projector resolution.
        # Secondly, due how the dispraity search for the projectors perspective works, multiple camera pixels
        # can be mapped to the same projector pixel, while other projector pixel will be left out.
        # For a dense depth map from the projectors point of view, the pixels are dilated.

        # TODO perf this gets faster with a larger kernel.. why?
        with self.stats.measure_time("dilate"):
            disp_map = cv2.dilate(rectified_disp_map, self.dilate_kernel)

        with self.stats.measure_time("remap disp"):
            disp = cv2.remap(
                disp_map,
                map1=self.calib_maps.disp_proj_mapxy_i16,
                map2=None,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

        return disp
    
    
    # ----------> crop ergion <------------------
    def crop_region(self, frame):
        thresholdValue = 70
        area, x, y, w, h = 0, 0, 0, 0, 0
        _, gray = cv2.threshold(frame, thresholdValue, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        gray = cv2.erode(gray, kernel) 
        gray = cv2.dilate(gray, kernel) 
        # gray = cv2.cvtColor(accumulatedImage, cv2.COLOR_BGR2GRAY)
        # Find conters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ix, iy, iw, ih = cv2.boundingRect(contour)
            if area < iw * ih and iw * ih > 100000:
                area = iw * ih
                x, y, w, h = ix, iy, iw, ih


        return (x, y, w, h), gray
    # ----------> crop ergion <------------------

    def reset_depth_diff(self):
        self.DEPTH_DIFF = None


    def colorize_depth_from_disp(self, disp_map: np.ndarray) -> np.ndarray:
        # NOTE: This depth calculatoin is quick but not correct. It does not take into account the
        # change of depth during the rotation back from the rectified coordinate system to the
        # unrectified coordinate system.
        with self.stats.measure_time("d2d_rect"):
            depth_map_f32 = disparity_to_depth_rectified(
                disp_map,
                self.calib_maps.P2,
            )
            # Save the depth map to a text file
            # np.savetxt('depth_map_f32.txt', depth_map_f32, fmt='%d')

        with self.stats.measure_time("clip_norm"):
            depth_map_u8, depth_map_1000 = clip_normalize_uint8_depth_frame(depth_map_f32, min_value=self.z_near, max_value=self.z_far)

        # -----------------------------------------------------------        

        if self.DEPTH_DIFF is None:
            depth_max = np.zeros_like(depth_map_u8)
            depth_max_1000 = np.zeros_like(depth_map_1000)            
            
            (x, y, w, h), gray = self.crop_region(depth_map_u8)                        
                        
            map_mask = np.argwhere(gray > 100)   

            min_val = np.min(depth_map_u8[tuple(np.transpose(map_mask))])
            max_val = np.max(depth_map_u8[tuple(np.transpose(map_mask))])         

            min_val_1000 = np.min(depth_map_1000[tuple(np.transpose(map_mask))])
            max_val_1000 = np.max(depth_map_1000[tuple(np.transpose(map_mask))])            
            
            depth_max[tuple(np.transpose(map_mask))] = max_val
            depth_max_1000[tuple(np.transpose(map_mask))] = max_val_1000            

            self.DEPTH_DIFF = depth_max - depth_map_u8
            self.DEPTH_DIFF_1000 = depth_max_1000 - depth_map_1000
            
            x_max = np.max(tuple(np.transpose(map_mask))[0]) - self._REGION_PAD
            x_min = np.min(tuple(np.transpose(map_mask))[0]) + self._REGION_PAD

            y_max = np.max(tuple(np.transpose(map_mask))[1]) - self._REGION_PAD
            y_min = np.min(tuple(np.transpose(map_mask))[1]) + self._REGION_PAD
            self.PROJ_REGION = [(y_min, x_min), (y_max, x_max)]          
                           

        depth_map_u8 = depth_map_u8 + self.DEPTH_DIFF
        depth_map_1000 = depth_map_1000 + self.DEPTH_DIFF_1000
        # depth_map_u8 = (depth_map_u8*self.DEPTH_DIFF).astype(np.uint8)

        depth_map_u8 = np.clip(depth_map_u8, 0, 255)        
        depth_map_1000 = np.clip(depth_map_1000, 0, 999)        
        
                
        # -----------------------------------------------------------
        # evs_frame => rgb_frame => crop_rgb_frame

        with self.stats.measure_time("color_map"):
            frame = generate_color_map(depth_map_u8)

        # crop the frame as the rectangle region detection
        # print(f"========== REGION =============> {self.PROJ_REGION}")
        [(x_min, y_min), (x_max, y_max)] = self.PROJ_REGION
        # [(x_min, y_min), (x_max, y_max)] = [(512, 160), (742, 570)]        
        
        crop_frame = np.ones_like(frame) * 255
        crop_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]

        crop_depth_map_u8  = np.zeros_like(depth_map_u8)
        crop_depth_map_u8[y_min:y_max, x_min:x_max] = depth_map_u8[y_min:y_max, x_min:x_max]

        crop_depth_map_1000  = np.zeros_like(depth_map_1000)
        crop_depth_map_1000[y_min:y_max, x_min:x_max] = depth_map_1000[y_min:y_max, x_min:x_max]

        
        return crop_frame, crop_depth_map_1000
        # return frame, depth_map_u8 
        # return frame, depth_map_u8_temp  
    
