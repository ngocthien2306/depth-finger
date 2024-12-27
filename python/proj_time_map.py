import numpy as np
import cv2
from dataclasses import dataclass


def generate_linear_projector_time_map(proj_width: int, proj_height: int, scan_upwards: bool) -> np.ndarray:
    # x and y coordinates of projector pixels
    ys, xs = np.mgrid[0:proj_height, 0:proj_width]

    if scan_upwards:
        # invert y axis to scan from bottom to top
        ys = ys[::-1]

    ## Apply a custom scan pattern, a sinusoidal wave for y-axis:
    ys = np.sin(ys * np.pi / proj_height) * proj_height / 2 + proj_height / 2

    # scan in x direction (right) first, than the y direction (determined by scan_upwards)
    pixel_indeces = xs * proj_height + ys

    projector_time_map = pixel_indeces / (proj_width * proj_height)

    return projector_time_map.astype(np.float32)


##
def generate_nonlinear_projector_time_map(proj_width: int, proj_height: int, scan_upwards: bool) -> np.ndarray:
    """
    Generates a non-linear projector time map using a sinusoidal scan pattern along the y-axis.
    
    :param proj_width: Width of the projector.
    :param proj_height: Height of the projector.
    :param scan_upwards: If True, the scan will be inverted along the y-axis.
    :return: A non-linear projector time map as a 2D numpy array.
    """
    # x and y coordinates of projector pixels
    ys, xs = np.mgrid[0:proj_height, 0:proj_width]

    if scan_upwards:
        # Invert y-axis to scan from bottom to top
        ys = ys[::-1]

    # Apply a custom non-linear scan pattern using a sine function for the y-axis
    # The sine function modulates the y-coordinates to create a non-linear mapping
    ys_nonlinear = np.sin(ys * np.pi / proj_height) * proj_height / 2 + proj_height / 2

    # Combine x and (nonlinear) y to create pixel indices in a raster scan order
    # The x axis remains linear, while y has the non-linear transformation
    pixel_indeces = xs * proj_height + ys_nonlinear

    # Normalize the indices to create a time map where values range from 0 to 1
    projector_time_map = pixel_indeces / (proj_width * proj_height)

    return projector_time_map.astype(np.float32)


def remap_proj_time_map(cam_proj_maps, proj_time_map, border_mode) -> np.ndarray:
    return cv2.remap(
        proj_time_map,
        cam_proj_maps.projector_mapx,
        cam_proj_maps.projector_mapy,
        cv2.INTER_NEAREST,
        border_mode,
    )


@dataclass
class ProjectorTimeMap:
    projector_time_map_rectified: np.ndarray

    @staticmethod
    def from_calib(calib_params, cam_proj_maps, scan_upwards=True, remap_border_mode=cv2.BORDER_REPLICATE):
        # scale_factor = 0.5  # Increase the time map scale (slower scan)
        # projector_time_map = generate_linear_projector_time_map(
        #     calib_params.projector_width, calib_params.projector_height, scan_upwards
        # )
        
        ##
        projector_time_map = generate_nonlinear_projector_time_map(
            calib_params.projector_width, calib_params.projector_height, scan_upwards
        )
        # projector_time_map *= scale_factor
        projector_time_map_rectified = remap_proj_time_map(
            cam_proj_maps, projector_time_map, border_mode=remap_border_mode
        )
        return ProjectorTimeMap(projector_time_map_rectified)

    @staticmethod
    def from_file(proj_time_map_path):
        projector_time_map_rectified = np.load(proj_time_map_path)
        return ProjectorTimeMap(projector_time_map_rectified)
