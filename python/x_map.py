import numba
import numpy as np


# Set the number of threads
numba.set_num_threads(2)

@numba.jit(nopython=True, parallel=True, cache=True, error_model="numpy")
def compute_x_map_from_time_map(
    time_map: np.ndarray, x_map_width: int, t_px_scale: int, X_OFFSET: int, num_scanlines: int
):
    """Create an X-Map (y, t -> x) from a time map (x, y -> t).

    To create the X-Map, we perform a search for the optimal x-coordinate for each t-coordinate,
    akin to the epipolar search in stereo vision.

    All x values in the X-Map will be offset by X_OFFSET, so that x=0 starts at X_OFFSET.
    """

    x_map = np.zeros((time_map.shape[0], x_map_width), dtype=np.int16)

    # when matching, disregard candidates with more than time of two scanlines difference:
    # this is important at the top and bottom of the projector image, where the time map
    # may not be defined for the full width of the projector
    max_t_diff = 2 / num_scanlines

    t_diffs = np.zeros((time_map.shape[0], x_map_width), dtype=np.float32)

    for y in numba.prange(x_map.shape[0]):
        for t_coord in range(x_map.shape[1]):
            # compute optimal x for each t

            t = t_coord / t_px_scale

            # TODO 0-value is not defined - but also the timestamp at the first pixel
            # to fix, add something akin X_OFFSET to the proj time map
            if t == 0:
                continue

            min_t_diff = np.inf
            min_t_diff_x = -1

            for x in range(time_map.shape[1]):
                t_map = time_map[y, x]
                if t_map == 0:
                    continue

                t_diff = np.abs(t - t_map)
                if t_diff < min_t_diff:
                    min_t_diff = t_diff
                    min_t_diff_x = x

            if min_t_diff_x != -1:
                if min_t_diff <= max_t_diff:
                    x_map[y, t_coord] = min_t_diff_x + X_OFFSET
                    t_diffs[y, t_coord] = min_t_diff

    return x_map, t_diffs



                    ##### Numba - GPU
# import numba
# from numba import cuda
# import numpy as np
# import math

# # Set the number of threads for CPU 
# # numba.set_num_threads(1)

# @cuda.jit
# def compute_x_map_from_time_map(time_map, x_map_width, t_px_scale, X_OFFSET, num_scanlines, x_map, t_diffs):
#     # Get the current thread index
#     y = cuda.grid(1)

#     # Make sure the current thread index does not exceed the size of the array
#     if y < x_map.shape[0]:
#         max_t_diff = 2 / num_scanlines
#         for t_coord in range(x_map.shape[1]):
#             t = t_coord / t_px_scale

#             if t == 0:
#                 continue

#             min_t_diff = 1e10  # Equivalent to np.inf
#             min_t_diff_x = -1

#             for x in range(time_map.shape[1]):
#                 t_map = time_map[y, x]
#                 if t_map == 0:
#                     continue

#                 # Use math.fabs instead of np.abs for GPU compatibility
#                 t_diff = math.fabs(t - t_map)
#                 if t_diff < min_t_diff:
#                     min_t_diff = t_diff
#                     min_t_diff_x = x

#             if min_t_diff_x != -1 and min_t_diff <= max_t_diff:
#                 x_map[y, t_coord] = min_t_diff_x + X_OFFSET
#                 t_diffs[y, t_coord] = min_t_diff


# if __name__ == "__main__":
#     
#     time_map = np.random.rand(100, 200).astype(np.float32)  # Input time map
#     x_map_width = 150
#     t_px_scale = 10
#     X_OFFSET = 0
#     num_scanlines = 200

#     # Allocate GPU memory for x_map and t_diffs
#     x_map_gpu = cuda.device_array((time_map.shape[0], x_map_width), dtype=np.int16)
#     t_diffs_gpu = cuda.device_array((time_map.shape[0], x_map_width), dtype=np.float32)

#     # Configure the number of threads and blocks
#     threads_per_block = 32
#     blocks_per_grid = (time_map.shape[0] + (threads_per_block - 1)) // threads_per_block

#     # Launch the kernel on the GPU
#     compute_x_map_from_time_map[blocks_per_grid, threads_per_block](
#         time_map, x_map_width, t_px_scale, X_OFFSET, num_scanlines, x_map_gpu, t_diffs_gpu
#     )

#     # Wait for the GPU to finish processing
#     cuda.synchronize()

#     # Copy the results from GPU to host
#     x_map = x_map_gpu.copy_to_host()
#     t_diffs = t_diffs_gpu.copy_to_host()

#     # Print the result
#     print("X Map:\n", x_map)
#     print("Time Differences:\n", t_diffs)
