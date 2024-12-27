from metavision_sdk_ui import EventLoop

from bias_events_iterator import NonBufferedBiasEventsIterator
from depth_reprojection_processor import DepthReprojectionProcessor, RuntimeParams

import click
import sys

import json
import socket
import threading
import struct

# ------------------------------------ socket --------------------------------------
class Socket():
    CONFIG_FILE:str
    ADDR: "socket._Address"
    STRUCT: struct.Struct

    def __init__(self, CONFIG_FILE):
        self.CONFIG_FILE = CONFIG_FILE
        self.setup_socket()

    def setup_socket(self):
        with open(self.CONFIG_FILE) as f:
                config = json.load(f)

        self.ADDR = (config["addr"], config["port"])
        self.STRUCT = struct.Struct(config["struct"])

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        

    def send_data(self, position):          
        packed_data = self.STRUCT.pack(*position)
        self.sock.sendto(packed_data, self.ADDR) 

# --------------------------------------------------------------------------


def project_events(bias, input, params, delta_t, ev_processor):
    mv_iterator = NonBufferedBiasEventsIterator(input_filename=input, delta_t=delta_t, bias_file=bias)
    # mv_iterator = BiasEventsIterator(input_filename=cli_params["input"], delta_t=8000, bias_file=cli_params["bias"])
    cam_height_reader, cam_width_reader = mv_iterator.get_size()  # Camera Geometry
    print('cam_height_reader', cam_height_reader, params.camera_height)
    print('cam_width_reader', cam_width_reader, params.camera_width)
    assert cam_height_reader == params.camera_height
    assert cam_width_reader == params.camera_width

    for evs in mv_iterator:
        with ev_processor.stats_printer.measure_time("main loop"):
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            if not len(evs):
                continue

            ev_processor.process_events(evs)

            if ev_processor.should_close():
                sys.exit(0)


@click.command()
@click.option("--projector-width", default=1080, help="Projector width in pixels", type=int)
@click.option("--projector-height", default=1920, help="Projector height in pixels", type=int)
@click.option("--projector-fps", default=60, help="Projector fps", type=int)
@click.option(
    "--projector-time-map",
    help="Path to calibrated projector time map file (*.npy). If left empty, a linear time map will be used.",
    type=click.Path(),
)
@click.option("--z-near", default=0.3, help="Minimum depth [m] for visualization", type=float)
@click.option("--z-far", default=0.7, help="Maximum depth [m] for visualization", type=float)
@click.option(
    "--calib",
    help="path to yaml file with camera and projector intrinsic and extrinsic calibration",
    type=click.Path(),
    required=True,
)
@click.option("--bias", help="Path to bias file, only required for live camera", type=click.Path())
@click.option(
    "--input", help="Either a .raw, .dat file for prerecordings. Don't specify for live capture.", type=click.Path()
)
@click.option("--loop-input", help="Loop input file", is_flag=True)
@click.option(
    "--no-frame-dropping", help="Process all events, even when processing lags behind the event stream", is_flag=False
)
@click.option(
    "--camera-perspective",
    help="By default the depth is rendered from the projector's perspectiev. Enable this flag to render from the camera perspective instead.",
    is_flag=True,
)
def main(bias, input, loop_input, **cli_params):
    # TODO remove these static values, retrieve from event stream
    params = RuntimeParams(camera_width=1280, camera_height=720, **cli_params)

    EV_PACKETS_PER_FRAME = 1
    delta_t = 1e6 / params.projector_fps / EV_PACKETS_PER_FRAME

    print(f"Using delta_t={delta_t:.2f} us to process {EV_PACKETS_PER_FRAME} ev packets per projector frame.")
    print(f"If you see frame drops, try reducing EV_PACKETS_PER_FRAME to 1. This may increase latency.")

    # ------ Add sock --------
    CONFIG_FILE = "../localhost/config.json"
    socket = Socket()

    # ------------------------


    with DepthReprojectionProcessor(params) as ev_processor:
        while True:
            print(f"============== check data list =================> hahahahah")
            project_events(bias, input, params, delta_t, ev_processor)
            _data = ev_processor.get_send_data()    
            print(f"============== check data list =================> {_data}")
            if loop_input:                
                ev_processor.reset()                
                # socket.send_data(_data[-1])
            else:
                break


if __name__ == "__main__":
    main()