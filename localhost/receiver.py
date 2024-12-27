import socket
import json
import time
import struct
import numpy as np

# Load the configuration file

CONFIG_FILE = "config.json"

with open(CONFIG_FILE) as f:
    config = json.load(f)

ADDR:"socket._Address" = (config["host"], config["port"])
STRUCT:struct.Struct = struct.Struct(config["struct"])


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
## Set the socket to reuse the address and port
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
## Reuse the port if available
if hasattr(socket, "SO_REUSEPORT"):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
sock.setblocking(False)


# Bind the socket to the address
sock.bind(ADDR)

list_points = []

# Loop of receiving data
prev_time = time.perf_counter()
num_100 = 0
try:
    while True:
        # Receive data from the sender
        try:            
            data = sock.recv(STRUCT.size)
            list_points.append((time.perf_counter() - prev_time)*1000)
            if len(list_points) > 101:
                print(f"===== time interval 100 samples ====> {np.average(list_points[-100:])}")
                
            if len(list_points) > 100000:
                list_points = []
            prev_time = time.perf_counter()

        except BlockingIOError:
            continue
        
        # print(f"Received data: {data}")
        unpacked_data = STRUCT.unpack(data)
        
        print(f"Unpacked data: {unpacked_data}")        
finally:
    sock.close()
