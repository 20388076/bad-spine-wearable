# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:49:54 2025

@author: AXILLIOS
"""

import socket

UDP_IP = "0.0.0.0" # Listen on all available interfaces
UDP_PORT = 12345 # Must match udpPort in your ESP32 code
BUFFER_SIZE = 256   # Buffer size for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

try:
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE) # Buffer size
        message = data.decode(errors="ignore").strip()
        print(f"{message}")
except KeyboardInterrupt:
        print("\nStopped by user.")
finally:
    sock.close()
    print("Socket closed.")