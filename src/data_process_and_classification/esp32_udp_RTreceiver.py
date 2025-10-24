# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:49:54 2025

@author: AXILLIOS
"""
'''
files =
good_1
good_2
good_3
mid_1
mid_2
mid_3
bad_1
bad_2
bad_3
test_good_1
test_good_2
test_good_3
test_mid_1
test_mid_2
test_mid_3
test_bad_1
test_bad_2
test_bad_3
'''
# -*- coding: utf-8 -*-
"""
ESP32 UDP Receiver with CSV logging
Author: AXILLIOS
Updated by: Teo + ChatGPT
"""

import socket, os

# === Configuration ===
OUTPUT_DIR = "0_RAW/series_of_experiments_2/9.71_Hz_sampling/TESTING"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_rate = 9.71
file = 'evl_good'
exp = '3'
csv_filename = os.path.join(OUTPUT_DIR, f"TS_{file}_{exp}_{sample_rate}.csv")

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345    # Must match the ESP32 UDP port
BUFFER_SIZE = 256

# === Initialize UDP Socket ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(20.0)

print(f"[INFO] Listening for UDP packets on {UDP_PORT}...")
print(f"[INFO] Saving data to: {csv_filename}")

try:
    with open(csv_filename, 'w', newline='') as f:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            message = data.decode(errors="ignore").strip()

            # Respond to handshake message
            if message == "ESP32_HANDSHAKE":
                sock.sendto(b"PC_ACK", addr)
                print("[HANDSHAKE] Replied to ESP32.")
                continue

            # Print and save message
            print(message)
            f.write(message + "\n")
            f.flush()

except KeyboardInterrupt:
    print("\n[STOPPED] User interrupted.")
finally:
    sock.close()
    print("[CLOSED] UDP socket closed.")