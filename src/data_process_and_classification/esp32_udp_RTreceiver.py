"""
Created on Wed Oct 22 18:49:54 2025

@author: AXILL'IOS
"""
import socket, os
'''
files =
{classifier_name}_good_1
{classifier_name}_good_2
{classifier_name}_good_3
{classifier_name}_mid_1
{classifier_name}_mid_2
{classifier_name}_mid_3
{classifier_name}_bad_1
{classifier_name}_bad_2
{classifier_name}_bad_3
'''
# === User input ===

classifier_names = ['DT','RF']
classifier_name = classifier_names[1] # 0 for DT and 1 for RF
scale = ['Non scaled', 'Scaled']
normalization = scale[0] # 0 Non scaled data and 1 scaled
sample_rate = 9.71
file = 'ts'
exp = '0'
# === Configuration ===
OUTPUT_DIR = f"0_RAW/series_of_experiments_2/9.71_Hz_sampling/TESTING_{classifier_name}/{normalization}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
if scale == 0:
    base_filename = f"{classifier_name}_{file}_{exp}_{sample_rate}nsc.csv"
else:
    base_filename = f"{classifier_name}_{file}_{exp}_{sample_rate}sc.csv"
    
csv_filename = os.path.join(OUTPUT_DIR, base_filename)
# === Auto-increment if file exists ===
counter = 1
while os.path.exists(csv_filename):
    name, ext = os.path.splitext(base_filename)
    csv_filename = os.path.join(OUTPUT_DIR, f"{name}({counter}){ext}")
    counter += 1

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
