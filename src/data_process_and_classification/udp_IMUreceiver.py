"""
UDP Receiver for ESP32 Sensor Data
Receives accelerometer and gyroscope data from ESP32 via UDP and saves to CSV file.

Usage:
    python udp_receiver.py

Configuration:
    - Change UDP_PORT if you changed it in the ESP32 code
    - CSV file will be saved with timestamp in filename
"""

import socket
import csv
import datetime
import os
import signal
import sys

# ===== CONFIGURATION =====
UDP_IP = "0.0.0.0"  # Listen on all network interfaces
UDP_PORT = 12345    # Must match the port in ESP32 code
BUFFER_SIZE = 256   # Buffer size for receiving data

# Create output directory if it doesn't exist
OUTPUT_DIR = "0_RAW/series_of_experiments_2/9.71_Hz_sampling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
'''
Classes =
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
sample_rate = 9.71
Class = 'testing movements'
exp = 'test'
# Generate filename with timestamp
#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(OUTPUT_DIR, f"{Class}_{exp}_{sample_rate}.csv")

# Global variables
sock = None
csv_file = None
csv_writer = None
packet_count = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*60)
    print(f"📊 Data Collection Summary")
    print("="*60)
    print(f"Total packets received: {packet_count}")
    print(f"Data saved to: {csv_filename}")
    print("="*60)
    
    if csv_file:
        csv_file.close()
    if sock:
        sock.close()
    
    print("\n✅ Receiver stopped gracefully. Goodbye!\n")
    sys.exit(0)

def main():
    global sock, csv_file, csv_writer, packet_count
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*60)
    print("🚀 ESP32 Sensor Data Receiver - UDP Mode")
    print("="*60)
    print(f"📡 Listening on: {UDP_IP}:{UDP_PORT}")
    print(f"💾 Saving data to: {csv_filename}")
    print("\n⏳ Waiting for data from ESP32...")
    print("   (Press Ctrl+C to stop)\n")
    print("-"*60)
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Open CSV file
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write CSV header
    csv_writer.writerow(['t (ms)', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z'])
    csv_file.flush()
    
    print("✅ Ready! Listening for incoming data...\n")
    
    # Main receiving loop
    try:
        while True:
            # Receive data from ESP32
            data, addr = sock.recvfrom(BUFFER_SIZE)
            
            # Decode and parse data
            data_str = data.decode('utf-8').strip()
            
            # Parse CSV format: timestamp,ax,ay,az,gx,gy,gz
            try:
                values = data_str.split(',')
                if len(values) == 7:
                    # Write to CSV
                    csv_writer.writerow(values)
                    csv_file.flush()  # Ensure data is written immediately
                    
                    packet_count += 1
                    
                    # Print progress every 10 packets
                    if packet_count % 10 == 0:
                        print(f"📦 Received {packet_count} packets | Latest: {data_str}")
                    
            except Exception as e:
                print(f"⚠️  Error parsing data: {e} | Raw data: {data_str}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()
