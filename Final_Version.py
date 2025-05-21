from __future__ import print_function
import csv
import cv2
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import multiprocessing
import inputs  # Importing the gamepad library
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, GpsLocationChanged
from olympe.messages.gimbal import set_target
from olympe.messages.camera import set_zoom_target
from olympe.messages.camera2.Command import SetZoomTarget
from olympe.enums.camera import availability
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
from pynput import keyboard
import numpy as np
import math

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

# Gets the drone IP address, and the port so the code can connect to it.
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

# Basic Water Bottle height (19.3 cm)
# The test will be done with the maximum zoom distance, and at a set distance of 30 cm.
# The 1.0 zoom test was done at 30 cm from the drone.
# The 3,0 zoom was tested at a distance of 101.5 cm from the drone.

# Finds the model which is saved as a .pt file, and saves it to a variable.
Model_Path = r"/home/labpc/Downloads/Water_Can_Version2.pt"
UAV_Model_Path = r"/home/labpc/Downloads/UAVVaste_Test_1t.pt"
OG_Model_Path = r"/home/labpc/Downloads/yolov8n-seg.pt"
Bottle_Path = r"/home/labpc/Downloads/Bottle.pt"
New_Test_Path = r"/home/labpc/Downloads/best(9).pt"
Bottle_Metal_Can_Test_Path = r"/home/labpc/Downloads/best(10).pt"
Bottle_Metal_Can_Test_Version_2_Path = r"/home/labpc/Downloads/best(11).pt"
model = YOLO(Bottle_Metal_Can_Test_Path)

# Define debounce delay in seconds (adjust as needed)
DEBOUNCE_DELAY = 0.3

# Define a global DataFrame to store GPS data
gps_data_df = pd.DataFrame(columns=['Timestamp', 'Latitude (Degrees)', 'Longitude (Degrees)', 'Altitude (ft)', 'Objects', 'Confidence', 'Pixel Count', 'DSF', 'Pixel Height', 'True Height (cm)', 'Water Bottle Mass (g)'])

# Average focal Point (pixels) from the camera matrix
fy = 777.69

# Pixel Scaling factors for each type of product which is in cm per pixel
# This was tested during the calibration stage of the experiment
PSFw = 0.036981132
PSFm = 0.033424658

# This is for the tested distance which would be used to calculate the distance scaling factor
# The distance is in centimeters
TD = 30.0

# Define the key to check
key_to_check = 'g'  # Example: 'a' key

# This defines the initial altitude for the system
Initial_Altitude = 0

# Outer radius of a PET water bottle (cm)
rw = 5.5

# Thickness of wall (cm)
tw = 0.03

# PET density (g/cm ^3)
Densityw = 1.38

# Outer radius of a commercial Aluminum can (cm)
rm = 5.8

# Thickness of wall for a commercial Aluminum can (cm)
tm = 0.02

# Aluminum Can Material density (g/cm ^3)
Densitym = 2.7


# Dictionary to map class IDs to object names
class_names = {
    0: 'Metal Can',
    1: 'Water Bottle',
    # Add other class IDs and names as needed
}
class StreamingExample:
    def __init__(self):
        # Create the olympe.Drone object from its IP address.
        self.drone = olympe.Drone(DRONE_IP)
        # Creates a temporary for storing output files.
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        # Prints the path for the temporary directory.
        print(f"Olympe streaming example output dir: {self.tempd}")
        # Existing initialization code...
        self.monitor_thread = None  # Add this line to store the monitor thread reference
        # Opens an empty list to store h264 statistics.
        self.h264_frame_stats = []
        # Opens a temporary directory for writing video stream statistics.
        self.h264_stats_file = open(os.path.join(self.tempd, "h264_stats.csv"), "w+")
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ["fps", "bitrate"]
        )
        self.h264_stats_writer.writeheader()
        self.frame_queue = multiprocessing.Queue()
        self.processed_frame_queue = multiprocessing.Queue()
        self.processes = []
        self.lock = threading.Lock()  # Add this line to initialize the lock
        self.running = multiprocessing.Event()
        self.running.set()

    def start(self):
        # Connect to drone
        assert self.drone.connect(retry=3)

        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"

        # Start the monitor thread
        self.monitor_thread = self.start_monitor_thread()  # Capture the thread reference

        # You can record the video stream from the drone if you plan to do some
        self.drone.streaming.set_output_files(
            video=os.path.join(self.tempd, "streaming.mp4"),
            metadata=os.path.join(self.tempd, "streaming_metadata.json"),
        )

        # Setup your callback functions to do some live video processing
        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            h264_cb=self.h264_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start the monitor thread
        self.monitor_thread = self.start_monitor_thread()  # Capture the thread reference

        # Start video streaming
        self.drone.streaming.start()

        # Start multiple processing threads
        num_processes = multiprocessing.cpu_count()
        self.processes = [multiprocessing.Process(target=self.yuv_frame_processing, args=(self.running,))
                          for _ in range(num_processes)]
        for p in self.processes:
            p.start()

        # Start the video output processing thread
        self.output_thread = threading.Thread(target=self.process_video_output)
        self.output_thread.start()

        # Monitor thread initialization
        self.monitor_thread = self.start_monitor_thread()  # Ensures monitor runs independently

    def stop(self):
        self.running.clear()
        for p in self.processes:
            p.join()

        # Ensure monitor thread is stopped and joined
        if self.monitor_thread:
            self.stop_monitor()
            self.monitor_thread.join()

        self.output_thread.join()
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()
        self.h264_stats_file.close()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        info = yuv_frame.info()
        height, width = info["raw"]["frame"]["info"]["height"], info["raw"]["frame"]["info"]["width"]
        yuv_data = yuv_frame.as_ndarray()
        self.frame_queue.put((yuv_data, height, width))
        yuv_frame.unref()

    def yuv_frame_processing(self, running):
        while running.is_set():
            try:
                yuv_data, height, width = self.frame_queue.get(timeout=0.1)
                yuv_data = yuv_data.reshape((height * 3 // 2, width))
                bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
                results = model(bgr_frame)
                annotated_frame = results[0].plot()
                self.processed_frame_queue.put(annotated_frame)
            except cv2.error as e:
                #print("We have a cv2 error.")
                a = 1
            except Exception as e:
                #print("We have a unexpected error.")
                a = 2


    def process_video_output(self):
        while self.running.is_set():
            try:
                frame = self.processed_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            cv2.imshow('Machine Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running.clear()
                break

        cv2.destroyAllWindows()

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def Frame_Check(self):
        try:
            # Get frame data from queue
            yuv_data, height, width = self.frame_queue.get(timeout=0.1)

            # Reshape and convert YUV to BGR
            yuv_data = yuv_data.reshape((height * 3 // 2, width))
            bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)

            # Run inference
            results = model(bgr_frame)

            return results

        except queue.Empty:
            # Handle empty queue case
            print("Queue was empty")
            return []  # Return an empty list instead of None
        except cv2.error as e:
            # Handle OpenCV errors
            print(f"OpenCV error: {e}")
            return []
        except Exception as e:
            # Handle other exceptions
            print(f"Unexpected error: {e}")
            return []


    def h264_frame_cb(self, h264_frame):
        """
        This function will be called by Olympe for each new h264 frame.

            :type yuv_frame: olympe.VideoFrame
        """

        # Get a ctypes pointer and size for this h264 frame
        frame_pointer, frame_size = h264_frame.as_ctypes_pointer()

        # For this example we will just compute some basic video stream stats
        # (bitrate and FPS) but we could choose to resend it over another.
        # interface or to decode it with our preferred hardware decoder.

        # Compute some stats and dump them in a csv file
        info = h264_frame.info()
        frame_ts = info["ntp_raw_timestamp"]
        if not bool(info["is_sync"]):
            while len(self.h264_frame_stats) > 0:
                start_ts, _ = self.h264_frame_stats[0]
                if (start_ts + 1e6) < frame_ts:
                    self.h264_frame_stats.pop(0)
                else:
                    break
            self.h264_frame_stats.append((frame_ts, frame_size))
            h264_fps = len(self.h264_frame_stats)
            h264_bitrate = 8 * sum(map(lambda t: t[1], self.h264_frame_stats))
            self.h264_stats_writer.writerow({"fps": h264_fps, "bitrate": h264_bitrate})

    def Monitor(self):
        global gps_logging_active  # Declare gps_logging_active as global
        global gps_data_df
        global Initial_Altitude
        boxes = None
        while self.running.is_set():
            results = self.Frame_Check()

            if results is not None:
                for result in results:
                    boxes = result.boxes.xyxy

                if boxes is not None and boxes.size(0) > 0:
                    gps_data = self.drone.get_state(GpsLocationChanged)

                    # Extract coordinates, and records time when it happens.
                    timestamp = datetime.now()
                    latitude = gps_data['latitude']
                    longitude = gps_data['longitude']
                    altitude = gps_data['altitude']

                    # Process detection results
                    detected_objects = []
                    pixel_counts = []  # Store the number of pixels for detected objects
                    pixel_height = []
                    Distance_Scalling_Factor = []
                    True_Height = []
                    Water_Bottle_Mass = []
                    for result in results:
                        labels = result.boxes.cls.cpu().numpy()  # Convert tensor to numpy array
                        confidences = result.boxes.conf.cpu().numpy()  # Convert tensor to numpy array

                        # Count pixels in detected object masks
                        if hasattr(result, 'masks') and result.masks.data is not None:
                            for mask in result.masks.data:
                                # Convert to binary mask based on type of `mask`
                                if hasattr(mask, 'cpu'):
                                    binary_mask = mask.cpu().numpy() > 0
                                else:
                                    binary_mask = mask > 0
                                pixel_count = np.sum(binary_mask)  # Count non-zero pixels
                                pixel_counts.append(pixel_count)

                    for i, label in enumerate(labels):
                        class_id = int(label)  # Convert tensor label to integer
                        object_name = class_names.get(class_id, "unknown")  # Get object name
                        detected_objects.append(object_name)

                    # Check if results contain valid data
                    if results and hasattr(results[0], 'masks') and hasattr(results[0].masks, 'data') and hasattr(
                            results[0], 'boxes'):
                        for i, (mask, bbox) in enumerate(zip(results[0].masks.data, results[0].boxes.xyxy)):
                            # Convert mask to binary
                            # Calculate pixel height of the object from bounding box
                            x_min, y_min, x_max, y_max = map(int, bbox)
                            box_width = x_max - x_min
                            box_height = y_max - y_min

                            # Uses the longest side of the detected object as the pixel height.
                            if box_height > box_width:
                                Pixel_Height = box_height
                            else:
                                Pixel_Height = box_height

                            # Current distance of the object from the drone in centimeters
                            Current_disance = 45.0

                            # Separates the calculation process based on the type of detected object

                            # This is for PET water bottles.
                            if class_id == 1:


                                # Distance scaling factor
                                DSF = Current_disance / TD

                                # Adjusted height based on distance scaling factor
                                Pixel_Height = Pixel_Height * DSF

                                # Adds the values to the variable
                                pixel_height.append(Pixel_Height)

                                Distance_Scalling_Factor.append(DSF)

                                # Creates a ratio based on the focal point and measured pixel height
                                PHT = (Pixel_Height / fy) * TD

                                # Converts the height in cm, to a float, so it can be used in the mass calculations.
                                PHT = float(PHT)

                                # Adds the variable to the list
                                True_Height.append(PHT)

                                # assigns a float variable to the exponent
                                n = float(2.0)

                                # Calculates the outer volume of the water bottle
                                Vouter = 3.14 * (rw ** n) * PHT

                                # Calculates the inner volume of the water bottle
                                Vinner = 3.14 * ((rw - tw) ** n) * PHT

                                # Solves for the volume of an empty water bottle based on the thickness of the walls
                                Vwater = Vouter - Vinner

                                # Solves for the mass based on the density of the material
                                MassW = Vwater * Densityw

                                # Adds the water bottle mass to the list
                                Water_Bottle_Mass.append(MassW)

                                print(f"Object {i + 1}:")
                                print(f" - Pixel Height: {pixel_height} pixels")
                                print(f" - Object Height: {PHT} centimeters")
                                print(Vouter)
                                print(Vinner)
                                print(MassW)

                            # This is for aluminum soda cans.
                            if class_id == 0:

                                print(f"Metal Can detected ")

                                # This is for water bottles.

                                # Distance scaling factor
                                DSF = Current_disance / TD

                                # Adjusted focal point based on distance scaling factor
                                Pixel_Height = Pixel_Height * DSF

                                # Adds the values to the variable
                                pixel_height.append(Pixel_Height)

                                Distance_Scalling_Factor.append(DSF)


                                PHT = (Pixel_Height / fy) * TD

                                PHT = float(PHT)

                                True_Height.append(PHT)

                                n = float(2.0)

                                Vouter = 3.14 * (rm ** n) * PHT

                                Vinner = 3.14 * ((rm - tm) ** n) * PHT

                                Vwater = Vouter - Vinner

                                Massm = Vwater * Densitym

                                Water_Bottle_Mass.append(Massm)

                                print(f"Object {i + 1}:")
                                print(f" - Pixel Height: {pixel_height} pixels")
                                print(f" - Object Height: {PHT} centimeters")



                    else:
                        print("No objects detected or invalid results format.")

                    # Prepare the new data as a DataFrame
                    # Adds any variables stored in the list to the dataframe
                    new_data = pd.DataFrame({
                        'Timestamp': [timestamp],
                        'Latitude (Degrees)': [latitude],
                        'Longitude (Degrees)': [longitude],
                        'Altitude (ft)': [altitude],
                        'Objects': [detected_objects],
                        'Confidence': [confidences],
                        'Pixel Count': [pixel_counts],
                        'DSF': [Distance_Scalling_Factor],
                        'Pixel Height': [pixel_height],
                        'True Height (cm)': [True_Height],
                        'Water Bottle Mass (g)': [Water_Bottle_Mass]
                    })

                    # Filter out columns that are empty or entirely NA
                    new_data = new_data.dropna(axis=1, how='all')

                    # Use the lock to ensure thread-safe access to gps_data_df
                    with self.lock:
                        gps_data_df = pd.concat([gps_data_df, new_data], ignore_index=True)

            time.sleep(0.1)

    def start_monitor_thread(self):
        monitor_thread = threading.Thread(target=self.Monitor)
        monitor_thread.start()
        return monitor_thread

    def stop_monitor(self):
        self.running.clear()  # Stops the Monitor loop

    def Zoom(self):
        try:
            # Set the zoom target
            set_zoom_target_command = set_zoom_target(
                cam_id=0,
                control_mode='level',
                target=1.0
            )
            # Max distance is 2.9999840259552 (value for the target).
            zoom_result = self.drone(set_zoom_target_command).wait()

            if zoom_result.success():
                # Print out the confirmed camera properties
                print("set zoom successfully.")
            else:
                print("Failed to retrieve camera zoom information.")
        except Exception as e:
            print(f"Error: {e}")

    def fly(self):
        # Declare gps_data_df as global
        global gps_data_df
        global gps_logging_active
        global Initial_Altitude  # Allows for the variable to be used in other functions

        # Initialize debounce timers for each button
        debounce_tl = time.time()
        debounce_tr = time.time()
        debounce_tul = time.time()
        debounce_tur = time.time()
        debounce_au = time.time()
        debounce_ad = time.time()
        debounce_Gi = time.time()
        debounce_rs = time.time()

        # Define global variables
        gps_logging_active = False  # Initialize global variable

        # Variable to keep track of camera position
        V = 0
        H = 0

        try:
            while self.running.is_set():

                events = inputs.get_gamepad()
                for event in events:
                    if event.ev_type == 'Absolute':
                        if event.code == 'ABS_RY':
                            y_value2 = event.state
                            if y_value2 > 30000 and (time.time() - debounce_ad) > DEBOUNCE_DELAY:  # Downward
                                self.drone(moveBy(0, 0, -0.25, 0)).wait()
                                print("Moving Downwards")
                                debounce_ad = time.time()
                            elif y_value2 < -30000 and (time.time() - debounce_au) > DEBOUNCE_DELAY:  # Upwards
                                self.drone(moveBy(0, 0, 0.25, 0)).wait()
                                print("Moving Upwards")
                                debounce_au = time.time()
                            elif abs(y_value2) <= 30000:  # Deadzone for y_value2
                                self.drone(moveBy(0, 0, 0, 0)).wait()  # Stop movement
                        elif event.code == 'ABS_Z':
                            if event.state <= 1000 and (time.time() - debounce_tul) > DEBOUNCE_DELAY:
                                self.drone(moveBy(0, -1, 0, 0)).wait()
                                print("Moving Left")
                                debounce_tul = time.time()
                        elif event.code == 'ABS_RZ':
                            if event.state <= 1000 and (time.time() - debounce_tur) > DEBOUNCE_DELAY:
                                self.drone(moveBy(0, 1, 0, 0)).wait()
                                print("Moving Right")
                                debounce_tur = time.time()
                        elif event.code == 'ABS_HAT0Y':
                            value = event.state
                            if value == -1 and (time.time() - debounce_Gi) > DEBOUNCE_DELAY:
                                # Set the gimbal target orientation
                                # Uses upwards keypad on the controller
                                H = V + 5
                                gimbal_command = set_target(
                                    gimbal_id=0,
                                    control_mode="position",  # Use "velocity" for smooth movement
                                    yaw_frame_of_reference="absolute",  # Can be "relative" or "absolute"
                                    yaw=0.0,
                                    pitch_frame_of_reference="absolute",  # Use "absolute" or "relative"
                                    pitch=H,
                                    roll_frame_of_reference="absolute",  # Use "absolute" or "relative"
                                    roll=0.0,
                                )

                                # Send the command and wait for the completion
                                command_result = self.drone(gimbal_command).wait()

                                # Check if the command was successful
                                if command_result.success():
                                    print("Gimbal target orientation set successfully.")
                                else:
                                    print("Failed to set gimbal target orientation.")

                                V = H

                            elif value == 1 and (time.time() - debounce_Gi) > DEBOUNCE_DELAY:
                                # Set the gimbal target orientation
                                # Uses downwards keypad on the controller
                                H = V - 5
                                gimbal_command = set_target(
                                    gimbal_id=0,
                                    control_mode="position",  # Use "velocity" for smooth movement
                                    yaw_frame_of_reference="absolute",  # Can be "relative" or "absolute"
                                    yaw=0.0,
                                    pitch_frame_of_reference="absolute",  # Use "absolute" or "relative"
                                    pitch=H,
                                    roll_frame_of_reference="absolute",  # Use "absolute" or "relative"
                                    roll=0.0,
                                )

                                # Send the command and wait for the completion
                                command_result = self.drone(gimbal_command).wait()

                                # Check if the command was successful
                                if command_result.success():
                                    print("Gimbal target orientation set successfully.")
                                else:
                                    print("Failed to set gimbal target orientation.")

                                V = H
                    elif event.ev_type == 'Key':
                        if event.code == 'BTN_SOUTH' and event.state == 1:  # A button
                            Original_Altitude = self.drone.get_state(GpsLocationChanged)
                            Initial_Altitude = Original_Altitude['altitude']
                            print(Initial_Altitude)
                            self.drone(TakeOff())
                            print("TakeOff")
                        elif event.code == 'BTN_EAST' and event.state == 1:  # B button
                            self.drone(Landing())
                            print("Landing")
                        elif event.code == 'BTN_WEST' and event.state == 1:  # Y button
                            self.drone(moveBy(0, 0, 0, 1)).wait()
                            print("Turning clockwise")
                        elif event.code == 'BTN_NORTH' and event.state == 1:  # X button
                            self.drone(moveBy(0, 0, 0, -1)).wait()
                            print("Turning counter clockwise")
                        elif event.code == 'BTN_TL':
                            if event.state == 1 and (time.time() - debounce_tl) > DEBOUNCE_DELAY:
                                self.drone(moveBy(-1, 0, 0, 0)).wait()
                                print("Moving Backwards")
                                debounce_tl = time.time()

                        elif event.code == 'BTN_TR':
                            if event.state == 1 and (time.time() - debounce_tr) > DEBOUNCE_DELAY:
                                self.drone(moveBy(1, 0, 0, 0)).wait()
                                print("Moving Forwards")
                                debounce_tr = time.time()


        except KeyboardInterrupt:
            print("Stopping control with Xbox controller...")
            print("GPS monitoring stopped.")
            # Save data to Excel file
            save_to_excel(gps_data_df)


        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
        print("Landed\n")


    def replay_with_vlc(self):
        # Replay this MP4 video file using VLC
        mp4_filepath = os.path.join(self.tempd, "streaming.mp4")
        subprocess.run(shlex.split(f"vlc --play-and-exit {mp4_filepath}"), check=True)

def save_to_excel(df):
    # Save DataFrame to Excel file
    filename = 'WaterBottle(45cm_Mass_Test).xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"GPS data saved to {filename}")




def test_streaming():
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.Zoom()
    streaming_example.fly()
    # Stop the video stream
    streaming_example.stop()
    # Recorded video stream postprocessing
    # streaming_example.replay_with_vlc()


if __name__ == "__main__":
    test_streaming()
