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
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
from pynput import keyboard

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

# Gets the drone IP address, and the port so the code can connect to it.
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

# Finds the model which is saved as a .pt file, and saves it to a variable.
Model_Path = r"/home/labpc/Downloads/Water_Can_Version2.pt"
model = YOLO(Model_Path)

# Define debounce delay in seconds (adjust as needed)
DEBOUNCE_DELAY = 0.3

# Define a global DataFrame to store GPS data
gps_data_df = pd.DataFrame(columns=['Timestamp', 'Latitude (Degrees)', 'Longitude (Degrees)', 'Altitude (ft)', 'Objects', 'Confidence'])

# Define the key to check
key_to_check = 'g'  # Example: 'a' key

# Dictionary to map class IDs to object names
class_names = {
    1: 'Water Bottle',       # Example class IDs and names
    0: 'Metal Can',
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
        self.running = multiprocessing.Event()
        self.running.set()

    def start(self):
        # Connect to drone
        assert self.drone.connect(retry=3)

        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"

        # You can record the video stream from the drone if you plan to do some
        # post processing.
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

    def stop(self):
        self.running.clear()
        for p in self.processes:
            p.join()

        self.output_thread.join()

        # Properly stop the video stream and disconnect
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
                logging.error(f"OpenCV error: {e}")
                print("We have a cv2 error.")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                print("We have a unexpected error.")

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
            yuv_data, height, width = self.frame_queue.get()

            # Reshape and convert YUV to BGR
            yuv_data = yuv_data.reshape((height * 3 // 2, width))
            bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)

            # Run inference
            results = model(bgr_frame)

            return results

        except queue.Empty:
            # Handle empty queue case
            print("Queue was empty")
            return None
        except cv2.error as e:
            # Handle OpenCV errors
            print(f"OpenCV error: {e}")
            return None
        except Exception as e:
            # Handle other exceptions
            print(f"Unexpected error: {e}")
            return None


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

    def start_keyboard_listener(self):
        def on_press(key):
            global gps_logging_active  # Declare gps_logging_active as global
            global gps_data_df
            try:
                # Prints the location of the drone before taking off.
                #print("GPS position before take-off :", self.drone.get_state(HomeChanged))
                if key.char == 'g':  # Start/Stop GPS logging with 'g' key
                    gps_logging_active = not gps_logging_active
                    if gps_logging_active:
                        print("GPS logging started.")
                        print("It worked")
                        # Wait for GPS location change
                        results = self.Frame_Check()
                        gps_data = self.drone.get_state(GpsLocationChanged)

                        # Extract coordinates, and records time when it happens.
                        timestamp = datetime.now()
                        latitude = gps_data['latitude']
                        longitude = gps_data['longitude']
                        altitude = gps_data['altitude']

                        # Process detection results
                        detected_objects = []
                        for result in results:
                            labels = result.boxes.cls.cpu().numpy()  # Convert tensor to numpy array
                            confidences = result.boxes.conf.cpu().numpy()  # Convert tensor to numpy array

                        for i, label in enumerate(labels):
                            class_id = int(label)  # Convert tensor label to integer
                            object_name = class_names.get(class_id, "unknown")  # Get object name
                            detected_objects.append(object_name)

                        a = confidences

                        # Adds the recorded data to the data frame (gps_data_df).
                        gps_data_df = pd.concat([gps_data_df, pd.DataFrame({
                            'Timestamp': [timestamp],
                            'Latitude (Degrees)': [latitude],
                            'Longitude (Degrees)': [longitude],
                            'Altitude (ft)': [altitude],
                            'Objects': [detected_objects],
                            'Confidence': [a]
                        })], ignore_index=True)
                    else:
                        print("GPS logging stopped.")
            except AttributeError:
                # Handle special keys or other exceptions if needed
                pass

        # Setup the keyboard listener
        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        return keyboard_listener

    def fly(self):
        # Declare gps_data_df as global
        global gps_data_df
        global gps_logging_active

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

        # Start the keyboard listener in a separate thread
        keyboard_listener = self.start_keyboard_listener()

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
                                H = V + 10
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
                                H = V - 10
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
            keyboard_listener.stop()  # Stop the keyboard listener

        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
        print("Landed\n")


    def replay_with_vlc(self):
        # Replay this MP4 video file using VLC
        mp4_filepath = os.path.join(self.tempd, "streaming.mp4")
        subprocess.run(shlex.split(f"vlc --play-and-exit {mp4_filepath}"), check=True)

def save_to_excel(df):
    # Save DataFrame to Excel file
    filename = 'MetalCan.xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"GPS data saved to {filename}")




def test_streaming():
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.fly()
    # Stop the video stream
    streaming_example.stop()
    # Recorded video stream postprocessing
    # streaming_example.replay_with_vlc()


if __name__ == "__main__":
    test_streaming()
