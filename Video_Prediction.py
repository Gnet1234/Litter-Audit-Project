from ultralytics import YOLO
import cv2
import pandas as pd

# Load a pretrained YOLOv8 model
Model_Path = r"C:\Users\Admin\Downloads\Water_Can_Version2.pt"
model = YOLO("yolov5n.pt")

# Specify the path to your video file
video_path = r"C:\Users\Admin\Documents\Drone Video\P0390039.MP4"
cap = cv2.VideoCapture(video_path)

# Variable for checking boxes
boxes = None

# Define a global DataFrame to store GPS data
gps_data_df = pd.DataFrame(columns=['Objects', 'Confidence'])

# Dictionary to map class IDs to object names
class_names = {
    2: 'Water Bottle',
    1: 'Metal Can',       
    0: 'Mask',
    # Add other class IDs and names as needed
}

def Video_Extraction():
    global gps_data_df  # Ensure the global DataFrame is used

    # Process video frame-by-frame
    while True:
        # Frame-by-frame video capture
        connected, frame = cap.read()
        if not connected:
            break  # Exit if the video ends

        # Obtain results from detection with YOLOv8 model on the frame
        results = model(frame)

        # Plot boundary & image masking from obtained results
        annotated_frame = results[0].plot()

        # Display results
        cv2.imshow('Machine Vision', annotated_frame)

        # Process detection results
        boxes = None # Variable used to store information about bounding boxes of detected objects.
        detected_objects = [] 
        for result in results:
            boxes = result.boxes.xyxy
        
        labels = results[0].boxes.cls.cpu().numpy()  # Convert tensor to numpy array
        confidences = results[0].boxes.conf.cpu().numpy()  # Convert tensor to numpy array

        # This gets information about the labels for each detected object, and adds them to the list.
        for i, label in enumerate(labels):
            class_id = int(label)  # Convert tensor label to integer
            object_name = class_names.get(class_id, "unknown")  # Get object name
            detected_objects.append(object_name) # Adds them to the list

        # Checks if there is any detected bouding boxes, if yes the label information, and confidence is added to the dataframe. 
        if boxes is not None and boxes.size(0) > 0:
            # Add the recorded data to the data frame (gps_data_df)
            gps_data_df = pd.concat([gps_data_df, pd.DataFrame({
                'Objects': [detected_objects],
                'Confidence': [confidences]
            })], ignore_index=True)

        # Stop process on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Save data to Excel file and release resources
    save_to_excel(gps_data_df)
    cap.release()
    cv2.destroyAllWindows()


def save_to_excel(df):
    # Save DataFrame to Excel file
    filename = 'Test.xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"GPS data saved to {filename}")


if __name__ == "__main__":
    Video_Extraction()
