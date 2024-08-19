The goal of this project is to use a Parrot Anafi drone, and object detection model to improve the efficiency of litter audits. Parrot Anafi was chosen because of its library which allows for communication between the drone, and other devices. This can be done through the Olympe library that is used for communication in the codes. Yolo is being used as the object detection model, for its ease of use and support.

Parrot Olympe can be installed with this: pip3 install parrot-olympe

More information about the library can be found here: https://developer.parrot.com/docs/olympe/index.html

Yolo can be installed with this: pip install ultralytics

More information can be found here: https://docs.ultralytics.com

The following codes show the steps that were taken to develop the objective, and are in the Important Files forlder:

GPU-TRAIN: This code trains the model, using the GPU of the computer. It uses cuda to achieve this, and it was used with a windows laptop with a nvidia GPU.

Final_Version: This code captures the video from the drone, and passes it through the object detection model. It uses a dataframe to log the detected objects, and save it to an excel file when the code is finished. It's also able to control the motion of the drone with the inputs library. It also uses a model trained on a custom dataset, it's called Water_Can_Version2.pt.

Further details can be found in the report.
