import numpy as np
import cv2
from imutils.video import FPS
import imutils
import time
import logging
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dt=1):
        self.dt = dt
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.1
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2].flatten()

class FruitCounter:
    def __init__(self, counting_line_y):
        self.counting_line_y = counting_line_y
        self.count = 0
        self.last_y = None

    def update(self, y):
        if self.last_y is not None:
            if self.last_y < self.counting_line_y and y >= self.counting_line_y:
                self.count += 1
        self.last_y = y
        return self.count

def main():
    # Define paths for the MobileNet SSD model
    PROTOTXT = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt"
    MODEL = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel"
    CONFIDENCE = 0.2

    # Initialize the list of class labels MobileNet SSD was trained on
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    # Initialize the video stream and FPS counter
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(1)  # Use cv2.VideoCapture instead of VideoStream
    if not vs.isOpened():
        print("[ERROR] Could not open video stream")
        return
    time.sleep(2.0)
    fps = FPS().start()  

    kf = KalmanFilter()
    counter = FruitCounter(counting_line_y=640)  # Adjust based on your frame size

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    # Initialize count to 0 at the beginning of each loop iteration
    count = 0

    while True:
        ret, frame = vs.read()
        
        # Check if the frame was successfully grabbed
        if not ret:
            print("[ERROR] No frame received from video stream")
            break

        frame = imutils.resize(frame, width=480)
        (h, w) = frame.shape[:2]

        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()
        
        

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Use the center of the detection for Kalman filter
                center_x = (startX + endX) / 2
                center_y = (startY + endY) / 2

                # Kalman filter update
                kf.predict()
                filtered_state = kf.update(np.array([center_x, center_y]))

                # Counting logic
                count = counter.update(filtered_state[1])

                # Draw bounding box and label
                # label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame using matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.text(10, 30, f"Count: {count}", fontsize=12, color='g')
        plt.pause(0.001)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
            # break

        fps.update()

    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    if fps._numFrames > 0:
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")
    else:
        print("[INFO] No frames processed, unable to calculate FPS")

    vs.release()  # Release the video capture object
    plt.ioff()
    plt.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
