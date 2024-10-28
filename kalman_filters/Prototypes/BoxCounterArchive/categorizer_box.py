import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from imutils.video import FPS

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
    def __init__(self, box):
        self.box = box  # The box is a tuple (x1, y1, x2, y2)
        self.count = 0
        self.tracked_objects = set()

    def is_inside_box(self, x, y):
        """ Check if the object's center is inside the box """
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, object_id, center):
        """ Update count if an object enters the box """
        x, y = center[:2]  # Take only the first two elements (x and y)
        if object_id not in self.tracked_objects and self.is_inside_box(x, y):
            self.count += 1
            self.tracked_objects.add(object_id)  # Track the object to avoid double counting
        return self.count


def main():
    
    PROTOTXT = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt"
    MODEL = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel"
    CONFIDENCE = 0.5
    count =0 

    # Initialize the list of class labels MobileNet SSD was trained on
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(1)
    if not vs.isOpened():
        print("[ERROR] Could not open video stream")
        return
    fps = FPS().start()

    kf = KalmanFilter()

    # Define the box where objects will be counted
    box_x1, box_y1, box_x2, box_y2 = 100, 100, 300, 300
    counter = FruitCounter(box=(box_x1, box_y1, box_x2, box_y2))

    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()

    while True:
        ret, frame = vs.read()

        if not ret:
            print("[ERROR] No frame received from video stream")
            break

        frame = imutils.resize(frame, width=480)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Draw the counting box on the frame
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                center_x = (startX + endX) / 2
                center_y = (startY + endY) / 2

                kf.predict()
                filtered_state = kf.update(np.array([center_x, center_y]))

                # Assign an object_id (could be a hash of center_x and center_y for simplicity)
                object_id = f"{int(center_x)}_{int(center_y)}"

                # Counting logic for objects entering the box
                count = counter.update(object_id, filtered_state)

                # Draw bounding box and label
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the object count on the frame
        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert BGR to RGB for displaying in matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clear the current plot and display the updated frame
        ax.clear()
        ax.imshow(frame_rgb)
        ax.set_title(f"Object Counter running at : {fps}")
        ax.axis("off")

        # Draw and flush the frame to the window
        plt.draw()
        plt.pause(0.01)

        fps.update()

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    if fps._numFrames > 0:
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")
    else:
        print("[INFO] No frames processed, unable to calculate FPS")

    vs.release()
    plt.close()

if __name__ == "__main__":
    main()

