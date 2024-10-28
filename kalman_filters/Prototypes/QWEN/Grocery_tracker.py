import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from imutils.video import FPS
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from datetime import datetime
import csv
import os
import random
import threading

# Kalman Filter implementation
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

# FruitCounter for tracking "bottle" category items
class FruitCounter:
    def __init__(self, box):
        self.box = box
        self.count = 0
        self.object_history = {}

    def is_inside_box(self, x, y):
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, object_id, center, category, confidence):
        x, y = center[:2]
        current_time = datetime.now()

        if category == "bottle" and self.is_inside_box(x, y):
            if object_id not in self.object_history:
                self.object_history[object_id] = {"first_seen": current_time, "last_counted": None}

            time_diff = (current_time - self.object_history[object_id]["first_seen"]).total_seconds()
            
            if self.object_history[object_id]["last_counted"] is None or \
               (current_time - self.object_history[object_id]["last_counted"]).total_seconds() > 3:
                if confidence > 0.7:
                    self.count += 1
                    self.object_history[object_id]["last_counted"] = current_time
                    return True, self.count

        return False, self.count

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def save_bottle_frame(frame, count, output_dir):
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y-%m-%d %H-%M-%S.%f") + f",bottle,{count}.jpg"
    file_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(file_path, frame)
    return file_path

def create_db():
    conn = sqlite3.connect('grocery_items.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items
                 (name TEXT, category TEXT, freshness INTEGER, date TEXT)''')
    conn.commit()
    conn.close()

def add_item_to_db(name, category, freshness):
    conn = sqlite3.connect('grocery_items.db')
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO items VALUES (?, ?, ?, ?)", (name, category, freshness, date))
    conn.commit()
    conn.close()

class GroceryScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grocery Scanner")
        self.root.geometry("1600x600")

        self.stop_video = False
        self.fruit_count = 0
        self.veg_count = 0
        self.other_count = 0

        # Set to keep track of detected items to avoid duplicates
        self.detected_items_set = set()  # This will store unique item labels
        
        self.create_gui()
        self.create_db()

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (Video feed and scanned items)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video feed
        self.video_frame = ttk.Frame(left_frame)
        self.video_frame.pack(pady=10)

        # Matplotlib figure for video feed
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.video_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # Scanned items
        scanned_items_label = ttk.Label(left_frame, text="Scanned Items:")
        scanned_items_label.pack()

        self.scanned_items_listbox = tk.Listbox(left_frame, height=15, width=40)
        self.scanned_items_listbox.pack(pady=10)

        # Right panel (Item input, counts, and controls)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Manual item entry
        item_entry_frame = ttk.LabelFrame(right_frame, text="Manual Item Entry", padding="10")
        item_entry_frame.pack(fill=tk.X, pady=10)

        item_name_label = ttk.Label(item_entry_frame, text="Item Name:")
        item_name_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.item_name_entry = ttk.Entry(item_entry_frame)
        self.item_name_entry.grid(row=0, column=1, padx=5, pady=5)

        category_label = ttk.Label(item_entry_frame, text="Category:")
        category_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.category_combobox = ttk.Combobox(item_entry_frame, values=["Fruit", "Vegetable", "Other"])
        self.category_combobox.grid(row=1, column=1, padx=5, pady=5)

        add_item_button = ttk.Button(item_entry_frame, text="Add Item", command=self.add_item)
        add_item_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Freshness and counts
        info_frame = ttk.LabelFrame(right_frame, text="Item Information", padding="10")
        info_frame.pack(fill=tk.X, pady=10)

        self.freshness_label = ttk.Label(info_frame, text="Freshness: N/A")
        self.freshness_label.pack()

        self.fruit_count_label = ttk.Label(info_frame, text="Fruits: 0")
        self.fruit_count_label.pack()
        self.veg_count_label = ttk.Label(info_frame, text="Vegetables: 0")
        self.veg_count_label.pack()
        self.other_count_label = ttk.Label(info_frame, text="Other Items: 0")
        self.other_count_label.pack()

        # Controls
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=10)

        start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        start_button.pack(side=tk.LEFT, padx=5)

        stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection)
        stop_button.pack(side=tk.LEFT, padx=5)

        demo_order_button = ttk.Button(self.root, text="Generate Demo Order", command=self.demo_order)
        demo_order_button.pack(pady=10)

        # Listbox to show scanned items
        self.scanned_items_listbox = tk.Listbox(self.root)
        self.scanned_items_listbox.pack(pady=10)

        # Label to show freshness
        self.freshness_label = tk.Label(self.root, text="Freshness: N/A")
        self.freshness_label.pack(pady=10)



    def write_to_csv(filename, data):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def save_bottle_frame(frame, count, output_dir):
        timestamp = datetime.now()
        filename = timestamp.strftime("%Y-%m-%d %H-%M-%S.%f") + f",bottle,{count}.jpg"
        file_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(file_path, frame)
        return file_path

    def create_db():
        conn = sqlite3.connect('grocery_items.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS items
                    (name TEXT, category TEXT, freshness INTEGER, date TEXT)''')
        conn.commit()
        conn.close()

    def add_item_to_db(name, category, freshness):
        conn = sqlite3.connect('grocery_items.db')
        c = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO items VALUES (?, ?, ?, ?)", (name, category, freshness, date))
        conn.commit()
        conn.close()


    def demo_order(self):
        # Simulate a hardcoded order with 2 bottles and 1 TV monitor
        order_items = [
            ("Bottle", "Other", 2),   # (Item Name, Category, Quantity)
            ("TV Monitor", "Other", 1)
        ]

        # Clear the current listbox to start fresh
        self.scanned_items_listbox.delete(0, tk.END)

        for item_name, category, quantity in order_items:
            for _ in range(quantity):
                freshness = self.check_freshness(category)

                # Check for duplicates before adding
                if f"{item_name} ({category})" not in self.detected_items_set:
                    self.detected_items_set.add(f"{item_name} ({category})")  # Add to the set
                    self.scanned_items_listbox.insert(tk.END, f"{item_name} ({category})")
                    self.update_count(category)
                    self.add_item_to_db(item_name, category, freshness)
                    self.freshness_label.config(text=f"Freshness: {freshness}%")

        messagebox.showinfo("Order Complete", "Demo order has been generated!")


    def create_db(self):
        create_db()

    def check_freshness(self, category):
        if category in ["Fruit", "Vegetable"]:
            freshness = random.randint(60, 100)
            if freshness < 75:
                messagebox.showwarning("Low Freshness", f"Warning: {category} freshness is low ({freshness}%)")
            return freshness
        return 100

    def update_count(self, category):
        if category == "Fruit":
            self.fruit_count += 1
        elif category == "Vegetable":
            self.veg_count += 1
        else:
            self.other_count += 1
        self.update_count_display()

    def update_count_display(self):
        self.fruit_count_label.config(text=f"Fruits: {self.fruit_count}")
        self.veg_count_label.config(text=f"Vegetables: {self.veg_count}")
        self.other_count_label.config(text=f"Other Items: {self.other_count}")

    def add_item(self):
        name = self.item_name_entry.get()
        category = self.category_combobox.get()
        if name and category:
            freshness = self.check_freshness(category)
            self.scanned_items_listbox.insert(tk.END, f"{name} ({category})")
            self.update_count(category)
            add_item_to_db(name, category, freshness)
            self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter both item name and category.")

    def start_detection(self):
        self.stop_video = False
        threading.Thread(target=self.run_object_detection, daemon=True).start()

    def stop_detection(self):
        self.stop_video = True

    def run_object_detection(self): 
        PROTOTXT = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt"
        MODEL = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel"
        CONFIDENCE = 0.5
        csv_filename = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\log.csv"
        bottle_image_dir = r"C:\Users\Pratham Jain\SisterDear\Flipkart\Real-Time-Object-Detection-With-OpenCV\bottle_images"

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]

        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        vs = cv2.VideoCapture(1)

        if not vs.isOpened():
            print("[ERROR] Could not open video stream")
            return

        fps = FPS().start()
        kf = KalmanFilter()

        box_x1, box_y1, box_x2, box_y2 = 150, 0, 350, 200
        counter = FruitCounter(box=(box_x1, box_y1, box_x2, box_y2))

        while not self.stop_video:
            ret, frame = vs.read()

            if not ret:
                break

            frame = imutils.resize(frame, width=480)
            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.ax.add_patch(plt.Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1, fill=False, edgecolor='blue', linewidth=2))

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]
                    object_id = i

                    if label == "bottle":
                        # Create a unique identifier for this item
                        unique_item_identifier = f"bottle-{object_id}"

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        centerX = (startX + endX) // 2
                        centerY = (startY + endY) // 2

                        z = np.array([[centerX], [centerY]])
                        predicted = kf.predict()
                        updated = kf.update(z)

                        is_bottle, current_count = counter.update(object_id, (centerX, centerY), "bottle", confidence)

                        if is_bottle:  # Check if the bottle is entering the box
                            # Only add to the list if it's a new detection
                            if unique_item_identifier not in self.detected_items_set:
                                self.detected_items_set.add(unique_item_identifier)  # Add to the set

                                # Insert the detected "bottle" into the order list
                                self.scanned_items_listbox.insert(tk.END, f"Bottle ({current_count})")
                                
                                # Update count for "bottle" items (fruits in your case)
                                self.update_count("Fruit")

                                # Log to CSV and save frame
                                data = [datetime.now(), "bottle", current_count]
                                write_to_csv(csv_filename, data)
                                save_bottle_frame(frame, current_count, bottle_image_dir)

                        # Draw the tracking information on the frame
                        self.ax.plot([predicted[0], updated[0]], [predicted[1], updated[1]], 'ro-')
                        self.ax.add_patch(plt.Rectangle((startX, startY), endX - startX, endY - startY, fill=False, edgecolor='red', linewidth=2))
                        self.ax.text(startX, startY - 10, f"{label}: {confidence:.2f}", color='red', fontsize=8, backgroundcolor='white')

            self.canvas.draw()
            fps.update()

        fps.stop()
        vs.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = GroceryScannerApp(root)
    root.mainloop()