import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from datetime import datetime
import time
import csv
import os
import random
import threading
import cv2
import numpy as np
import imutils
from imutils.video import FPS
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from collections import deque
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class FreshnessDetector:
    @staticmethod
    def check_freshness(category):
        if category in ["Fruits and Vegetables", "Dairy and Eggs"]:
            return random.randint(70, 100)
        else:
            return 100

class EventBasedCapture:
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.background = None
        self.frame_buffer = deque(maxlen=5)  # Store last 5 frames

    def detect_event(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray
            return False, frame

        frameDelta = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frameDelta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True, frame

        self.background = gray
        return False, frame

    def update_buffer(self, frame):
        self.frame_buffer.append(frame)

    def get_event_frames(self):
        return list(self.frame_buffer)


class MobileNetDetector:
    def __init__(self, prototxt_path, model_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect_person(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # Person class in COCO dataset
                    return True
        return False


def write_to_csv(data):
    csv_filename = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\log.csv"
    with open(csv_filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def save_bottle_frame(frame, count, output_dir):
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y-%m-%d %H-%M-%S.%f") + f",Soda,{count}.jpg"
    file_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(file_path, frame)
    return file_path


def create_db():
    conn = sqlite3.connect("grocery_items.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS items
                 (name TEXT, category TEXT, freshness INTEGER, date TEXT)"""
    )
    conn.commit()
    conn.close()


def add_item_to_db(name, category, freshness):
    conn = sqlite3.connect("grocery_items.db")
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO items VALUES (?, ?, ?, ?)", (name, category, freshness, date)
    )
    conn.commit()
    conn.close()


class GroceryScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grocery Scanner")
        self.root.geometry("1600x800")
        self.freshness_detector = FreshnessDetector()
        self.stop_video = False
        self.fruits_and_vegetables_count = 0
        self.staples_count = 0
        self.snacks_count = 0
        self.beverages_count = 0
        self.packed_food_count = 0
        self.personal_and_baby_care_count = 0
        self.household_care_count = 0
        self.dairy_and_eggs_count = 0
        self.home_and_kitchen_count = 0

        self.detected_items_set = set()
        self.last_detection_times = {}  # Store last detection times for each item
        self.cooldown_time = 5
        self.current_order = []
        self.order_status = "Not Started"
        self.event_detector = EventBasedCapture()
        self.mobilenet_detector = MobileNetDetector(
            r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt",
            r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel",
        )

        self.create_gui()
        self.create_db()

    def create_gui(self):
        self.root.geometry("1300x800")

        # Optional: Prevent resizing the window
        # self.root.resizable(True, True)
        self.style = ttk.Style()
        self.style.theme_use("vista")

        # Configure root to expand properly
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Left frame for scanned items listbox and video frame
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10)

        # Configure left_frame to expand proportionally
        left_frame.grid_rowconfigure(1, weight=1)  # Make the video frame expand
        left_frame.grid_columnconfigure(0, weight=1)

        # Scanned items label and listbox at the top
        scanned_items_label = ttk.Label(left_frame, text="Scanned Items:")
        scanned_items_label.grid(row=0, column=0, pady=5, sticky="ew")

        self.scanned_items_listbox = tk.Listbox(left_frame, height=5, width=40)
        self.scanned_items_listbox.grid(row=1, column=0, pady=5, sticky="nsew")

        # Video frame below the listbox
        self.video_frame = ttk.Frame(left_frame)
        self.video_frame.grid(row=2, column=0, pady=0, sticky="nsew")

        # Create and attach the canvas for the video display
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.video_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Right frame for buttons and item entry
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10)

        # Configure right_frame for proportional scaling
        right_frame.grid_rowconfigure(5, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Order management frame in the right frame
        order_frame = ttk.LabelFrame(right_frame, text="Order Management", padding="10")
        order_frame.grid(row=0, column=0, sticky="ew", pady=10)

        self.order_status_label = ttk.Label(
            order_frame, text="Order Status: Not Started"
        )
        self.order_status_label.grid(row=0, column=0, columnspan=2, pady=5)

        generate_order_button = ttk.Button(
            order_frame, text="Generate Order", command=self.generate_order
        )
        generate_order_button.grid(row=1, column=0, padx=5)

        complete_order_button = ttk.Button(
            order_frame, text="Complete Order", command=self.complete_order
        )
        complete_order_button.grid(row=1, column=1, padx=5)

        # Combined entry frame for both buttons
        item_entry_frame = ttk.LabelFrame(right_frame, text="Item Entry", padding="10")
        item_entry_frame.grid(row=1, column=0, sticky="ew", pady=10)

        item_name_label = ttk.Label(item_entry_frame, text="Item Name:")
        item_name_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.item_name_entry = ttk.Entry(item_entry_frame)
        self.item_name_entry.grid(row=0, column=1, padx=5, pady=5)

        category_label = ttk.Label(item_entry_frame, text="Category:")
        category_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.category_combobox = ttk.Combobox(
            item_entry_frame,
            values=[
                "Fruits and Vegetables",
                "Staples",
                "Snacks",
                "Beverages",
                "Packed Food",
                "Personal and Baby Care",
                "Household Care",
                "Dairy and Eggs",
                "Home and Kitchen",
            ],
        )
        self.category_combobox.grid(row=1, column=1, padx=5, pady=5)

        add_item_button = ttk.Button(
            item_entry_frame, text="Add Item", command=self.add_item
        )
        add_item_button.grid(row=2, column=0, columnspan=2, pady=5)

        manual_button = ttk.Button(
            item_entry_frame,
            text="Manual Put Items (Camera Malfunction)",
            command=self.manual,
        )
        manual_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Item Information
        info_frame = ttk.LabelFrame(right_frame, text="Item Information", padding="10")
        info_frame.grid(row=2, column=0, sticky="ew", pady=10)

        self.freshness_label = ttk.Label(info_frame, text="Freshness: N/A")
        self.freshness_label.grid(row=0, column=0)

        self.fruits_and_vegetables_count_label = ttk.Label(
            info_frame, text="Fruits and Vegetables: 0"
        )
        self.fruits_and_vegetables_count_label.grid(row=1, column=0)

        self.staples_count_label = ttk.Label(info_frame, text="Staples: 0")
        self.staples_count_label.grid(row=2, column=0)

        self.snacks_count_label = ttk.Label(info_frame, text="Snacks: 0")
        self.snacks_count_label.grid(row=3, column=0)

        self.beverages_count_label = ttk.Label(info_frame, text="Beverages: 0")
        self.beverages_count_label.grid(row=4, column=0)

        self.packed_food_count_label = ttk.Label(info_frame, text="Packed Food: 0")
        self.packed_food_count_label.grid(row=5, column=0)

        self.personal_and_baby_care_count_label = ttk.Label(
            info_frame, text="Personal and Baby Care: 0"
        )
        self.personal_and_baby_care_count_label.grid(row=6, column=0)

        self.household_care_count_label = ttk.Label(
            info_frame, text="Household Care: 0"
        )
        self.household_care_count_label.grid(row=7, column=0)

        self.dairy_and_eggs_count_label = ttk.Label(
            info_frame, text="Dairy and Eggs: 0"
        )
        self.dairy_and_eggs_count_label.grid(row=8, column=0)

        self.home_and_kitchen_count_label = ttk.Label(
            info_frame, text="Home and Kitchen: 0"
        )
        self.home_and_kitchen_count_label.grid(row=9, column=0)

        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.grid(row=3, column=0, sticky="ew", pady=10)

        start_icon = self.load_icon("power_button.png")
        stop_icon = self.load_icon("no.png")

        start_button = ttk.Button(
            control_frame,
            text="Start Detection",
            image=start_icon,
            compound=tk.LEFT,
            command=self.start_detection,
        )
        start_button.image = start_icon
        start_button.grid(row=0, column=0, padx=5)

        stop_button = ttk.Button(
            control_frame,
            text="Stop Detection",
            image=stop_icon,
            compound=tk.LEFT,
            command=self.stop_detection,
        )
        stop_button.image = stop_icon
        stop_button.grid(row=0, column=1, padx=5)

    def load_icon(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icons", filename)
        return ImageTk.PhotoImage(Image.open(icon_path).resize((20, 20)))

    def update_count_display(self):
        self.fruits_and_vegetables_count_label.config(
            text=f"Fruits and Vegetables: {self.fruits_and_vegetables_count}"
        )
        self.staples_count_label.config(text=f"Staples: {self.staples_count}")
        self.snacks_count_label.config(text=f"Snacks: {self.snacks_count}")
        self.beverages_count_label.config(text=f"Beverages: {self.beverages_count}")
        self.packed_food_count_label.config(
            text=f"Packed Food: {self.packed_food_count}"
        )
        self.personal_and_baby_care_count_label.config(
            text=f"Personal and Baby Care: {self.personal_and_baby_care_count}"
        )
        self.household_care_count_label.config(
            text=f"Household Care: {self.household_care_count}"
        )
        self.dairy_and_eggs_count_label.config(
            text=f"Dairy and Eggs: {self.dairy_and_eggs_count}"
        )
        self.home_and_kitchen_count_label.config(
            text=f"Home and Kitchen: {self.home_and_kitchen_count}"
        )

    def update_count(self, category):
        if category == "Fruits and Vegetables":
            self.fruits_and_vegetables_count += 1
        elif category == "Staples":
            self.staples_count += 1
        elif category == "Snacks":
            self.snacks_count += 1
        elif category == "Beverages":
            self.beverages_count += 1
        elif category == "Packed Food":
            self.packed_food_count += 1
        elif category == "Personal and Baby Care":
            self.personal_and_baby_care_count += 1
        elif category == "Household Care":
            self.household_care_count += 1
        elif category == "Dairy and Eggs":
            self.dairy_and_eggs_count += 1
        elif category == "Home and Kitchen":
            self.home_and_kitchen_count += 1
        self.update_count_display()

    def update_order_display(self):
        self.scanned_items_listbox.delete(0, tk.END)  # Clear the listbox
        for item, details in self.current_order.items():
            self.scanned_items_listbox.insert(
                tk.END, f"{item} ({details['category']}) x{details['quantity']}"
            )

    def add_item(self):
        name = self.item_name_entry.get()
        category = self.category_combobox.get()
        if name and category:
            # freshness = self.freshness_detector.check_freshness(category)

            # Check if the item is already in the current order
            if name in self.current_order:
                self.current_order[name]["quantity"] += 1  # Increment quantity
            else:
                self.current_order[name] = {
                    "category": category,
                    "quantity": 1,
                }  # Add new item

            self.update_order_display()
            # self.add_item_to_db(name, category, freshness)
            # self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"{name} added successfully!")
        else:
            messagebox.showwarning(
                "Input Error", "Please enter both item name and category."
            )

    def manual(self):
        name = self.item_name_entry.get()
        category = self.category_combobox.get()
        if name and category:
            freshness = self.freshness_detector.check_freshness(category)

            # Check if the item is already in the current order
            if name in self.current_order:
                self.current_order[name]["quantity"] -= 1  # decrease quantity
            else:
                self.current_order[name] = {
                    "category": category,
                    "freshness": freshness,
                    "quantity": 1,
                }  # Add new item

            self.update_order_display()
            self.add_item_to_db(name, category, freshness)
            self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"{name} added successfully!")
        else:
            messagebox.showwarning(
                "Input Error", "Please enter both item name and category."
            )

    def adder(self,name,category):
        if name and category:
            freshness = self.freshness_detector.check_freshness(category)
            # Check if the item is already in the current order
            if name in self.current_order:
                self.current_order[name]["quantity"] -= 1  # decrease quantity
            else:
                self.current_order[name] = {
                    "category": category,
                    "freshness": freshness,
                    "quantity": 1,
                }  # Add new item

            self.update_order_display()
            self.add_item_to_db(name, category, freshness)
            self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"{name} added successfully!")
        else:
            messagebox.showwarning(
                "Input Error", "Please rescan the item"
            )

    def start_detection(self):
        self.stop_video = False
        threading.Thread(target=self.run_object_detection, daemon=True).start()
        messagebox.showinfo("Detection Started", "Object detection has been started.")

    def stop_detection(self):
        self.stop_video = True
        messagebox.showinfo("Detection Stopped", "Object detection has been stopped.")

    def create_db(self):
        create_db()

    def add_item_to_db(self, name, category, freshness):
        add_item_to_db(name, category, freshness)

    def generate_order(self):
        if self.order_status != "Not Started":
            messagebox.showwarning(
                "Order in Progress",
                "Please complete the current order before generating a new one.",
            )
            return

        # Generate a random order
        items = [
            "Rice", "Flour", "Chips", "Nuts", "Juice", "Soda", "Canned Beans", "Pasta",
            "Diapers", "Shampoo", "Detergent", "Disinfectant", "Yogurt", "Cheese",
            "Cutlery", "Cookware", "Banana", "Guava"
        ]
        categories = [
            "Staples", "Staples", "Snacks", "Snacks", "Beverages", "Beverages",
            "Packed Food", "Packed Food", "Personal and Baby Care", "Personal and Baby Care",
            "Household Care", "Household Care", "Dairy and Eggs", "Dairy and Eggs",
            "Home and Kitchen", "Home and Kitchen", "Fruits and Vegetables", "Fruits and Vegetables"
        ]

        order_size = random.randint(3, 7)
        # order_size = 1
        self.current_order = {}
        for _ in range(order_size):
            idx = random.randint(0, len(items) - 1)
            item = items[idx]
            category = categories[idx]
            # quantity = random.randint(1,5)
            quantity = 1
            if item in self.current_order:
                self.current_order[item]["quantity"] += quantity
            else:
                self.current_order[item] = {"category": category, "quantity": quantity}

        # Display the order
        self.scanned_items_listbox.delete(0, tk.END)
        for item, details in self.current_order.items():
            category = details["category"]
            quantity = details["quantity"]
            self.scanned_items_listbox.insert(
                tk.END, f"{item} ({category}) x{quantity}"
            )

        self.order_status = "In Progress"
        self.order_status_label.config(text="Order Status: In Progress")
        messagebox.showinfo(
            "Order Generated",
            "A new order has been generated. Start scanning items to fulfill the order.",
        )

    def complete_order(self):
        if self.order_status != "In Progress":
            messagebox.showwarning(
                "No Active Order", "There is no active order to complete."
            )
            return

        # Check if all items in the order have been fulfilled
        for item, details in self.current_order.items():
            if details["quantity"] > 0:
                messagebox.showwarning(
                    "Incomplete Order",
                    f"The item '{item}' has not been fully added to the order.",
                )
                return

        # Order is complete
        self.order_status = "Completed"
        self.order_status_label.config(text="Order Status: Completed")
        messagebox.showinfo(
            "Order Completed", "The order has been successfully completed!"
        )
        data = [datetime.now(), "New Order Started", "Order_Number : 987654321"]
        write_to_csv(data)
        self.reset_order()

    def reset_order(self):
            self.current_order = {}
            self.detected_items_set.clear()
            self.scanned_items_listbox.delete(0, tk.END)
            self.fruits_and_vegetables_count = 0
            self.staples_count = 0
            self.snacks_count = 0
            self.beverages_count = 0
            self.packed_food_count = 0
            self.personal_and_baby_care_count = 0
            self.household_care_count = 0
            self.dairy_and_eggs_count = 0
            self.home_and_kitchen_count = 0
            self.update_count_display()
            self.freshness_label.config(text="Freshness: N/A")
            self.order_status = "Not Started"
            self.order_status_label.config(text="Order Status: Not Started")

    def process_frame(self, frame):
        # Implement object detection here then call self.adder(item, category)
        # For demonstration, let's assume we detect a random item
        items = list(self.current_order.keys())
        if items:
            detected_item = random.choice(items)
            current_time = time.time()

            # Check the last detection time of the item
            if detected_item in self.last_detection_times:
                time_since_last_detection = current_time - self.last_detection_times[detected_item]
                if time_since_last_detection < self.cooldown_time:
                    print(f"Cooldown: {detected_item} detected recently, skipping.")
                    return  # Skip detection for this item due to cooldown

            category = self.current_order[detected_item]['category']
            if self.current_order[detected_item]['quantity'] > 0:
                self.current_order[detected_item]['quantity'] -= 1
            self.update_detection(detected_item, category)

            # Save the frame with timestamp and item name in the specified folder
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{detected_item}_{timestamp}.jpg"
            save_path = os.path.join(
                r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images",
                filename)
            cv2.imwrite(save_path, frame)

            # Update the last detection time for this item
            self.last_detection_times[detected_item] = current_time

    def load_previous_detections(self):
        """Load previous detection timestamps from filenames in the bottle_images folder."""
        folder_path = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images"
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".jpg"):
                # Extract item name and timestamp from the file name
                parts = file_name.split('_')
                item_name = '_'.join(parts[:-2])  # In case item names have underscores
                timestamp_str = '_'.join(parts[-2:]).replace('.jpg', '')

                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S").timestamp()
                    self.last_detection_times[item_name] = timestamp  # Store last detection time
                except ValueError:
                    pass  # Ignore invalid filenames

    def update_detection(self, item, category):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.video_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack()
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            f"Detected: {item}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        self.canvas.draw()

        self.scanned_items_listbox.insert(tk.END, f"{item} ({category})")
        self.update_count(category)
        freshness = FreshnessDetector.check_freshness(category)
        self.freshness_label.config(text=f"Freshness: {freshness}%")

    def update_gui(self, frame):
        # Convert the BGR frame from OpenCV to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(frame_rgb)
        self.ax.axis('off')
        self.canvas.draw()

    def run_object_detection(self):
        vs = cv2.VideoCapture(0)

        while not self.stop_video:
            ret, frame = vs.read()
            if not ret:
                break

            self.event_detector.update_buffer(frame)
            event_detected, _ = self.event_detector.detect_event(frame)

            if event_detected:
                event_frames = self.event_detector.get_event_frames()
                for event_frame in event_frames:
                    if not self.mobilenet_detector.detect_person(event_frame):
                        # Process and save the frame
                        print("Person not detected, frame saved")
                        self.process_frame(event_frame)
                    else:
                        print("Person detected, frame discarded")

            # Update GUI with the current frame
            self.update_gui(frame)

        vs.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = GroceryScannerApp(root)
    root.mainloop()
