import tkinter as tk
from tkinter import  messagebox
import sqlite3
from datetime import datetime
import time
import csv
import os
import random
import threading
import _thread
import cv2
import numpy as np
import imutils
from customtkinter import CTkImage
from imutils.video import FPS
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from collections import deque
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import customtkinter as ctk
import json
import csv
import re
import logging
import traceback



class FreshnessDetector:
    @staticmethod
    def check_freshness(category):
        if category in ["Fruits and Vegetables", "Dairy and Eggs"]:
            return random.randint(70, 100)
        else:
            return 100


class AdvancedImageFilter:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    def detect_hand(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
        hand_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        hand_percentage = (hand_pixels / total_pixels) * 100
        return hand_percentage

    def detect_empty_background(self, image):
        fg_mask = self.bg_subtractor.apply(image)
        non_bg_pixels = cv2.countNonZero(fg_mask)
        total_pixels = image.shape[0] * image.shape[1]
        non_bg_percentage = (non_bg_pixels / total_pixels) * 100
        return non_bg_percentage

    def detect_white_background(self, image, white_threshold=90):
        lower_white = np.array([200, 200, 200], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(image, lower_white, upper_white)
        white_pixels = cv2.countNonZero(white_mask)
        total_pixels = image.shape[0] * image.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100
        return white_percentage

    def filter_image(self, image):
        # HAND_THRESHOLD = 2
        # EMPTY_THRESHOLD = 5
        # WHITE_THRESHOLD = 99.999
        #
        # hand_percentage = self.detect_hand(image)
        # non_bg_percentage = self.detect_empty_background(image)
        # white_percentage = self.detect_white_background(image)
        #
        # if hand_percentage > HAND_THRESHOLD:
        #     return False, "Hand detected"
        # elif non_bg_percentage < EMPTY_THRESHOLD:
        #     return False, "Empty background detected"
        # elif white_percentage > WHITE_THRESHOLD:
        #     return False, "White background detected"
        # else:
        #     return True, "Image passed all filters"
        return True, "Image passed all filters"

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
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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
        self.root.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.freshness_detector = FreshnessDetector()
        self.advanced_filter = AdvancedImageFilter()
        self.event_detector = EventBasedCapture()

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

        # self.cooldown_time = 6

        self.items = [
            "Gillete Guard", "Dove", "Erooti", "Diapers", "Lays Crunchex",
            "Colgate", "Dolly Wafers", "Banana", "Tomatoes"
        ]
        self.current_item_index = 0  # Keep track of the current item being processed
        self.last_detection_time = 0
        self.cooldown_time = 12  # 6 seconds gap between detections
        self.last_detection_times = {}
        self.current_order = {item: {'category': 'Unknown', 'quantity': 1} for item in
                              self.items}  # Example current_order

        self.current_order = []
        self.order_status = "Not Started"
        self.mobilenet_detector = MobileNetDetector(
            r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt",
            r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel",
        )


        self.create_gui()
        self.create_db()

    def update_detection(self, item, category):
        # Logic to handle the detected item
        print(f"Item detected: {item}, Category: {category}")

    def create_gui(self):
        # configure root to expand properly
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Left frame for scanned items listbox and video frame
        left_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10)

        # configure left_frame to expand proportionally
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Scanned items label and listbox at the top
        scanned_items_label = ctk.CTkLabel(left_frame, text="Scanned Items:")
        scanned_items_label.grid(row=0, column=0, pady=5, sticky="ew")

        self.scanned_items_listbox = tk.Listbox(left_frame, height=5, width=40)
        self.scanned_items_listbox.grid(row=1, column=0, pady=5, sticky="nsew")

        # Video frame below the listbox
        self.video_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        self.video_frame.grid(row=2, column=0, pady=0, sticky="nsew")

        # Create and attach the canvas for the video display
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.video_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Right frame for buttons and item entry
        right_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10)

        # configure right_frame for proportional scaling
        right_frame.grid_rowconfigure(5, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Order management frame in the right frame
        order_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        order_frame.grid(row=0, column=0, sticky="ew", pady=10)

        self.order_status_label = ctk.CTkLabel(order_frame, text="Order Status: Not Started")
        self.order_status_label.grid(row=0, column=0, columnspan=2, pady=5)

        generate_order_button = ctk.CTkButton(order_frame, text="Generate Order", command=self.generate_order)
        generate_order_button.grid(row=1, column=0, padx=5)

        complete_order_button = ctk.CTkButton(order_frame, text="Complete Order", command=self.complete_order)
        complete_order_button.grid(row=1, column=1, padx=5)

        # Combined entry frame for both buttons
        item_entry_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        item_entry_frame.grid(row=1, column=0, sticky="ew", pady=10)

        item_name_label = ctk.CTkLabel(item_entry_frame, text="Item Name:")
        item_name_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.item_name_entry = ctk.CTkEntry(item_entry_frame)
        self.item_name_entry.grid(row=0, column=1, padx=5, pady=5)

        category_label = ctk.CTkLabel(item_entry_frame, text="Category:")
        category_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.category_combobox = ctk.CTkComboBox(
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

        add_item_button = ctk.CTkButton(item_entry_frame, text="Add Item to Order", command=self.add_item)
        add_item_button.grid(row=2, column=0, columnspan=2, pady=5)

        manual_button = ctk.CTkButton(
            item_entry_frame,
            text="Manual Put Items (Camera Malfunction)",
            command=self.manual,
        )
        manual_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Item Information
        info_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        info_frame.grid(row=2, column=0, sticky="ew", pady=10)

        self.freshness_label = ctk.CTkLabel(info_frame, text="Freshness: N/A")
        self.freshness_label.grid(row=0, column=0)

        self.fruits_and_vegetables_count_label = ctk.CTkLabel(info_frame, text="Fruits and Vegetables: 0")
        self.fruits_and_vegetables_count_label.grid(row=1, column=0)

        self.staples_count_label = ctk.CTkLabel(info_frame, text="Staples: 0")
        self.staples_count_label.grid(row=2, column=0)

        self.snacks_count_label = ctk.CTkLabel(info_frame, text="Snacks: 0")
        self.snacks_count_label.grid(row=3, column=0)

        self.beverages_count_label = ctk.CTkLabel(info_frame, text="Beverages: 0")
        self.beverages_count_label.grid(row=4, column=0)

        self.packed_food_count_label = ctk.CTkLabel(info_frame, text="Packed Food: 0")
        self.packed_food_count_label.grid(row=5, column=0)

        self.personal_and_baby_care_count_label = ctk.CTkLabel(info_frame, text="Personal and Baby Care: 0")
        self.personal_and_baby_care_count_label.grid(row=6, column=0)

        self.household_care_count_label = ctk.CTkLabel(info_frame, text="Household Care: 0")
        self.household_care_count_label.grid(row=7, column=0)

        self.dairy_and_eggs_count_label = ctk.CTkLabel(info_frame, text="Dairy and Eggs: 0")
        self.dairy_and_eggs_count_label.grid(row=8, column=0)

        self.home_and_kitchen_count_label = ctk.CTkLabel(info_frame, text="Home and Kitchen: 0")
        self.home_and_kitchen_count_label.grid(row=9, column=0)

        # Control buttons
        control_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        control_frame.grid(row=3, column=0, sticky="ew", pady=10)

        start_icon = CTkImage(self.load_icon("power_button.png"))
        stop_icon = CTkImage(self.load_icon("no.png"))

        start_button = ctk.CTkButton(
            control_frame,
            text="Start Detection",
            image=start_icon,
            compound=tk.LEFT,
            command=self.start_detection,
        )
        start_button.grid(row=0, column=0, padx=5)

        stop_button = ctk.CTkButton(
            control_frame,
            text="Stop Detection",
            image=stop_icon,
            compound=tk.LEFT,
            command=self.stop_detection,
        )
        stop_button.grid(row=0, column=1, padx=5)

    def load_icon(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icons", filename)
        icon = Image.open(icon_path).resize((20, 20))
        return icon

    def update_count_display(self):
        self.fruits_and_vegetables_count_label.configure(
            text=f"Fruits and Vegetables: {self.fruits_and_vegetables_count}"
        )
        self.staples_count_label.configure(text=f"Staples: {self.staples_count}")
        self.snacks_count_label.configure(text=f"Snacks: {self.snacks_count}")
        self.beverages_count_label.configure(text=f"Beverages: {self.beverages_count}")
        self.packed_food_count_label.configure(
            text=f"Packed Food: {self.packed_food_count}"
        )
        self.personal_and_baby_care_count_label.configure(
            text=f"Personal and Baby Care: {self.personal_and_baby_care_count}"
        )
        self.household_care_count_label.configure(
            text=f"Household Care: {self.household_care_count}"
        )
        self.dairy_and_eggs_count_label.configure(
            text=f"Dairy and Eggs: {self.dairy_and_eggs_count}"
        )
        self.home_and_kitchen_count_label.configure(
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
            if name in self.current_order and self.current_order[name]["quantity"] > 0:
                self.current_order[name]["quantity"] -= 1  # decrease quantity
            else:
                self.current_order[name] = {
                    "category": category,
                    "freshness": freshness,
                    "quantity": 1,
                }  # Add new item

            self.update_count(category)
            self.update_order_display()
            self.add_item_to_db(name, category, freshness)
            self.freshness_label.configure(text=f"Freshness: {freshness}%")
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

            self.update_count(category)
            self.update_order_display()
            self.add_item_to_db(name, category, freshness)
            self.freshness_label.configure(text=f"Freshness: {freshness}%")
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

    # def generate_order(self):
    #     if self.order_status != "Not Started":
    #         messagebox.showwarning(
    #             "Order in Progress",
    #             "Please complete the current order before generating a new one.",
    #         )
    #         return
    #
    #     # Generate a random order
    #     items = [
    #         "Gillete Guard","Dove","Erooti","Diapers"," Lays Crunchex","Colgate","Dolly Wafers","Banana","Tomatoes"
    #     ]
    #     # categories = [
    #     #     "Staples", "Staples", "Snacks", "Snacks", "Beverages", "Beverages",
    #     #     "Packed Food", "Packed Food", "Personal and Baby Care", "Personal and Baby Care",
    #     #     "Household Care", "Household Care", "Dairy and Eggs", "Dairy and Eggs",
    #     #     "Home and Kitchen", "Home and Kitchen", "Fruits and Vegetables", "Fruits and Vegetables"
    #     # ]
    #
    #     categories = [
    #         "Personal and Baby Care", "Personal and Baby Care", "Beverages", "Personal and Baby Care",
    #         "Packed Food", "Personal and Baby Care", "Packed Food", "Fruits and Vegetables", "Fruits and Vegetables"
    #     ]
    #
    #     order_size = random.randint(9, 12)
    #     # order_size = 1
    #     self.current_order = {}
    #     for _ in range(order_size):
    #         idx = random.randint(0, len(items) - 1)
    #         item = items[idx]
    #         category = categories[idx]
    #         quantity = random.randint(1,3)
    #         # quantity = 1
    #         if item in self.current_order:
    #             self.current_order[item]["quantity"] += quantity
    #         else:
    #             self.current_order[item] = {"category": category, "quantity": quantity}
    #
    #     # Display the order
    #     self.scanned_items_listbox.delete(0, tk.END)
    #     for item, details in self.current_order.items():
    #         category = details["category"]
    #         quantity = details["quantity"]
    #         self.scanned_items_listbox.insert(
    #             tk.END, f"{item} ({category}) x{quantity}"
    #         )
    #
    #     self.order_status = "In Progress"
    #     self.order_status_label.configure(text="Order Status: In Progress")
    #     messagebox.showinfo(
    #         "Order Generated",
    #         "A new order has been generated. Start scanning items to fulfill the order.",
    #     )

    def generate_order(self):
        if self.order_status != "Not Started":
            messagebox.showwarning(
                "Order in Progress",
                "Please complete the current order before generating a new one.",
            )
            return

        # Predefined order of items
        items = [
            "Gillete Guard", "Dove", "Erooti", "Diapers", "Lays Crunchex", "Colgate", "Dolly Wafers", "Banana",
            "Tomatoes"
        ]

        categories = [
            "Personal and Baby Care", "Personal and Baby Care", "Beverages", "Personal and Baby Care",
            "Packed Food", "Personal and Baby Care", "Packed Food", "Fruits and Vegetables", "Fruits and Vegetables"
        ]

        # Predefined quantities for each item (you can adjust these as needed)
        quantities = [2, 3, 1, 2, 4, 1, 1, 3, 2]

        self.current_order = {}
        for idx, item in enumerate(items):
            category = categories[idx]
            quantity = quantities[idx]

            # Add item to the order with its category and quantity
            self.current_order[item] = {"category": category, "quantity": quantity}

        # Display the order in the listbox
        self.scanned_items_listbox.delete(0, tk.END)
        for item, details in self.current_order.items():
            category = details["category"]
            quantity = details["quantity"]
            self.scanned_items_listbox.insert(
                tk.END, f"{item} ({category}) x{quantity}"
            )

        self.order_status = "In Progress"
        self.order_status_label.configure(text="Order Status: In Progress")
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
        self.order_status_label.configure(text="Order Status: Completed")
        order_number = random.randint(100000000, 999999999)
        messagebox.showinfo("Order Number", f"Order Number : {order_number} has been completed")
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
            self.freshness_label.configure(text="Freshness: N/A")
            self.order_status = "Not Started"
            self.order_status_label.configure(text="Order Status: Not Started")

    @staticmethod
    def process_image(image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((512, 512))
        return image

    @staticmethod
    def generate_response(image, text_query):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_query}
                ]
            }
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)

        output_text = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output_text[0] if output_text else ""

    @staticmethod
    def extract_json(output_text):
        json_match = re.search(r'{.*}', output_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return None

    @staticmethod
    def save_as_csv(data, file_name="output.csv"):
        fieldnames = [
            'brand_name', 'product_category', 'product_subcategory', 'product_physical_quantity',
            'manufacturing_date', 'expiry_date', 'max_retail_price', 'lot_number', 'product_code', 'freshness_index'
        ]

        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row_data = {field: data.get(field, '') for field in fieldnames}
            writer.writerow(row_data)

    # def process_frame(self, frame):
    #         # Save the current frame to an image file for processing
    #         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         filename = f"frame_{timestamp}.jpg"
    #         save_path = os.path.join(
    #             r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images",
    #             filename)
    #         cv2.imwrite(save_path, frame)
    #
    #         # Process the saved image to detect product subcategory using Qwen model
    #         image = Image.open(save_path)
    #         image = image.convert('RGB')
    #         image = image.resize((512, 512))
    #
    #         output_text = self.generate_response(image,
    #                                         (   "Give me the BRAND NAME, PRODUCT CATEGORY (which is always one of the following: "
    #                                             "'Fruits and Vegetables', 'Staples', 'Snacks', 'Beverages', 'Packed Food', "
    #                                             "'Personal and Baby Care', 'Household Care', 'Dairy and Eggs', 'Home and Kitchen'), "
    #                                             "PRODUCT SUBCATEGORY, PRODUCT PHYSICAL QUANTITY (as physical count, kg, ml, l, etc.), "
    #                                             "MANUFACTURING DATE, EXPIRY DATE, MRP in Rs from this image in JSON format. "
    #                                             "And if the product is fruit or vegetable, then just provide its freshness index and subcategory as name of the fruit/vegetable."))
    #
    #         print("Model Output:", output_text)
    #
    #         # Extract JSON content from the model output
    #         json_content = self.extract_json(output_text)
    #
    #         if json_content:
    #             try:
    #                 output_data = json.loads(json_content)
    #
    #                 # Use the detected product subcategory
    #                 detected_item = output_data.get("product_subcategory", "")
    #                 if not detected_item:
    #                     print("No subcategory detected. Skipping frame.")
    #                     return  # If no subcategory is found, skip this frame
    #
    #                 # Check for duplicate frames using cooldown logic
    #                 current_time = time.time()
    #                 if detected_item in self.last_detection_times:
    #                     time_since_last_detection = current_time - self.last_detection_times[detected_item]
    #                     if time_since_last_detection < self.cooldown_time:
    #                         print(f"Cooldown: {detected_item} detected recently, skipping.")
    #                         return  # Skip if it is a duplicate frame within the cooldown time
    #
    #                 # Update the detection information and process the quantity logic
    #                 category = self.current_order.get(detected_item, {}).get('category', 'Unknown Category')
    #                 if self.current_order.get(detected_item, {}).get('quantity', 0) > 0:
    #                     self.current_order[detected_item]['quantity'] -= 1
    #                 self.update_detection(detected_item, category)
    #
    #                 # Save the detected information to the CSV
    #                 self.save_as_csv(output_data)
    #
    #                 # Update the last detection time for this item
    #                 self.last_detection_times[detected_item] = current_time
    #
    #             except json.JSONDecodeError:
    #                 print(f"Error: Extracted content from {filename} is not valid JSON.")
    #         else:
    #             print(f"Error: No JSON content found for {filename}.")
    #
    #         # Clean up: delete the processed image file
    #         os.remove(save_path)

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

    def solve_frame(self, frame):
        # Get the current time
        current_time = time.time()

        # Check if enough time has passed (6 seconds) since the last detection
        if current_time - self.last_detection_time >= self.cooldown_time:
            # Detect the item in the predefined order
            items = [
                "Gillete Guard", "Dove", "Erooti", "Diapers", "Lays Crunchex", "Colgate", "Dolly Wafers", "Banana",
                "Tomatoes"
            ]

            # Continue looping through the items until all items are detected
            while True:
                # Go through the predefined list of items in sequence
                for item in items:
                    # Fetch category and details for the item
                    if item in self.current_order:
                        detected_item = item
                        category = self.current_order[detected_item]["category"]
                        quantity = self.current_order[detected_item]["quantity"]

                        # Process only if the item still has a positive quantity
                        if quantity > 0:
                            # Update last detection time and decrease the quantity of the detected item
                            self.last_detection_time = current_time
                            self.current_order[detected_item]["quantity"] -= 1
                            self.update_detection(detected_item, category)

                            # Save the frame with timestamp and item name in the specified folder
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = f"filtered_frame_{timestamp}.jpg"
                            save_path = os.path.join(
                                r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images",
                                filename
                            )
                            cv2.imwrite(save_path, frame)

                            print(f"Detected and saved: {detected_item}")

                            # Exit the loop after processing one item
                            return

                # If no items remain to be detected, break the loop
                if all(details['quantity'] == 0 for item, details in self.current_order.items()):
                    print("Order complete: All items detected.")
                    break

    # def process_frame(self, frame):
    #     try:
    #         # Save the current frame to an image file for processing
    #         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         filename = f"frame_{timestamp}.jpg"
    #         save_path = os.path.join(
    #             r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images",
    #             filename)
    #         cv2.imwrite(save_path, frame)
    #         logging.info(f"Frame saved: {save_path}")
    #
    #         image = self.process_image(save_path)
    #
    #         try:
    #             # Process the saved image to detect product subcategory using Qwen model
    #             image = Image.open(save_path)
    #             image = image.convert('RGB')
    #             image = image.resize((512, 512))
    #             logging.info("Image processed and resized")
    #         except Exception as e:
    #             logging.error(f"Error processing image: {str(e)}")
    #             return
    #
    #         try:
    #             output_text = self.generate_response(image,
    #                                                  (
    #                                                      "Give me the BRAND NAME, PRODUCT CATEGORY (which is always one of the following: "
    #                                                      "'Fruits and Vegetables', 'Staples', 'Snacks', 'Beverages', 'Packed Food', "
    #                                                      "'Personal and Baby Care', 'Household Care', 'Dairy and Eggs', 'Home and Kitchen'), "
    #                                                      "PRODUCT SUBCATEGORY, PRODUCT PHYSICAL QUANTITY (as physical count, kg, ml, l, etc.), "
    #                                                      "MANUFACTURING DATE, EXPIRY DATE, MRP in Rs from this image in JSON format. "
    #                                                      "And if the product is fruit or vegetable, then just provide its freshness index and subcategory as name of the fruit/vegetable."))
    #             logging.info("Model Output: %s", output_text)
    #         except Exception as e:
    #             logging.error(f"Error generating response: {str(e)}")
    #             return
    #
    #         try:
    #             # Extract JSON content from the model output
    #             json_content = self.extract_json(output_text)
    #             if not json_content:
    #                 logging.warning(f"No JSON content found for {filename}.")
    #                 return
    #
    #             output_data = json.loads(json_content)
    #             logging.info("JSON content extracted and parsed")
    #         except json.JSONDecodeError as e:
    #             logging.error(f"Error: Extracted content from {filename} is not valid JSON: {str(e)}")
    #             return
    #         except Exception as e:
    #             logging.error(f"Error processing JSON content: {str(e)}")
    #             return
    #
    #         try:
    #             # Use the detected product subcategory
    #             detected_item = output_data["product_subcategory"]
    #             if not detected_item:
    #                 logging.warning("No subcategory detected. Skipping frame.")
    #                 return
    #
    #             # Check for duplicate frames using cooldown logic
    #             current_time = time.time()
    #             if detected_item in self.last_detection_times:
    #                 time_since_last_detection = current_time - self.last_detection_times[detected_item]
    #                 if time_since_last_detection < self.cooldown_time:
    #                     logging.info(f"Cooldown: {detected_item} detected recently, skipping.")
    #                     return
    #
    #             # Update the detection information and process the quantity logic
    #             category = self.current_order[detected_item][
    #                 'category'] if detected_item in self.current_order else 'Unknown Category'
    #             if detected_item in self.current_order and self.current_order[detected_item]['quantity'] > 0:
    #                 self.current_order[detected_item]['quantity'] -= 1
    #             self.update_detection(detected_item, category)
    #
    #             # Save the detected information to the CSV
    #             self.save_as_csv(output_data)
    #             logging.info(f"Detection updated and saved for {detected_item}")
    #
    #             # Update the last detection time for this item
    #             self.last_detection_times[detected_item] = current_time
    #         except KeyError as e:
    #             logging.error(f"KeyError when processing detected item: {str(e)}")
    #         except Exception as e:
    #             logging.error(f"Error processing detected item: {str(e)}")
    #
    #     except Exception as e:
    #         logging.error(f"Unexpected error in process_frame: {str(e)}")
    #         logging.error(traceback.format_exc())
    #     finally:
    #         try:
    #             # Clean up: delete the processed image file
    #             os.remove(save_path)
    #             logging.info(f"Processed image file removed: {save_path}")
    #         except Exception as e:
    #             logging.error(f"Error removing processed image file: {str(e)}")

    def process_frame_with_timeout(self, frame, timeout=10):
        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = self.process_frame(frame)
            except Exception as e:
                exception = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            _thread.interrupt_main()  # Raise KeyboardInterrupt in main thread
            thread.join()  # Wait for thread to finish
            logging.error("process_frame timed out")
            return None

        if exception:
            print("not cvoool")
            raise exception

        return result

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
        self.freshness_label.configure(text=f"Freshness: {freshness}%")

    def update_gui(self, frame):
        # Convert the BGR frame from OpenCV to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(frame_rgb)
        self.ax.axis('off')
        self.canvas.draw()

    def run_object_detection(self):
        while not self.stop_video:
            try:
                vs = cv2.VideoCapture(0)  # Try camera index 1
                if not vs.isOpened():
                    vs = cv2.VideoCapture(1)  # If 1 fails, try camera index 0

                if not vs.isOpened():
                    messagebox.showerror("Error", "Failed to open camera. Please check your camera connection.")
                    break

                while not self.stop_video:
                    ret, frame = vs.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                    self.event_detector.update_buffer(frame)
                    event_detected, _ = self.event_detector.detect_event(frame)

                    if event_detected:
                        event_frames = self.event_detector.get_event_frames()
                        for event_frame in event_frames:
                            # Check if a person is detected or if the advanced filter is passed
                            person_not_detected = not self.mobilenet_detector.detect_person(event_frame)
                            passed_filter = self.advanced_filter is None or \
                                            self.advanced_filter.filter_image(event_frame)[0]

                            if person_not_detected and passed_filter:
                                print("Person not detected, frame saved")
                                try:
                                    self.solve_frame(event_frame)
                                except KeyboardInterrupt:
                                    logging.warning("process_frame timed out and was interrupted")
                                except Exception as e:
                                    logging.error(f"Error in process_frame: {str(e)}")
                            else:
                                print(
                                    "Person detected, frame discarded" if not person_not_detected else "Frame discarded due to filter")

                            # Wait for 3 seconds before processing the next frame
                            time.sleep(1)

                    # Update GUI with the current frame
                    self.update_gui(frame)

            except cv2.error as e:
                print(f"OpenCV error: {str(e)}")
                messagebox.showerror("Error", f"An error occurred: {str(e)}\nTrying to reinitialize camera...")
                time.sleep(1)  # Wait a bit before trying again
            finally:
                if vs is not None and vs.isOpened():
                    vs.release()

        print("Object detection stopped")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    # Load the model and processor once
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    root = ctk.CTk()
    # root = tk.Tk()
    app = GroceryScannerApp(root)
    root.mainloop()
