import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from datetime import datetime
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

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

# FruitCounter for tracking "Soda" category items
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

        if category == "Soda" and self.is_inside_box(x, y):
            if object_id not in self.object_history:
                self.object_history[object_id] = {"first_seen": current_time, "last_counted": None}

            time_diff = (current_time - self.object_history[object_id]["first_seen"]).total_seconds()
            
            if self.object_history[object_id]["last_counted"] is None or \
               (current_time - self.object_history[object_id]["last_counted"]).total_seconds() > 5:
                if confidence > 0.7:
                    self.count += 1
                    self.object_history[object_id]["last_counted"] = current_time
                    return True, self.count

        return False, self.count

class FreshnessDetector:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.image_folder = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images"

    def process_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((512, 512))
        return image

    def generate_response(self, image, text_query):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_query}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output_text[0]

    def check_freshness(self, category):
    #    if category in ["Beverages"]:
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if image_files:
            image_path = os.path.join(self.image_folder, image_files[0])
            image = self.process_image(image_path)
            today_date = datetime.now().strftime("%Y-%m-%d")
        #    output_text = self.generate_response(image, "Give the freshness of the fruit on a scale of 1 to 100% direct answer one word eg:30")
            output_text = self.generate_response(image, f"Analyze the image taken on {today_date} and identify if it contains a fruit, vegetable, or egg. If it is a fruit or vegetable, evaluate its freshness on a scale from 1 to 100. If it is not a fruit or vegetable, determine if the product is expired. Return 100 if the product is not expired, or 0 if it is expired. If the image contains an egg, return 0 if the egg is broken or damaged, and return 100 if the egg is intact.")

            print(f"Detected Freshness: {output_text}")
            return output_text
        else:
            print("No image files found.")
            return "No image available"
    #    return 100


def write_to_csv(data):
    csv_filename = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\log.csv"
    with open(csv_filename, 'a', newline='') as file:
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
        self.current_order = []
        self.order_status = "Not Started"

        self.create_gui()
        self.create_db()

    def create_gui(self):
        self.root.geometry("1300x800")

        # Optional: Prevent resizing the window
        # self.root.resizable(True, True)
        self.style = ttk.Style()
        self.style.theme_use('vista')

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

        self.order_status_label = ttk.Label(order_frame, text="Order Status: Not Started")
        self.order_status_label.grid(row=0, column=0, columnspan=2, pady=5)

        generate_order_button = ttk.Button(order_frame, text="Generate Order", command=self.generate_order)
        generate_order_button.grid(row=1, column=0, padx=5)

        complete_order_button = ttk.Button(order_frame, text="Complete Order", command=self.complete_order)
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
        self.category_combobox = ttk.Combobox(item_entry_frame, values=[
            "Fruits and Vegetables",
            "Staples",
            "Snacks",
            "Beverages",
            "Packed Food",
            "Personal and Baby Care",
            "Household Care",
            "Dairy and Eggs",
            "Home and Kitchen"])
        self.category_combobox.grid(row=1, column=1, padx=5, pady=5)

        add_item_button = ttk.Button(item_entry_frame, text="Add Item", command=self.add_item)
        add_item_button.grid(row=2, column=0, columnspan=2, pady=5)

        manual_button = ttk.Button(item_entry_frame, text="Manual Put Items (Camera Malfunction)", command=self.manual)
        manual_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Item Information
        info_frame = ttk.LabelFrame(right_frame, text="Item Information", padding="10")
        info_frame.grid(row=2, column=0, sticky="ew", pady=10)

        self.freshness_label = ttk.Label(info_frame, text="Freshness: N/A")
        self.freshness_label.grid(row=0, column=0)

        self.fruits_and_vegetables_count_label = ttk.Label(info_frame, text="Fruits and Vegetables: 0")
        self.fruits_and_vegetables_count_label.grid(row=1, column=0)

        self.staples_count_label = ttk.Label(info_frame, text="Staples: 0")
        self.staples_count_label.grid(row=2, column=0)

        self.snacks_count_label = ttk.Label(info_frame, text="Snacks: 0")
        self.snacks_count_label.grid(row=3, column=0)

        self.beverages_count_label = ttk.Label(info_frame, text="Beverages: 0")
        self.beverages_count_label.grid(row=4, column=0)

        self.packed_food_count_label = ttk.Label(info_frame, text="Packed Food: 0")
        self.packed_food_count_label.grid(row=5, column=0)

        self.personal_and_baby_care_count_label = ttk.Label(info_frame, text="Personal and Baby Care: 0")
        self.personal_and_baby_care_count_label.grid(row=6, column=0)

        self.household_care_count_label = ttk.Label(info_frame, text="Household Care: 0")
        self.household_care_count_label.grid(row=7, column=0)

        self.dairy_and_eggs_count_label = ttk.Label(info_frame, text="Dairy and Eggs: 0")
        self.dairy_and_eggs_count_label.grid(row=8, column=0)

        self.home_and_kitchen_count_label = ttk.Label(info_frame, text="Home and Kitchen: 0")
        self.home_and_kitchen_count_label.grid(row=9, column=0)

        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.grid(row=3, column=0, sticky="ew", pady=10)

        start_icon = self.load_icon("power_button.png")
        stop_icon = self.load_icon("no.png")

        start_button = ttk.Button(control_frame, text="Start Detection", image=start_icon, compound=tk.LEFT,
                                  command=self.start_detection)
        start_button.image = start_icon
        start_button.grid(row=0, column=0, padx=5)

        stop_button = ttk.Button(control_frame, text="Stop Detection", image=stop_icon, compound=tk.LEFT,
                                 command=self.stop_detection)
        stop_button.image = stop_icon
        stop_button.grid(row=0, column=1, padx=5)

    def load_icon(self, filename):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, "icons", filename)
            return ImageTk.PhotoImage(Image.open(icon_path).resize((20, 20)))

    def create_db(self):
        conn = sqlite3.connect('grocery_items.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS items
                     (name TEXT, category TEXT, freshness INTEGER, date TEXT)''')
        conn.commit()
        conn.close()

    def add_item_to_db(self, name, category, freshness):
        conn = sqlite3.connect('grocery_items.db')
        c = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO items VALUES (?, ?, ?, ?)", (name, category, freshness, date))
        conn.commit()
        conn.close()

    # def check_freshness(self, category):
    #     if category in ["Fruits and Vegetables"]:
    #         freshness = random.randint(10, 100)
    #         if freshness < 33:
    #             messagebox.showwarning("Low Freshness", f"Warning: {category} freshness is low ({freshness}%)")
    #         return freshness
    #     return 100
    
    def update_count_display(self):
        self.fruits_and_vegetables_count_label.config(text=f"Fruits and Vegetables: {self.fruits_and_vegetables_count}")
        self.staples_count_label.config(text=f"Staples: {self.staples_count}")
        self.snacks_count_label.config(text=f"Snacks: {self.snacks_count}")
        self.beverages_count_label.config(text=f"Beverages: {self.beverages_count}")
        self.packed_food_count_label.config(text=f"Packed Food: {self.packed_food_count}")
        self.personal_and_baby_care_count_label.config(text=f"Personal and Baby Care: {self.personal_and_baby_care_count}")
        self.household_care_count_label.config(text=f"Household Care: {self.household_care_count}")
        self.dairy_and_eggs_count_label.config(text=f"Dairy and Eggs: {self.dairy_and_eggs_count}")
        self.home_and_kitchen_count_label.config(text=f"Home and Kitchen: {self.home_and_kitchen_count}")


    def update_count(self, category):
        if category == "Fruits and Vegetables":
            self.fruit_count += 1
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
            self.scanned_items_listbox.insert(tk.END, f"{item} ({details['category']}) x{details['quantity']}")

    def add_item(self):
        name = self.item_name_entry.get()
        category = self.category_combobox.get()
        if name and category:
            # freshness = self.freshness_detector.check_freshness(category)

            # Check if the item is already in the current order
            if name in self.current_order:
                self.current_order[name]['quantity'] += 1  # Increment quantity
            else:
                self.current_order[name] = {'category': category, 'quantity': 1}  # Add new item

            self.update_order_display()
            # self.add_item_to_db(name, category, freshness)
            # self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"{name} added successfully!")
        else:
            messagebox.showwarning("Input Error", "Please enter both item name and category.")

    def manual(self):
        name = self.item_name_entry.get()
        category = self.category_combobox.get()
        if name and category:
            freshness = self.freshness_detector.check_freshness(category)

            # Check if the item is already in the current order
            if name in self.current_order:
                self.current_order[name]['quantity'] -= 1  # decrease quantity
            else:
                self.current_order[name] = {'category': category, 'freshness': freshness, 'quantity': 1}  # Add new item

            self.update_order_display()
            self.add_item_to_db(name, category, freshness)
            self.freshness_label.config(text=f"Freshness: {freshness}%")
            self.item_name_entry.delete(0, tk.END)
            messagebox.showinfo("Success", f"{name} added successfully!")
        else:
            messagebox.showwarning("Input Error", "Please enter both item name and category.")

    def start_detection(self):
        self.stop_video = False
        threading.Thread(target=self.run_object_detection, daemon=True).start()
        messagebox.showinfo("Detection Started", "Object detection has been started.")

    def stop_detection(self):
        self.stop_video = True
        messagebox.showinfo("Detection Stopped", "Object detection has been stopped.")

    def demo_order(self):
        # Simulate a hardcoded order with 2 bottles and 1 TV monitor
        order_items = [
            ("Soda", "Other", 2),   # (Item Name, Category, Quantity)
            ("TV Monitor", "Other", 1)
        ]

        # Clear the current listbox to start fresh
        self.scanned_items_listbox.delete(0, tk.END)

        for item_name, category, quantity in order_items:
            for _ in range(quantity):
                freshness = FreshnessDetector.check_freshness(category)

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

    def update_detection(self, item, category):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.video_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack()
        self.ax.clear()
        self.ax.text(0.5, 0.5, f"Detected: {item}", horizontalalignment='center', verticalalignment='center')
        self.canvas.draw()

        self.scanned_items_listbox.insert(tk.END, f"{item} ({category})")
        self.update_count(category)
        freshness = FreshnessDetector.check_freshness(category)
        self.freshness_label.config(text=f"Freshness: {freshness}%")
    
    def generate_order(self):
        if self.order_status != "Not Started":
            messagebox.showwarning("Order in Progress", "Please complete the current order before generating a new one.")
            return

        # Generate a random order      
        items = ["Rice", "Flour", "Chips", "Nuts", "Juice", "Soda", "Canned Beans", "Pasta", "Diapers", "Shampoo", "Detergent", "Disinfectant", "Yogurt", "Cheese", "Cutlery", "Cookware","Banana","Guava"]
        categories = ["Staples", "Staples", "Snacks", "Snacks", "Beverages", "Beverages", "Packed Food", "Packed Food", "Personal and Baby Care", "Personal and Baby Care", "Household Care", "Household Care", "Dairy and Eggs", "Dairy and Eggs", "Home and Kitchen", "Home and Kitchen","Fruits and Vegetables"]

        # order_size = random.randint(3, 7)
        # TypeError: list indices must be integers or slices, not str
        # self.current_order = [(items[i], categories[i], random.randint(1, 3)) for i in random.sample(range(len(items)), order_size)]

        # # Display the order
        # self.scanned_items_listbox.delete(0, tk.END)
        # for item, category, quantity in self.current_order:
        #     self.scanned_items_listbox.insert(tk.END, f"{item} ({category}) x{quantity}")

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
                self.current_order[item]['quantity'] += quantity
            else:
                self.current_order[item] = {'category': category, 'quantity': quantity}

        # Display the order
        self.scanned_items_listbox.delete(0, tk.END)
        for item, details in self.current_order.items():
            category = details['category']
            quantity = details['quantity']
            self.scanned_items_listbox.insert(tk.END, f"{item} ({category}) x{quantity}")

        self.order_status = "In Progress"
        self.order_status_label.config(text="Order Status: In Progress")
        messagebox.showinfo("Order Generated", "A new order has been generated. Start scanning items to fulfill the order.")

    # def complete_order(self):
    #     if self.order_status != "In Progress":
    #         messagebox.showwarning("No Active Order", "There is no active order to complete.")
    #         return

    #     # Check if all items in the order have been scanned
    #     scanned_items = {item.split(" (")[0] for item in self.scanned_items_listbox.get(0, tk.END)}

    #     for item in list(self.current_order.keys()):
    #         if item not in scanned_items:
    #             messagebox.showwarning("Incomplete Order", f"The following item is missing: {item}")
    #             return
    #         else:
    #             # Decrease quantity or remove if quantity is zero
    #             self.current_order[item]['quantity'] -= 1
    #             if self.current_order[item]['quantity'] <= 0:
    #                 del self.current_order[item]  # Remove item from current order

    #     # Order is complete
    #     self.order_status = "Completed"
    #     self.order_status_label.config(text="Order Status: Completed")
    #     messagebox.showinfo("Order Completed", "The order has been successfully completed!")

    #     # Reset for the next order
    #     self.current_order = {}
    #     self.detected_items_set.clear()
    #     self.scanned_items_listbox.delete(0, tk.END)
    #     self.fruits_and_vegetables_count = 0
    #     self.staples_count = 0
    #     self.snacks_count = 0
    #     self.beverages_count = 0
    #     self.packed_food_count = 0
    #     self.personal_and_baby_care_count = 0
    #     self.household_care_count = 0
    #     self.dairy_and_eggs_count = 0
    #     self.home_and_kitchen_count = 0
    #     self.update_count_display()
    #     self.freshness_label.config(text="Freshness: N/A")

    #     self.order_status = "Not Started"
    #     self.order_status_label.config(text="Order Status: Not Started")


    def complete_order(self):
        if self.order_status != "In Progress":
            messagebox.showwarning("No Active Order", "There is no active order to complete.")
            return

        # Check if all items in the order have been fulfilled
        for item, details in self.current_order.items():
            if details['quantity'] > 0:
                messagebox.showwarning("Incomplete Order", f"The item '{item}' has not been fully added to the order.")
                return

        # Order is complete
        self.order_status = "Completed"
        self.order_status_label.config(text="Order Status: Completed")
        messagebox.showinfo("Order Completed", "The order has been successfully completed!")
        data = [datetime.now(),"New Order Started", "Order_Number : 987654321"]
        write_to_csv(data)
        
        # Reset for the next order
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

    def run_object_detection(self): 
        PROTOTXT = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.prototxt.txt"
        MODEL = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\MobileNetSSD_deploy.caffemodel"
        CONFIDENCE = 0.5
        csv_filename = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\log.csv"
        bottle_image_dir = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images"

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "Soda", "bus", "car", "cat", "chair", "cow", "diningtable",
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

                    if label in ["Soda", "tvmonitor"]:  # Check for both "Soda" and "TV monitor"
                        unique_item_identifier = f"{label}-{object_id}"
                        
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        centerX = (startX + endX) // 2
                        centerY = (startY + endY) // 2

                        z = np.array([[centerX], [centerY]])
                        predicted = kf.predict()
                        updated = kf.update(z)

                        is_inside_box, current_count = counter.update(object_id, (centerX, centerY), "Soda", confidence)

                        if is_inside_box:  # Check if the Soda is entering the box
                            # Only add to the list if it's a new detection
                            if unique_item_identifier not in self.detected_items_set:
                                self.detected_items_set.add(unique_item_identifier)  # Add to the set

                                freshness = self.freshness_detector.check_freshness(label)  # Check freshness for the detected item
                                self.freshness_label.config(text=f"Freshness: {freshness}%")
                                self.add_item_to_db(label.capitalize(), "Other", freshness)  # Save to DB

                                # Check if the label is in current_order
                                if label in self.current_order:
                                    if self.current_order[label]['quantity'] > 0:
                                        self.current_order[label]['quantity'] -= 1  # Decrement quantity
                                    else:
                                        # Quantity is 0, so we won't decrement further
                                        self.scanned_items_listbox.insert(tk.END, f"Offer_item {label.capitalize()} ({current_count})")
                                        # Optionally, you might want to initialize this item in current_order
                                        self.current_order[label]['quantity'] = 0
                                else:
                                    # If the label is not in current_order, add it to the scanned_items_listbox
                                    self.scanned_items_listbox.insert(tk.END, f"Offer_item {label.capitalize()} ({current_count})")
                                    # Add the item to current_order with quantity set to 0
                                    self.current_order[label] = {'category': "Other", 'freshness': freshness, 'quantity': 0}


                                    # Update the order display
                                self.update_order_display()
                                    # Update count for items using dictionary for later
                                # self.update_count("Beverages")
                                    # Log to CSV and save frame
                                data = [datetime.now(),label, current_count]
                                write_to_csv(data)
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