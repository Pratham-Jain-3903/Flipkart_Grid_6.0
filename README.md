---

# Smart Retail Scanner Pro 🛒

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An intelligent retail automation system that leverages computer vision and machine learning to streamline inventory management and enhance the shopping experience. Perfect for modern retail environments looking to automate their operations while maintaining accuracy and efficiency.

## 🎥 Demo

[![Watch the video](path/to/github_assets/2.png)](https://drive.google.com/file/d/15eoGBw3SjDriYymjUkb_L5rgUbx-RZPX/view?usp=drive_link)

## ✨ Features

- **Real-time Product Detection**: Automated scanning and recognition of products using computer vision
- **One-Click Interface**: User-friendly for warehouse staff, allowing quick switching between automatic and manual modes
- **User Alerts**: Real-time notifications for incorrect, missing, extra, or expired products in orders
- **Smart Inventory Management**: Tracks product movement and updates inventory in real-time
- **Freshness Detection**: AI-powered OCR-based assessment for perishable goods and expiration detection for packaged products
- **Scalability**: Efficient deployment across all product categories with minimal hardware upgrades
- **Hybrid Cloud-Edge Model**: Uses low-cost edge devices for real-time inference and cloud for centralized logging and analytics
- **Logging System**: Continuous logging of order and item data, helping identify frequently mis-scanned products
- **User-Friendly Interface**: Modern dark-themed UI built with CustomTkinter for ease of use

## 🛠️ Tech Stack

- **Frontend**: CustomTkinter, Matplotlib
- **Backend**: Python 3.8+
- **Computer Vision**: OpenCV, imutils
- **ML/AI**:
  - PyTorch (Quantized models for edge deployment)
  - Qwen-VL2-2B (fine-tuned for OCR-based freshness detection)
  - MobileNet for human filtering
- **Database**: SQLite3
- **Edge Computing**: Compatible with NVIDIA Jetson Nano and Raspberry Pi
- **Cloud Integration**: Configurable cloud logging to minimize idle server times
- **Deployment**: Docker containers for easy scaling
- **Monitoring**: Custom logging and performance metrics

## 🚀 Getting Started

### Prerequisites

- GPU with at least 4GB VRAM is recommended.

```bash
python -m pip install -r requirements.txt
```

### Running the Application

```bash
python GUI_Grocery_tracker_filter.py
```

## 🏗️ Architecture

The system operates on a hybrid edge-cloud architecture:

1. **Edge Layer** (Jetson Nano/Raspberry Pi):
   - Real-time video processing and product recognition
   - Local caching and preprocessing
   - Quantized ML models for efficient inference with just 8GB RAM and 4GB VRAM

2. **Cloud Layer**:
   - Centralized logging and data analytics
   - Model updates and inventory synchronization
   - Storage and processing for long-term analytics

![Dataflow Diagram](path/to/github_assets/5.png)

## 📊 Performance

- **Average Detection Time**: <500ms
- **Detection Accuracy**: F1 score of 0.832, with >95% accuracy in controlled environments
- **Camera Support**: Up to 4 simultaneous feeds
- **Memory Footprint**: <2GB on edge devices

![Did you know we can ?!](path/to/github_assets/4.png)

## 📸 Interface

![Interface](path/to/github_assets/9.png)

## 🔒 Security Features

- **Data Transmission**: Encrypted and secure
- **Role-Based Access Control**: Ensures only authorized access
- **Audit Logging**: Keeps detailed logs for analysis and security
- **Cloud Integration**: Fully secure and configurable
Here’s an expanded section on challenges and solutions for the README:

---

## 🏆 Challenges and Solutions

### 1. **Scalability**
   - **Challenge**: Our system needed to handle a wide range of product categories and locations, especially during peak times, with minimal hardware upgrades. Given Flipkart’s large operational scale, the solution also had to be easily deployable and adaptable across different warehouse setups.
   - **Solution**: We designed a scalable hybrid cloud-edge model that efficiently supports high traffic volumes. By deploying quantized, fine-tuned Qwen-VL models on low-cost edge devices (e.g., Jetson Nano), the system can handle real-time inferences at the warehouse level, reducing reliance on central servers. This edge-centric approach minimizes latency and optimizes resource allocation, ensuring each warehouse can operate independently with minimal intervention from centralized cloud services. Additionally, the cloud component is designed to handle long-term storage, analytics, and model updates, seamlessly syncing with edge devices and keeping infrastructure upgrades to a minimum.

### 2. **Usability**
   - **Challenge**: Warehouse staff may have limited technical training, so the system needed to be highly intuitive, user-friendly, and reliable, requiring minimal interaction while handling complex tasks.
   - **Solution**: We developed a one-click interface with a dark-themed UI for easy readability and real-time response. The interface allows staff to switch between automatic and manual modes instantly, view order details, and monitor the status of each item in real-time. Additionally, we incorporated real-time user alerts that flag any incorrect, missing, extra, or expired products, enabling staff to make quick corrections. This simplified UI design ensures that minimal training is required, allowing staff to focus on operational efficiency without being distracted by complex system interactions.

### 3. **Infrastructure Efficiency**
   - **Challenge**: Given Flipkart’s diverse infrastructure, the system needed to optimize existing resources without requiring extensive additional investment.
   - **Solution**: Our hybrid model leverages both edge and cloud resources to maximize efficiency. By performing real-time inferencing on edge devices, we reduce the load on central servers, which in turn lowers idle server time and operational costs. A TCP connection ensures smooth data transfer between the edge and cloud, allowing for seamless data logging and analysis while keeping network usage optimized. Furthermore, our cloud-based logging system supports centralized monitoring and reporting, helping to streamline operations and reduce waste from unnecessary compute cycles. This approach minimizes the need for frequent upgrades and better utilizes the available resources across Flipkart's distributed network.

--- 

This expanded section describes the specific challenges encountered and how the system architecture and design choices directly address each one.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community
- PyTorch team
- Anthropic's Claude for documentation assistance
- CustomTkinter developers

## 📞 Contact

For any queries regarding implementation or collaboration, please reach out to Prathamjain3903@gmail.com

--- 
