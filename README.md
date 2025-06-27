# 🐑 BhedChaal - AI-Powered Crowd Management System

**BhedChaal** is an intelligent crowd monitoring and anomaly detection system that leverages AI and computer vision to ensure safety, predict dangerous crowd behavior, and assist in efficient public management during events, rallies, and gatherings.

## 🚀 Features

- 🎥 **Live Crowd Density Detection** using CSRNet
- 👥 **Real-time Face Detection** with YOLOv11-Face
- 🧠 **Anomaly Detection** for unusual crowd behavior
- 🗺️ **Homography Transformation** for top-view mapping
- 🎮 **Physics-Based Simulation** to predict crowd movement
- 🖼️ **Real-ESRGAN** for image enhancement
- 📲 Web-accessible via camera (desktop/mobile)

---

## 📌 Use Cases

- Kumbh Mela, religious gatherings
- Political rallies, stadiums
- Railway stations, airports
- Disaster management zones
- College fests, large event venues

---

## 🛠️ Tech Stack

| Component              | Tech Used           |
|------------------------|---------------------|
| Frontend               | React               |
| Backend                | Flask               |
| AI/ML Models           | YOLOv11-Face, CSRNet, Anomaly Detector |
| Image Enhancement      | Real-ESRGAN         |
| Simulation             | Physics-based Modeling |
| Deployment             | Web-based (camera access) |

---

## 🧠 Architecture Overview

```plaintext
[Camera Feed] --> [YOLOv11-Face] --> [Crowd Density + Anomaly Detection]
                        ↓
         [Homography Mapping + Top-view Simulation]
                        ↓
         [Real-time Dashboard + Alert System]
