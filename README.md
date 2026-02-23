# 🎙️ Audio Locator: Real-Time Spatial Audio Localization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge&logo=creative-commons" alt="License">
</p>

**Audio Locator** is a neural network based model designed to identify and track the azimuth of sound sources in real-time. 

The system generates synthetic data using HRTFs to simulate realistic sound propagation in a 3D space, then trains on it until it learns to precisely predict the azimuth of sound sources.

---

## 🌟 Key Features

- **🚀 Real-Time Localization**: Low-latency inference capable of running at >50Hz refresh rates.
- **🎧 HRTF Understanding**: Uses slab's HRIRs for realistic data generation with linear and circular trajectories around the listener.

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```powershell
# Clone the repository
git clone https://github.com/xFrah/audio_locator.git
cd audio_locator

# Install dependencies and create venv
uv sync
```

---

### 📉 Training
Start training with the custom HRTF-based dataset generator:

```powershell
uv run train.py
```
*The training script includes a live comparison plot to monitor model performance against ground truth.*



### ⚡ Real-Time Evaluation
Simulate real-time processing on an existing audio file:

```powershell
uv run eval_realtime.py
```

### 🔨 Data Generation
To generate a standalone dataset for debugging:

```powershell
uv run dataset.py
```

---

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. 

See the [LICENSE](LICENSE) file for the full text.
