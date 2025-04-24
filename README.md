# 🖼️ 2D to 3D Image Conversion App

This web application allows users to convert 2D images into interactive 3D point clouds using deep learning. Built with **Streamlit** and powered by **MiDaS**, a state-of-the-art monocular depth estimation model from Intel-ISL, the app visualizes depth data in real time using **Plotly** and allows downloads in `.ply` format.

## 🚀 Features

- 📤 Upload `.jpg`, `.jpeg`, or `.png` images
- ⚙️ Depth estimation using pre-trained MiDaS model (PyTorch)
- 🧠 3D point cloud generation from 2D image depth map
- 🔁 Real-time interactive 3D visualization using Plotly
- 💾 Export 3D model as `.ply` file for use in Blender, MeshLab, etc.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, PyTorch, MiDaS (torch.hub)
- **Visualization:** Plotly (Scatter3d)
- **Tools:** NumPy, PIL, VS Code, Torch Hub

## 🧩 How It Works

1. Upload an image using the Streamlit UI.
2. MiDaS model generates a depth map from the image.
3. Each pixel is projected into 3D space using camera projection logic.
4. RGB values are mapped to 3D coordinates.
5. Point cloud is rendered and made available for download.

## 📂 Installation & Run

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
git clone https://github.com/your-username/2d-to-3d-app.git
cd 2d-to-3d-app
pip install -r requirements.txt
streamlit run app.py


🧠 Learnings & Future Scope
This project provided hands-on experience with deploying deep learning models, working with image depth estimation, and building intuitive data apps. Future enhancements may include:

Mesh reconstruction and smoothing

Support for video input

Integration with AR/VR platforms

🤝 Authors
Vaishnavi S

Rakshana P. B

Saai Charan

Sandmanleo E
