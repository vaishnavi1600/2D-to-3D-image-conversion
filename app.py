import streamlit as st
from PIL import Image
import numpy as np
import torch
import plotly.graph_objs as go

st.set_page_config(page_title="2D to 3D Viewer", layout="wide")

st.title("🖼️ 2D to 3D Point Cloud Viewer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    @st.cache_resource
    def load_midas_model():
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        return midas, transform

    midas, transform = load_midas_model()

    # Convert PIL Image to numpy array and then to tensor
    img_np = np.array(image)
    input_tensor = transform(img_np)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    h, w = depth.shape
    fx, fy = 1.0, 1.0
    cx, cy = w / 2, h / 2

    points, colors, original_coords = [], [], []
    for y in range(0, h, 4):  # Lower resolution for speed
        for x in range(0, w, 4):
            z = depth[y, x]
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            r, g, b = img_np[y, x]
            points.append([X, -Y, z])
            colors.append(f'rgb({r},{g},{b})')
            original_coords.append((y, x))  # Store original image coordinates

    # Extract XYZ
    x_vals, y_vals, z_vals = zip(*points)

    # 3D Plot
    scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=1.5,
            color=colors,
            opacity=0.8
        )
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    fig = go.Figure(data=[scatter], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Save PLY
    ply_filename = "output_model.ply"
    with open(ply_filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, (pt, (y, x)) in enumerate(zip(points, original_coords)):
            r, g, b = img_np[y, x]  # Use stored original coordinates
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {r} {g} {b}\n")

    with open(ply_filename, "rb") as f:
        st.download_button("⬇️ Download 3D Model (.ply)", f, file_name=ply_filename)

    st.success("✅ View and download ready!")
