import os
import subprocess
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

# Download MobileNetV3-Large model weights from Google Drive
MODEL_PATH = "best.pt"
MODEL_ID = "1-bSdWUkeEASlu1KFveeFo-R3gaWrwPoY"  # <-- just the file ID

if not os.path.exists(MODEL_PATH):
    st.info("Downloading MobileNetV3-Large weights from Google Drive...")
    try:
        subprocess.run(["gdown", "--id", MODEL_ID, "--output", MODEL_PATH], check=True)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

# Load model and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

# Build MobileNetV3-Large and match classifier to your label count
mobilenet_model = models.mobilenet_v3_large(weights=None)  # no ImageNet download
mobilenet_model.classifier[-1] = nn.Linear(
    mobilenet_model.classifier[-1].in_features,
    len(class_labels)
)

# Load checkpoint (plain state_dict or wrapped; also handle DDP 'module.' prefix)
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict):
    for k in ("state_dict", "model_state_dict", "net", "model"):
        if k in state and isinstance(state[k], dict):
            state = state[k]
            break

try:
    mobilenet_model.load_state_dict(state, strict=True)
except RuntimeError:
    stripped = {k.replace("module.", ""): v for k, v in state.items()}
    mobilenet_model.load_state_dict(stripped, strict=True)

mobilenet_model.eval().to(device)

# Grad-CAM target layer for MobileNetV3-Large (last conv block)
try:
    cam_extractor = GradCAM(mobilenet_model, target_layer="features.16")
except Exception:
    cam_extractor = GradCAM(mobilenet_model, target_layer="features.15")

# Image transform (include ImageNet normalization like your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Streamlit App UI
st.title("Otoscopic Classifier with Grad-CAM (MobileNetV3-Large)")

uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict (no torch.no_grad() here so TorchCAM can use gradients)
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = mobilenet_model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    # Guard for label index
    label = class_labels[pred_class] if pred_class < len(class_labels) else f"class_{pred_class}"
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")

    # Grad-CAM heatmap
    activation_maps = cam_extractor(class_idx=pred_class, scores=output)
    heatmap_tensor = activation_maps[0].detach().cpu()
    heatmap_pil = to_pil_image(heatmap_tensor)
    heatmap_resized = heatmap_pil.resize(image.size)
    cam_image = overlay_mask(image, heatmap_resized, alpha=0.5)
    st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
