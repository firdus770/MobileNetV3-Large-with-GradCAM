import os
import subprocess
import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

# Download ResNet18 model from Google Drive
MODEL_PATH = "best.pt"
MODEL_ID = "1-bSdWUkeEASlu1KFveeFo-R3gaWrwPoY"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading MobileNetV3 Large from Google Drive...")
    try:
        subprocess.run(["gdown", "--id", MODEL_ID, "--output", MODEL_PATH], check=True)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

# Load model and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, len(class_labels))

try:
    resnet_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

resnet_model.eval().to(device)

# Grad-CAM and image transform setup
cam_extractor = GradCAM(resnet_model, target_layer="layer4")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit App UI
st.title("Otoscopic Classifier with Grad-CAM (ResNet18)")

uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = resnet_model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    st.success(f"Prediction: {class_labels[pred_class]}")
    st.info(f"Confidence: {confidence:.2f}")

    # Grad-CAM heatmap
    activation_map = cam_extractor(pred_class, output)
    heatmap_tensor = activation_map[0].detach().cpu()
    heatmap_pil = to_pil_image(heatmap_tensor)
    heatmap_resized = heatmap_pil.resize(image.size)
    cam_image = overlay_mask(image, heatmap_resized, alpha=0.5)
    st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
