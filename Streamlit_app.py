import streamlit as st
import torch
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Mammogram Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI-Enabled Breast Cancer Mammogram Classifier")
st.markdown("Comparing **5 Pretrained Backbones** evaluated under strict patient-level leakage control.")

REPO_ID = "St0Lexy/breast_model"

# ---------------------------------------------------------
# HUGGINGFACE MODEL LOADER (DEFINED FIRST TO PREVENT NameError)
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(model_name: str, device: str = "cpu"):
    """
    Downloads checkpoint from Hugging Face Hub, initializes matching architecture, 
    and returns model in eval mode.
    """
    filename_map = {
        "DenseNet121": "densenet121_best.pth",
        "MobileNetV2": "mobilenet_v2_best.pth",
        "VGG16": "vgg16_best.pth",
        "EfficientNetB0": "efficientnet_b0_best.pth",
        "ResNet50": "resnet50_best.pth",
    }
    
    file_name = filename_map.get(model_name)
    if not file_name:
        raise ValueError(f"Unknown model name: {model_name}")

    # Fetch weights from Hugging Face
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=file_name)

    # Initialize correct network structure based on model type
    if model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )
    elif model_name == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )

    # Load parameters safely across CPU/GPU
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------
def apply_clahe_and_filtering(pil_img):
    """Applies CLAHE and 3x3 Median Filter (Chapter 3)."""
    img_np = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    filtered = cv2.medianBlur(enhanced, 3)
    rgb_enhanced = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_enhanced)

# ---------------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# ---------------------------------------------------------
def generate_gradcam(model, input_tensor, target_class, model_name):
    """Generates coarse spatial activation heatmaps via hooks."""
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target last conv layer based on architecture
    if model_name == "DenseNet121":
        target_layer = model.features.denseblock4
    elif model_name == "MobileNetV2":
        target_layer = model.features[-1]
    elif model_name == "VGG16":
        target_layer = model.features[28]
    elif model_name == "EfficientNetB0":
        target_layer = model.features[-1]
    elif model_name == "ResNet50":
        target_layer = model.layer4[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward pass for predicted class
    score = output[0, target_class]
    score.backward()

    # Process gradients
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = feature_maps[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * f_maps[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    handle_f.remove()
    handle_b.remove()
    
    return cam

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🩺 Clinical Assistant (Inference)", "📊 Model Comparison Matrix", "ℹ️ System Methodology"])

# ---------------------------------------------------------
# TAB 1: SINGLE IMAGE INFERENCE
# ---------------------------------------------------------
with tab1:
    st.subheader("Single Image Diagnostic Assistant")
    
    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_file = st.file_uploader("Upload Mammogram Crop (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        apply_clahe = st.checkbox("Apply CLAHE & Median Filter Preprocessing", value=True)
        selected_model = st.selectbox("Select Model Architecture", ["DenseNet121", "MobileNetV2", "VGG16", "EfficientNetB0", "ResNet50"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        processed_image = apply_clahe_and_filtering(raw_image) if apply_clahe else raw_image.convert("RGB")

        with col_left:
            st.image(processed_image, caption="Processed Input Image", use_container_width=True)

        with col_right:
            st.subheader("Diagnostic Prediction")
            
            with st.spinner(f"Loading `{selected_model}` weights from Hugging Face (`St0Lexy/breast_model`)..."):
                model = load_trained_model(selected_model)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(processed_image).unsqueeze(0)

            # Enable grad computation for Grad-CAM
            input_tensor.requires_grad = True
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            benign_prob, malignant_prob = probs[0].item(), probs[1].item()

            classes = ["BENIGN", "MALIGNANT"]
            target_class = torch.argmax(probs).item()
            pred_class = classes[target_class]
            confidence = max(benign_prob, malignant_prob) * 100

            if pred_class == "MALIGNANT":
                st.error(f"**Predicted Class:** {pred_class}")
            else:
                st.success(f"**Predicted Class:** {pred_class}")

            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            
            st.write("---")
            st.write("**Probability Distribution:**")
            st.progress(float(benign_prob), text=f"Benign: {benign_prob*100:.1f}%")
            st.progress(float(malignant_prob), text=f"Malignant: {malignant_prob*100:.1f}%")

            # Render Grad-CAM Heatmap
            st.write("---")
            st.write("**Grad-CAM Explainability Map (Lesion Focus):**")
            try:
                cam = generate_gradcam(model, input_tensor, target_class, selected_model)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                orig_resized = np.array(processed_image.resize((224, 224)))
                overlay = cv2.addWeighted(orig_resized, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render Grad-CAM heatmap: {e}")

# ---------------------------------------------------------
# TAB 2: MODEL BENCHMARKING
# ---------------------------------------------------------
with tab2:
    st.subheader("Test Set Performance Summary (Patient-Level Split)")
    
    st.dataframe(
        {
            "Model Architecture": ["DenseNet121", "MobileNetV2", "VGG16", "EfficientNetB0", "ResNet50"],
            "Accuracy": [0.721, 0.721, 0.719, 0.709, 0.686],
            "Sensitivity (Recall)": [0.770, 0.750, 0.660, 0.625, 0.535],
            "Specificity": [0.679, 0.696, 0.768, 0.781, 0.814],
            "AUC-ROC": [0.798, 0.789, 0.796, 0.793, 0.772],
        },
        use_container_width=True
    )
    st.info("💡 **Key Takeaway:** DenseNet121 achieved the highest sensitivity (77.0%) and AUC-ROC (0.798), while MobileNetV2 achieved an identical accuracy (72.1%) with significantly lower computational overhead.")

# ---------------------------------------------------------
# TAB 3: SYSTEM METHODOLOGY
# ---------------------------------------------------------
with tab3:
    st.subheader("System Architecture & Experimental Methodology")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🧪 Data Pipeline
        * **Dataset:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM).
        * **Leakage Prevention:** Split strictly at the **patient level** (not image level) to prevent data leakage and performance inflation.
        * **Filtering:** Ambiguous BI-RADS categories (0 and 3) were excluded.
        * **Preprocessing:** CLAHE contrast enhancement + 3x3 Median noise reduction.
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Training Protocol
        * **Strategy:** Two-Stage Transfer Learning.
        * **Stage 1:** Frozen backbone, classification head trained for 10 epochs.
        * **Stage 2:** Unfrozen top blocks fine-tuned with Adam optimizer ($lr=10^{-5}$).
        * **Explainability:** Integrated Grad-CAM for qualitative decision verification.
        """)
