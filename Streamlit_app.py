import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# HUGGINGFACE MODEL LOADER (MATCHING YOUR COLAB SCRIPT)
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(name: str):
    filename_map = {
        "DenseNet121": "densenet121_best.pt",
        "MobileNetV2": "mobilenet_v2_best.pt",
        "VGG16": "vgg16_best.pt",
        "EfficientNetB0": "efficientnet_b0_best.pt",
        "ResNet50": "resnet50_best.pt",
    }
    
    file_name = filename_map.get(name)
    if not file_name:
        raise ValueError(f"Unknown model name: {name}")

    # Fetch weights from Hugging Face
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=file_name)

    # Build exact architecture matching Colab Cell 9
    if name == 'ResNet50':
        m = models.resnet50(weights=None)
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 256), nn.ReLU(),
                              nn.Dropout(0.3), nn.Linear(256, 1))
    elif name == 'VGG16':
        m = models.vgg16(weights=None)
        in_feat = m.classifier[6].in_features
        m.classifier[6] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 256), nn.ReLU(),
                                         nn.Dropout(0.3), nn.Linear(256, 1))
    elif name == 'DenseNet121':
        m = models.densenet121(weights=None)
        in_feat = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 256), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(256, 1))
    elif name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 256), nn.ReLU(),
                                         nn.Dropout(0.3), nn.Linear(256, 1))
    elif name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 256), nn.ReLU(),
                                         nn.Dropout(0.3), nn.Linear(256, 1))

    # Load weights onto correct device
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    m.load_state_dict(state_dict)
    m.to(DEVICE)
    m.eval()

    return m

# ---------------------------------------------------------
# PREPROCESSING PIPELINE (PAD TO SQUARE + CLAHE + MEDIAN)
# ---------------------------------------------------------
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        pad_left = (side - w) // 2
        pad_top = (side - h) // 2
        pad_right = side - w - pad_left
        pad_bottom = side - h - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

class ClaheEnhance:
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        arr = cv2.medianBlur(arr, 3)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(arr)

def get_eval_transform(apply_clahe=True):
    transform_list = [PadToSquare()]
    if apply_clahe:
        transform_list.append(ClaheEnhance())
    transform_list.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transforms.Compose(transform_list)

# ---------------------------------------------------------
# GRAD-CAM IMPLEMENTATION (MATCHING YOUR COLAB CELL 14)
# ---------------------------------------------------------
TARGET_LAYER_GETTER = {
    'ResNet50': lambda m: m.layer4[-1],
    'VGG16': lambda m: m.features[-1],
    'DenseNet121': lambda m: m.features[-1],
    'EfficientNetB0': lambda m: m.features[-1],
    'MobileNetV2': lambda m: m.features[-1],
}

def generate_gradcam(model, img_tensor, name):
    """Generates Grad-CAM without using backward hooks."""
    activations = []

    def forward_hook(module, inp, out):
        # Detach and clone to ensure no in-place modifications interfere with autograd
        activations.append(out)

    target_layer = TARGET_LAYER_GETTER[name](model)
    handle_f = target_layer.register_forward_hook(forward_hook)

    with torch.enable_grad():
        input_var = img_tensor.unsqueeze(0).to(DEVICE)
        input_var.requires_grad = True
        
        # Forward pass
        logits = model(input_var)
        score = logits[0, 0]
        
        # Extract feature maps captured during forward pass
        acts = activations[0]
        
        # Calculate gradients directly on feature maps
        grads = torch.autograd.grad(outputs=score, inputs=acts, retain_graph=True)[0]

    handle_f.remove()

    # Process numpy arrays safely
    grads_np = grads[0].cpu().detach().numpy()
    acts_np = acts[0].cpu().detach().numpy()
    
    weights = np.mean(grads_np, axis=(1, 2))
    
    cam = np.zeros(acts_np.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts_np[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

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
        
        # Display image preview
        with col_left:
            st.image(raw_image, caption="Uploaded Image Crop", use_container_width=True)

        with col_right:
            st.subheader("Diagnostic Prediction")
            
            with st.spinner(f"Fetching `{selected_model}` from Hugging Face (`St0Lexy/breast_model`)..."):
                model = load_trained_model(selected_model)

            eval_transform = get_eval_transform(apply_clahe)
            input_tensor = eval_transform(raw_image)

            # Pass through model for sigmoid probability
                        # Run forward pass cleanly without setting input_tensor.requires_grad
            input_tensor_device = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor_device)
                prob_malignant = torch.sigmoid(logits).item()
                prob_benign = 1.0 - prob_malignant

            # Class determination
            pred_class = "MALIGNANT" if prob_malignant >= 0.5 else "BENIGN"
            confidence = (prob_malignant if pred_class == "MALIGNANT" else prob_benign) * 100

            if pred_class == "MALIGNANT":
                st.error(f"**Predicted Class:** {pred_class}")
            else:
                st.success(f"**Predicted Class:** {pred_class}")

            st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
            
            st.write("---")
            st.write("**Probability Distribution:**")
            st.progress(float(prob_benign), text=f"Benign: {prob_benign*100:.1f}%")
            st.progress(float(prob_malignant), text=f"Malignant: {prob_malignant*100:.1f}%")


            # Render Grad-CAM Heatmap
            st.write("---")
            st.write("**Grad-CAM Explainability Map (Lesion Focus):**")
            try:
                cam = generate_gradcam(model, input_tensor, selected_model)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Format processed image for overlay
                img_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                img_np = np.uint8(255 * img_np)

                overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption="Grad-CAM Lesion Heatmap Overlay", use_container_width=True)
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
        * **Leakage Prevention:** Split strictly at the **patient level** (not image level) using `GroupShuffleSplit` on `patient_id`.
        * **Filtering:** Ambiguous BI-RADS assessment categories (0 and 3) dropped.
        * **Preprocessing:** `PadToSquare` aspect-ratio preservation, CLAHE contrast enhancement, and 3x3 Median noise reduction.
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Training Protocol
        * **Strategy:** Two-Stage Transfer Learning.
        * **Stage 1:** Frozen backbone, custom classification head (`Dropout(0.4) -> Linear(256) -> Dropout(0.3) -> Linear(1)`) trained for 6 epochs (`lr=1e-3`).
        * **Stage 2:** Unfrozen full network fine-tuned for 25 epochs (`lr=1e-5`) with early stopping on validation AUC.
        * **Imbalance Handling:** `pos_weight` applied in `BCEWithLogitsLoss`.
        * **Explainability:** Integrated Grad-CAM registered at the final convolutional layer of each backbone.
        """)
