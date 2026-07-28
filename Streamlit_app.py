import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Mammogram Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI-Enabled Breast Cancer Mammogram Classifier")
st.markdown("Comparing **5 Pretrained Backbones** evaluated under strict patient-level leakage control.")

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🩺 Clinical Assistant (Inference)", "📊 Model Comparison Matrix", "ℹ️ System Methodology"])

# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------
def apply_clahe_and_filtering(pil_img):
    """Applies CLAHE and 3x3 Median Filter as used in Chapter 3."""
    img_np = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    filtered = cv2.medianBlur(enhanced, 3)
    rgb_enhanced = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_enhanced)

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
            st.image(processed_image, caption="Input Image (Ready for Model)", use_container_width=True)

        with col_right:
            st.subheader("Diagnostic Prediction")
            
            # Load model from Hugging Face
            with st.spinner(f"Fetching `{selected_model}` weights from Hugging Face (`St0Lexy/breast_model`)..."):
                model = load_trained_model(selected_model)

            # Preprocessing transform for PyTorch
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(processed_image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                benign_prob, malignant_prob = probs[0].item(), probs[1].item()

            classes = ["BENIGN", "MALIGNANT"]
            pred_class = classes[torch.argmax(probs).item()]
            confidence = max(benign_prob, malignant_prob) * 100

            if pred_class == "MALIGNANT":
                st.error(f"**Predicted Class:** {pred_class}")
            else:
                st.success(f"**Predicted Class:** {pred_class}")

            st.metric(label="Model Confidence Level", value=f"{confidence:.2f}%")
            
            st.write("---")
            st.write("**Class Probability Distribution:**")
            st.progress(float(benign_prob), text=f"Benign: {benign_prob*100:.1f}%")
            st.progress(float(malignant_prob), text=f"Malignant: {malignant_prob*100:.1f}%")

# ---------------------------------------------------------
# TAB 2: MODEL BENCHMARKING
# ---------------------------------------------------------
with tab2:
    st.subheader("Test Set Performance Summary")
    
    st.dataframe(
        {
            "Model Architecture": ["DenseNet121", "MobileNetV2", "VGG16", "EfficientNetB0", "ResNet50"],
            "Accuracy": [0.721, 0.721, 0.719, 0.709, 0.686],
            "Sensitivity": [0.770, 0.750, 0.660, 0.625, 0.535],
            "Specificity": [0.679, 0.696, 0.768, 0.781, 0.814],
            "AUC-ROC": [0.798, 0.789, 0.796, 0.793, 0.772],
        },
        use_container_width=True
    )
