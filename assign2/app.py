import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Function to calculate precision and recall
def calculate_metrics(predictions, ground_truths, class_labels):
    """Computes precision and recall for each class."""
    metrics = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in class_labels}

    for pred, gt in zip(predictions, ground_truths):
        pred_classes = [p[5] for p in pred]  # Extract predicted class labels
        gt_classes = [g[5] for g in gt]  # Extract ground truth class labels

        for cls in class_labels:
            TP = sum(1 for p in pred_classes if p == cls and p in gt_classes)  # True Positives
            FP = sum(1 for p in pred_classes if p == cls and p not in gt_classes)  # False Positives
            FN = sum(1 for g in gt_classes if g == cls and g not in pred_classes)  # False Negatives

            metrics[cls]["TP"] += TP
            metrics[cls]["FP"] += FP
            metrics[cls]["FN"] += FN

    # Compute Precision and Recall
    results = []
    for cls, values in metrics.items():
        TP, FP, FN = values["TP"], values["FP"], values["FN"]
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        results.append({"Class": cls, "Precision": round(precision, 2), "Recall": round(recall, 2)})

    return pd.DataFrame(results)

# Streamlit UI
st.set_page_config(page_title="AI Object Detection", layout="centered")

st.title("🔍 AI-Powered Object Detection App")
st.write("Upload an image to detect objects and evaluate model performance.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run inference
    results = model(image_np)
    
    predictions = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            predictions.append((x1, y1, x2, y2, conf, label))  # Store label instead of index

            # Draw bounding boxes
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Objects", use_container_width=True)


    # Example Ground Truths (Should be real labels in practice)
    ground_truth = [
        (50, 50, 200, 200, 1.0, "RBC"),
        (250, 250, 400, 400, 1.0, "WBC"),
        (100, 100, 150, 150, 1.0, "Platelets"),
    ]

    # Calculate and display precision & recall
    st.write("### Model Performance Metrics")
    metrics_df = calculate_metrics([predictions], [ground_truth], ["RBC", "WBC", "Platelets"])
    st.dataframe(metrics_df)
