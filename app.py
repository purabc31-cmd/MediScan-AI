import base64
import hmac
import json
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from torchvision import transforms
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from models.classifier import build_classifier
from models.unet import UNet2D
from inference.preprocessing import decode_medical_image
from inference.gradcam import GradCAM
from inference.visuals import png_base64, segmentation_overlay, gradcam_overlay

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_DOWNLOAD_MB", "15")) * 1024 * 1024
DEVICE_NAME = os.getenv("DEVICE", "auto")
DEVICE = torch.device("cuda" if torch.cuda.is_available() and DEVICE_NAME != "cpu" else "cpu")
TOKEN = os.getenv("MEDISCAN_SERVICE_TOKEN", "")
WEIGHTS = Path("weights")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
LOCAL_UPLOAD_ENABLED = os.getenv("LOCAL_UPLOAD_ENABLED", "false").lower() == "true"
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
_classifier = None
_segmenter = None


def authorized():
    supplied = request.headers.get("Authorization", "")
    expected = "Bearer " + TOKEN
    return bool(TOKEN) and hmac.compare_digest(supplied, expected)


def require_token():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401
    return None


def download_scan(file_url):
    parsed = urlparse(str(file_url))
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only HTTPS scan URLs are accepted.")
    allowed = [x.strip().lower() for x in os.getenv("ALLOWED_DOWNLOAD_HOSTS", "").split(",") if x.strip()]
    host = parsed.hostname.lower()
    if not allowed or not any(host == item or host.endswith("." + item) for item in allowed):
        raise ValueError("The signed scan URL host is not in ALLOWED_DOWNLOAD_HOSTS.")
    response = requests.get(file_url, timeout=25, stream=True)
    response.raise_for_status()
    chunks = []
    total = 0
    limit = int(os.getenv("MAX_DOWNLOAD_MB", "15")) * 1024 * 1024
    for chunk in response.iter_content(1024 * 128):
        if chunk:
            total += len(chunk)
            if total > limit:
                raise ValueError("Scan exceeds the configured download size limit.")
            chunks.append(chunk)
    return b"".join(chunks)


def load_classifier():
    global _classifier, CLASS_NAMES
    weights_path = WEIGHTS / "mri_classifier.pth"
    if not weights_path.exists():
        return None
    if _classifier is None:
        classes_path = WEIGHTS / "mri_classes.json"
        if classes_path.exists():
            CLASS_NAMES = json.loads(classes_path.read_text(encoding="utf-8"))
        model = build_classifier(len(CLASS_NAMES))
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _classifier = model
    return _classifier


def load_segmenter():
    global _segmenter
    weights_path = WEIGHTS / "mri_unet.pth"
    if not weights_path.exists():
        return None
    if _segmenter is None:
        model = UNet2D()
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        _segmenter = model
    return _segmenter


def make_classifier_tensor(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def make_segmentation_tensor(image):
    resized = image.convert("L").resize((256, 256))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, None]).to(DEVICE)


def build_pdf(payload):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MediScan AI — Preliminary Research Output", styles["Title"]),
        Spacer(1, 12),
        Paragraph("This document summarizes outputs supplied by the research model service. It is not a medical diagnosis.", styles["BodyText"]),
        Spacer(1, 14),
    ]
    classification = payload.get("classification") or {}
    rows = [
        ["Field", "Value"],
        ["Research scan ID", str(payload.get("scan_id", "Not supplied"))],
        ["Modality", str(payload.get("modality", "Not supplied"))],
        ["Predicted model class", str(classification.get("predicted_class", "Not available"))],
        ["Model score", str(classification.get("confidence", "Not available"))],
        ["Model versions", json.dumps(payload.get("model_versions") or {}, ensure_ascii=False)],
        ["Evaluation metrics", json.dumps(payload.get("metrics") or "Not supplied", ensure_ascii=False)],
    ]
    table = Table(rows, colWidths=[130, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F0ED")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#23364A")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 16), Paragraph("Limitations", styles["Heading2"])])
    for item in payload.get("limitations") or ["Research prototype output; independent clinical validation has not been established."]:
        story.append(Paragraph("• " + str(item), styles["BodyText"]))
    story.extend([
        Spacer(1, 16),
        Paragraph("<b>Important:</b> AI-generated preliminary research output only. Not a medical diagnosis. Do not use for treatment or clinical decision-making. Review by a qualified radiologist and independent validation are required.", styles["BodyText"])
    ])
    doc.build(story)
    return buffer.getvalue()



def is_loopback_request():
    """Permit unauthenticated upload/report routes only from the same machine."""
    return request.remote_addr in {"127.0.0.1", "::1"}


def local_development_guard():
    if not LOCAL_UPLOAD_ENABLED or not is_loopback_request():
        return jsonify({
            "error": "local_development_route_disabled",
            "message": "Local upload is disabled. Enable LOCAL_UPLOAD_ENABLED only for localhost development; use an authenticated private backend for deployment."
        }), 403
    return None


@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/assets/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR / "assets", filename)


@app.get("/capabilities")
def capabilities():
    return jsonify({
        "service": "MediScan AI",
        "local_upload_enabled": LOCAL_UPLOAD_ENABLED and is_loopback_request(),
        "modalities": {
            "MRI": {
                "status": "available" if (WEIGHTS / "mri_classifier.pth").exists() else "model_required",
                "input_scope": "single 2D image or DICOM slice",
                "segmentation": (WEIGHTS / "mri_unet.pth").exists()
            },
            "CT": {"status": "not_implemented"},
            "X-ray": {"status": "not_implemented"}
        },
        "clinical_validation": "not_established"
    })


@app.post("/predict/mri/upload")
def predict_mri_upload():
    guard = local_development_guard()
    if guard:
        return guard
    if not (WEIGHTS / "mri_classifier.pth").exists():
        return jsonify({
            "error": "mri_classifier_weights_missing",
            "message": "MRI model weights are not installed. Train and evaluate a compatible model before running analysis."
        }), 503
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "file_required"}), 400
    filename = uploaded.filename
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".dcm"}:
        return jsonify({"error": "unsupported_file_type", "message": "Upload PNG, JPG, JPEG or a single-slice DICOM file."}), 415
    raw = uploaded.read()
    if not raw:
        return jsonify({"error": "empty_file"}), 400
    try:
        image = decode_medical_image(raw, filename)
        tensor = make_classifier_tensor(image)
        classifier = load_classifier()
        if classifier is None:
            return jsonify({"error": "mri_classifier_weights_missing"}), 503
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        classification = {
            "predicted_class": CLASS_NAMES[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(len(CLASS_NAMES))}
        }
        gradcam_png = None
        try:
            gradcam = GradCAM(classifier, classifier.layer4[-1])
            try:
                cam = gradcam(tensor, index)
            finally:
                gradcam.close()
            gradcam_png = png_base64(gradcam_overlay(image, cam))
        except Exception:
            app.logger.exception("Grad-CAM visualisation unavailable")
        segmentation_png = None
        segmenter = load_segmenter()
        if segmenter is not None:
            with torch.no_grad():
                mask = torch.sigmoid(segmenter(make_segmentation_tensor(image)))[0, 0].cpu().numpy()
            segmentation_png = png_base64(segmentation_overlay(image, mask))
        return jsonify({
            "status": "completed",
            "modality": "MRI",
            "classification": classification,
            "gradcam_png_base64": gradcam_png,
            "segmentation_png_base64": segmentation_png,
            "model_versions": {
                "classifier": "resnet50-mri-v1",
                "segmentation": "unet2d-mri-v1" if segmenter is not None else None
            },
            "metrics": None,
            "limitations": [
                "Research prototype; not clinically validated.",
                "The displayed score is an uncalibrated model score, not a clinical probability.",
                "Only one 2D image or DICOM slice is analysed; complete MRI volumes are not supported.",
                "The predicted class depends on the dataset labels used to train the local model.",
                "Grad-CAM is an exploratory model explanation and does not establish lesion location."
            ]
        })
    except (ValueError, OSError) as error:
        return jsonify({"error": "invalid_scan", "message": str(error)}), 400
    except Exception:
        app.logger.exception("Local MRI inference failed")
        return jsonify({"error": "mri_inference_failed", "message": "The image could not be processed by the configured research model."}), 500


@app.post("/reports/generate/local")
def generate_local_report():
    guard = local_development_guard()
    if guard:
        return guard
    payload = request.get_json(silent=True) or {}
    if not payload.get("classification"):
        return jsonify({"error": "classification_required"}), 400
    try:
        pdf_bytes = build_pdf(payload)
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="MediScan-AI-research-report.pdf"
        )
    except Exception:
        app.logger.exception("Local PDF report generation failed")
        return jsonify({"error": "pdf_generation_failed"}), 500


@app.get("/health")
def health():
    return jsonify({
        "service": "MediScan AI Python inference",
        "status": "online",
        "device": str(DEVICE),
        "mri_classifier_weights": (WEIGHTS / "mri_classifier.pth").exists(),
        "mri_segmentation_weights": (WEIGHTS / "mri_unet.pth").exists(),
        "ct_models": "not_implemented",
    })


@app.post("/predict/mri")
def predict_mri():
    auth_error = require_token()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    file_url = payload.get("file_url")
    if not file_url:
        return jsonify({"error": "missing_file_url"}), 400

    classifier = load_classifier()
    if classifier is None:
        return jsonify({
            "error": "mri_classifier_weights_missing",
            "message": "Train the MRI classifier and place weights/mri_classifier.pth before inference."
        }), 503

    try:
        raw = download_scan(file_url)
        image = decode_medical_image(raw, payload.get("file_name", "scan"))
        tensor = make_classifier_tensor(image)
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        classification = {
            "predicted_class": CLASS_NAMES[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(len(CLASS_NAMES))}
        }

        gradcam_png = None
        try:
            gradcam = GradCAM(classifier, classifier.layer4[-1])
            cam = gradcam(tensor, index)
            gradcam.close()
            gradcam_png = png_base64(gradcam_overlay(image, cam))
        except Exception:
            gradcam_png = None

        segmentation_png = None
        segmenter = load_segmenter()
        if segmenter is not None:
            with torch.no_grad():
                mask = torch.sigmoid(segmenter(make_segmentation_tensor(image)))[0, 0].cpu().numpy()
            segmentation_png = png_base64(segmentation_overlay(image, mask))

        return jsonify({
            "status": "completed",
            "scan_id": str(payload.get("scan_id", "")),
            "modality": "MRI",
            "classification": classification,
            "segmentation_png_base64": segmentation_png,
            "gradcam_png_base64": gradcam_png,
            "model_versions": {
                "classifier": "resnet50-mri-v1",
                "segmentation": "unet2d-mri-v1" if segmenter is not None else None
            },
            "metrics": None,
            "limitations": [
                "Research prototype; not clinically validated.",
                "A model score is not a calibrated clinical probability.",
                "This endpoint analyses one 2D image or DICOM slice, not a complete 3D MRI volume.",
                "Classification and segmentation quality depend on the training data and must be independently evaluated."
            ]
        })
    except ValueError as error:
        return jsonify({"error": "invalid_scan_source", "message": str(error)}), 400
    except Exception:
        app.logger.exception("MRI inference failed")
        return jsonify({"error": "mri_inference_failed", "message": "The image could not be processed by the current research model."}), 500


@app.post("/predict/ct")
def predict_ct():
    auth_error = require_token()
    if auth_error:
        return auth_error
    return jsonify({
        "error": "ct_model_not_implemented",
        "message": "CT inference is not implemented in this starter. Build and validate separate CT-specific preprocessing and model weights before enabling it."
    }), 501


@app.post("/reports/generate")
def generate_report():
    auth_error = require_token()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    if not payload.get("classification"):
        return jsonify({"error": "classification_required"}), 400
    try:
        pdf_bytes = build_pdf(payload)
        return jsonify({
            "status": "completed",
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii")
        })
    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({"error": "pdf_generation_failed"}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
