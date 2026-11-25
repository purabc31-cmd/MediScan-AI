# MediScan AI — Medical Imaging Research Prototype

MediScan AI is a research prototype with a responsive web workspace and a Python inference service. It is **not clinically validated and must not be used for diagnosis or treatment decisions**.

## Current web interface

Run the Flask service and open `http://127.0.0.1:5000` to use the olive/cream responsive interface. It provides:
- Modality selection and capability status for MRI, CT and X-ray.
- Drag-and-drop file selection and local image preview.
- Zoom, rotation, brightness and contrast controls.
- MRI research inference when compatible trained weights are installed.
- Optional display of model-produced Grad-CAM/segmentation outputs.
- Local PDF research report download.

The interface intentionally marks CT and X-ray as not implemented. MRI inference currently supports one 2D image or single DICOM slice, not a complete 3D MRI study. The local upload route is disabled by default.

### Enable local upload for localhost development only

1. Copy `.env.example` to `.env`.
2. Set `LOCAL_UPLOAD_ENABLED=true`.
3. Keep the Flask development server bound to `127.0.0.1` and open it from the same computer at `http://127.0.0.1:5000` or `http://localhost:5000`.
4. Do not expose this development server to a LAN, public network, reverse proxy or the internet. The local upload/report routes require a loopback client and loopback Host header.
5. For deployment, leave `LOCAL_UPLOAD_ENABLED=false` and integrate an authenticated private backend. Do not expose the model service token to browser JavaScript.

The upload UI accepts PNG, JPG, JPEG and a single-slice DICOM file up to the configured request limit. A compatible `weights/mri_classifier.pth` is required for MRI inference; optional `weights/mri_unet.pth` enables the binary segmentation output. Scores are raw model outputs, not calibrated clinical probabilities.

See [the architecture and development roadmap](docs/ARCHITECTURE_AND_ROADMAP.md) for the full implementation status, planned phases and safety gates.

---

This is the Python AI engine for the MediScan AI research prototype. It is designed to connect to the Hatchable backend created for the project.

## Current implementation scope

- Brain MRI 2D image classification with ResNet50.
- Optional binary tumour-region segmentation with a 2D U-Net.
- Grad-CAM visualization for classification.
- PDF research report generation.
- Protected Flask endpoints for backend-to-backend calls.
- PNG, JPEG and single-slice DICOM decoding.

**Not clinically validated. Not for diagnosis or treatment decisions.** A working API is not evidence of model accuracy. Train and evaluate the models on suitable, de-identified, properly licensed datasets before interpreting predictions.

CT endpoints intentionally return `501 Not Implemented` until a separate CT-specific dataset, preprocessing pipeline, models and validation process have been implemented. Do not reuse MRI weights for CT.

## 1. Environment

Recommended: Python 3.10 or 3.11. Create a virtual environment:

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set a long random `MEDISCAN_SERVICE_TOKEN`. The same token must be entered into Hatchable's Setup page as `MEDISCAN_AI_SERVICE_TOKEN`.

## 2. Dataset layout

### Classification

Arrange a labelled dataset in ImageFolder format:

```text
data/classification/
  train/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
  val/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
  test/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
```

Use patient-level splits: images/slices from one patient must not appear in more than one split. Keep the test set untouched until final evaluation. Check dataset licensing and labels before use.

Train:

```bash
python -m training.train_classifier --data-dir data/classification --epochs 25 --batch-size 16 --image-size 224
```

Optional ImageNet initialization can be enabled with `--pretrained`; this downloads public torchvision weights during training. It does not provide medical validation.

### Segmentation

Prepare paired 2D image and binary mask PNG files:

```text
data/segmentation/
  images/
    train/
    val/
  masks/
    train/
    val/
```

Each image must have a mask with the same filename. Masks should be black for background and white for the region of interest. This starter uses binary segmentation, not BraTS multi-region labels.

Train:

```bash
python -m training.train_segmentation --data-dir data/segmentation --epochs 40 --batch-size 8 --image-size 256
```

The segmentation script saves `weights/mri_unet.pth`.

## 3. Start the Flask service

```bash
python app.py
```

The local service listens on `http://127.0.0.1:5000`.

- `GET /health`
- `POST /predict/mri`
- `POST /predict/ct` (intentionally not implemented yet)
- `POST /reports/generate`

All inference/report routes require:

```http
Authorization: Bearer YOUR_MEDISCAN_SERVICE_TOKEN
Content-Type: application/json
```

The Hatchable backend sends a short-lived signed `file_url` to `/predict/mri`. For safety, configure `ALLOWED_DOWNLOAD_HOSTS` in `.env` to include the hostname used by your Hatchable private storage signed URLs. The service rejects non-HTTPS URLs and hosts not on this allow-list. Do not remove this protection when deploying.

Example:

```env
ALLOWED_DOWNLOAD_HOSTS=your-approved-storage-host.example
```

Use the actual hostname from a signed upload URL; do not guess it.

## 4. Configure Hatchable

In the Hatchable project Setup page, set:

- `MEDISCAN_AI_SERVICE_URL` = the deployed Flask service base URL, e.g. `https://your-private-ai-service.example`
- `MEDISCAN_AI_SERVICE_TOKEN` = the same long random token used by Flask.

Keep the inference service private. Never put the service token in Lovable frontend environment variables or browser JavaScript.

## 5. Model files

After training, these files should exist:

```text
weights/
  mri_classifier.pth
  mri_classes.json
  mri_unet.pth
```

Model files are excluded from Git by default. Use Git LFS or a private model registry if you need to version/share large weights.

## 6. Evaluation before use

For classification, report a held-out patient-level test set with:
- Confusion matrix.
- Per-class precision, recall and F1.
- Sensitivity and specificity per class.
- Calibration and uncertainty analysis.

For segmentation, report:
- Dice coefficient.
- IoU.
- Sensitivity/recall.
- False-positive behaviour.

Document dataset provenance, inclusion criteria, preprocessing, patient-level splitting, model versions, limitations and external validation. Do not publish a single accuracy percentage without context.

## 7. Important limitations

- This is a starter implementation, not a validated medical device.
- The starter MRI classification labels are a simple four-class research task; dataset labels and acquisition protocols vary.
- The starter U-Net is binary 2D segmentation. It is not a complete 3D BraTS pipeline.
- Single-slice DICOM inference does not analyse a complete 3D MRI volume.
- CT is not implemented in this starter.
- A high softmax score is not equivalent to clinical confidence or diagnostic certainty.
- Never upload identifiable patient data to a development or public service.
