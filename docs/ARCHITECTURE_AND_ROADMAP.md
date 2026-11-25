# MediScan AI — Architecture and Development Roadmap

**Repository:** [purabc31-cmd/MediScan-AI](https://github.com/purabc31-cmd/MediScan-AI)  
**Status:** Research prototype scaffold; not clinically validated and not for diagnosis or treatment decisions.

## Current repository baseline

The existing `main` branch already contains a Python Flask inference service, ResNet50 MRI classifier interface, optional 2D U-Net segmentation interface, Grad-CAM visualisation, PDF research report generation, training scripts, Docker configuration, and environment-variable templates.

Implemented in code:
- Authenticated Flask endpoints: `/health`, `/predict/mri`, `/predict/ct`, and `/reports/generate`.
- PNG/JPEG and single-slice DICOM decoding.
- MRI classification and optional segmentation **only when compatible trained weights are supplied**.
- Grad-CAM and segmentation visualisations derived from model output.
- Training scripts for a four-class 2D MRI ImageFolder classifier and a binary 2D segmentation model.

Not implemented or not established:
- No trained model weights are committed; MRI inference returns a service-unavailable response until weights are trained and installed.
- CT endpoint deliberately returns 501. CT needs its own datasets, preprocessing, model and evaluation.
- There is no end-to-end MRI volume (3D) processing; the current route analyses one image/slice.
- X-ray analysis, modality/anatomy auto-identification, anatomical side/structure localisation, calibrated confidence, secure account/history UI, and the complete web frontend are not present in this repository snapshot.
- No clinical validation or regulatory clearance is established.

## Target architecture

```text
Browser UI (responsive, olive/cream palette)
  | HTTPS; consent; file type/size validation
  v
Web/API layer (Flask initially; versioned REST endpoints)
  | short-lived authorised upload reference; no public patient file URLs
  v
Private object storage + metadata store
  | modality-specific study ingestion
  v
Imaging pipeline
  |-- X-ray: 2D image validation / preprocessing / approved X-ray model
  |-- CT: DICOM series or NIfTI volume / 3D preprocessing / CT-specific model
  |-- MRI: DICOM series or NIfTI volume / 3D preprocessing / MRI-specific model
  v
Model adapter layer
  |-- model/version/dataset provenance
  |-- validated detection/classification/segmentation outputs
  |-- localisation only when the model supports it
  v
Report service
  |-- deterministic structured findings
  |-- optional LLM wording of validated outputs only
  |-- PDF export with limitations and provenance
  v
Results UI + consent-based history/deletion
```

The LLM must never independently interpret pixels or invent findings. Model scores must be labelled as raw scores unless calibration has been evaluated. Grad-CAM is an explanatory visualisation, not proof of a lesion location. Never fabricate overlays or anatomical labels.

## Phased roadmap

| Phase | Scope | Exit criteria |
|---|---|---|
| 1 | Frontend and responsive design | Home, upload/dashboard, results, about, accessible states and supplied colour palette |
| 2 | Backend upload and study ingestion | MIME/signature validation, size limits, secure temporary storage, DICOM/NIfTI handling, deletion |
| 3 | Model adapters | Separate model interfaces and documented weights, dataset licence, input assumptions and version |
| 4 | Detection and localisation | Outputs tied to validated model capabilities; no unsupported anatomy/side claims |
| 5 | Explanation service | LLM receives structured model output only; safe, source-grounded educational explanations |
| 6 | Results and PDF | Viewer controls, honest overlays, slice navigation where supported, export with provenance and limitations |
| 7 | Testing and evaluation | Unit/integration/security tests; patient-level holdout; task-specific metrics and calibration where relevant |
| 8 | GitHub integration | Clean commits, docs, dependency and secret checks, no patient data or model weights by accident |
| 9 | Deployment and domain | HTTPS, private storage, secure secrets, operational monitoring, custom domain configuration |

## Immediate build sequence

1. Preserve and document the existing inference service rather than replacing functioning code.
2. Create the responsive frontend with the requested colours: olive `#6C6F44`, warm cream `#FFFEEC`, light grey `#E6E6E6`, and dark brown `#1A120D`.
3. Add a real upload API and safe storage workflow. The current service expects a signed HTTPS file URL from an external backend; it does not itself implement a browser upload UI.
4. Add capability-aware status responses so unsupported or unconfigured modalities are visibly marked unavailable.
5. Add modality-specific integrations only after selecting datasets/models and defining evaluation plans.

## Privacy, security and medical safety gates

- Obtain explicit consent before processing/storing medical images.
- Do not put patient names, identifiers, DICOM headers, images, or reports in logs or public Git repositories.
- Validate file signatures as well as extensions; enforce limits and safe parsing.
- Keep secrets server-side; use HTTPS and short-lived access URLs with host allow-lists.
- Define retention and deletion policies before storing uploads.
- Do not describe output as diagnosis or prescribe treatment.
- Provide urgent-care/emergency guidance for concerning symptoms; analysis must never delay medical attention.
- Require qualified radiologist review and independent validation before any clinical use.

## Local development

Recommended Python: 3.10 or 3.11. See `README.md` for the existing setup and training commands. Copy `.env.example` to `.env`, set a long random service token, and configure the exact approved signed-storage hostname in `ALLOWED_DOWNLOAD_HOSTS`. Never commit `.env`.

## Current limitations / activation checklist

- [ ] Train and evaluate MRI classifier on a licensed, de-identified dataset using patient-level splits; keep test data untouched.
- [ ] Train/evaluate segmentation model against documented reference masks.
- [ ] Implement and validate full-study DICOM/NIfTI volume ingestion.
- [ ] Select independent CT and X-ray datasets/models before building those adapters.
- [ ] Decide on private object storage and user authentication before enabling history.
- [ ] Run automated tests, vulnerability/dependency checks, and deployment smoke tests.

**This roadmap documents the current code honestly. It does not claim model performance, clinical accuracy, or feature availability that has not been demonstrated.**
