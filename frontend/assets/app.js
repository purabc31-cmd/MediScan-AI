const state = {
  modality: "MRI",
  file: null,
  previewUrl: null,
  rotation: 0,
  zoom: 100,
  brightness: 100,
  contrast: 100,
  lastResult: null,
  capabilities: null
};

const $ = (id) => document.getElementById(id);
const fileInput = $("scan-file");
const dropzone = $("dropzone");
const preview = $("scan-preview");
const dicomPreview = $("dicom-placeholder");
const viewerEmpty = $("viewer-empty");
const analyzeButton = $("analyze-button");

function isDicom(file) {
  return file && file.name.toLowerCase().endsWith(".dcm");
}

function updatePreviewTransform() {
  preview.style.transform = `rotate(${state.rotation}deg) scale(${state.zoom / 100})`;
  preview.style.filter = `brightness(${state.brightness}%) contrast(${state.contrast}%)`;
  $("zoom-value").textContent = `${state.zoom}%`;
  $("brightness-value").textContent = `${state.brightness}%`;
  $("contrast-value").textContent = `${state.contrast}%`;
}

function updateFileUI() {
  const hasFile = Boolean(state.file);
  $("file-preview").hidden = !hasFile;
  dropzone.hidden = hasFile;
  viewerEmpty.hidden = hasFile;
  preview.hidden = !hasFile || isDicom(state.file);
  dicomPreview.hidden = !hasFile || !isDicom(state.file);
  $("file-name").textContent = state.file?.name || "";
  $("file-size").textContent = state.file ? `${(state.file.size / 1024 / 1024).toFixed(2)} MB` : "";
  $("file-thumb").querySelector("span").textContent = isDicom(state.file) ? "DCM" : "IMG";
  $("rotate-button").disabled = !hasFile || isDicom(state.file);
  $("reset-view-button").disabled = !hasFile || isDicom(state.file);
  analyzeButton.disabled = !hasFile;
  if (state.file && !isDicom(state.file)) {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(state.file);
    preview.src = state.previewUrl;
  } else {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    preview.removeAttribute("src");
  }
  resetView();
  updateAnalyzeHint();
}

function resetView() {
  state.rotation = 0;
  state.zoom = 100;
  state.brightness = 100;
  state.contrast = 100;
  $("zoom-control").value = 100;
  $("brightness-control").value = 100;
  $("contrast-control").value = 100;
  updatePreviewTransform();
}

function setFile(file) {
  if (!file) return;
  const extensionOk = /\.(png|jpe?g|dcm)$/i.test(file.name);
  if (!extensionOk) {
    setHint("Unsupported file type. Choose PNG, JPG, JPEG or DICOM.", true);
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    setHint("This file exceeds the 15 MB development upload limit.", true);
    return;
  }
  if (!file.size) {
    setHint("The selected file is empty.", true);
    return;
  }
  state.file = file;
  state.lastResult = null;
  $("results-section").hidden = true;
  updateFileUI();
}

function clearFile() {
  state.file = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  state.lastResult = null;
  updateFileUI();
  $("results-section").hidden = true;
  setHint("Select an image to continue.");
}

function setHint(message, error = false) {
  const node = $("action-hint");
  node.textContent = message;
  node.style.color = error ? "#a24d3d" : "";
}

function updateAnalyzeHint() {
  if (!state.file) return;
  const modality = state.capabilities?.modalities?.[state.modality];
  if (!modality || modality.status === "not_implemented") {
    setHint(`${state.modality} analysis is not implemented in this repository yet.`);
  } else if (!state.capabilities?.local_upload_enabled) {
    setHint("Local upload is disabled. Enable it for localhost development or connect a secure private backend.");
  } else if (modality.status === "model_required") {
    setHint("MRI upload is wired, but trained MRI classifier weights are not installed yet.");
  } else {
    setHint("Ready for local research inference. Only one 2D image or DICOM slice is supported.");
  }
}

function selectModality(modality) {
  state.modality = modality;
  document.querySelectorAll(".modality-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.modality === modality);
  });
  updateAnalyzeHint();
}

function updateServiceStatus() {
  const status = $("service-status");
  const statusText = status.querySelector("span:last-child");
  const dot = status.querySelector(".status-dot");
  if (!state.capabilities) {
    statusText.textContent = "Service unavailable";
    dot.classList.add("muted");
    return;
  }
  statusText.textContent = state.capabilities.local_upload_enabled ? "Local development enabled" : "Service connected · local upload off";
  dot.classList.toggle("muted", !state.capabilities.local_upload_enabled);
  $("mri-status").textContent = state.capabilities.modalities.MRI.status === "available" ? "Model ready" : "Weights required";
  $("ct-status").textContent = state.capabilities.modalities.CT.status === "not_implemented" ? "Not active" : state.capabilities.modalities.CT.status;
  $("xray-status").textContent = state.capabilities.modalities["X-ray"].status === "not_implemented" ? "Not active" : state.capabilities.modalities["X-ray"].status;
  updateAnalyzeHint();
}

async function loadCapabilities() {
  try {
    const response = await fetch("/capabilities", { headers: { "Accept": "application/json" } });
    if (!response.ok) throw new Error("Capabilities endpoint unavailable");
    state.capabilities = await response.json();
    updateServiceStatus();
  } catch (error) {
    state.capabilities = null;
    updateServiceStatus();
    console.warn("Could not load MediScan AI capabilities:", error);
  }
}

function showMessage(message, error = false) {
  $("results-section").hidden = false;
  $("result-grid").hidden = true;
  $("overlay-grid").hidden = true;
  $("analysis-message").textContent = message;
  $("result-status").textContent = error ? "NOT AVAILABLE" : "INFORMATION";
  $("download-report").hidden = true;
}

function renderResult(data) {
  state.lastResult = data;
  const classification = data.classification || {};
  $("analysis-message").textContent = "The configured model returned a research classification. This output is not a diagnosis, and this interface cannot establish whether a disease or anatomical abnormality is present.";
  $("result-grid").hidden = false;
  $("result-status").textContent = "MODEL OUTPUT";
  $("predicted-class").textContent = classification.predicted_class || "Not supplied";
  const score = Number(classification.confidence);
  $("model-score").textContent = Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : "Not supplied";
  $("model-version").textContent = data.model_versions?.classifier || "Not supplied";
  const limits = Array.isArray(data.limitations) ? data.limitations : [];
  $("limitations-list").replaceChildren(...limits.map((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    return li;
  }));
  const gradcam = data.gradcam_png_base64;
  const segmentation = data.segmentation_png_base64;
  $("gradcam-figure").hidden = !gradcam;
  $("segmentation-figure").hidden = !segmentation;
  $("overlay-grid").hidden = !gradcam && !segmentation;
  if (gradcam) $("gradcam-image").src = `data:image/png;base64,${gradcam}`;
  if (segmentation) $("segmentation-image").src = `data:image/png;base64,${segmentation}`;
  $("download-report").hidden = false;
}

async function analyse() {
  if (!state.file) return;
  if (state.modality !== "MRI") {
    showMessage(`${state.modality} inference is not implemented in the current service. No analysis was performed and no findings have been generated.`, true);
    return;
  }
  if (!state.capabilities?.local_upload_enabled) {
    showMessage("Local upload is disabled for this request. For safety, browser code never contains the private service token. Enable LOCAL_UPLOAD_ENABLED only for localhost development, or connect an authenticated private backend that issues short-lived signed file URLs.", true);
    return;
  }
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Processing…";
  showMessage("Checking model availability and processing the selected image. This may take a moment.");
  try {
    const form = new FormData();
    form.append("file", state.file, state.file.name);
    const response = await fetch("/predict/mri/upload", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      showMessage(data.message || data.error || "The MRI model could not process this image.", true);
      return;
    }
    renderResult(data);
  } catch (error) {
    showMessage("The analysis service could not be reached. Check that the Flask application is running and try again.", true);
  } finally {
    analyzeButton.disabled = !state.file;
    analyzeButton.innerHTML = 'Analyse scan <span aria-hidden="true">→</span>';
  }
}

async function downloadReport() {
  if (!state.lastResult) return;
  const payload = {
    modality: "MRI",
    classification: state.lastResult.classification,
    model_versions: state.lastResult.model_versions,
    metrics: state.lastResult.metrics,
    limitations: state.lastResult.limitations
  };
  const button = $("download-report");
  button.disabled = true;
  button.textContent = "Generating PDF…";
  try {
    const response = await fetch("/reports/generate/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.message || data.error || "PDF generation is unavailable.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "MediScan-AI-research-report.pdf";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    $("analysis-message").textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Download research PDF <span aria-hidden="true">↓</span>';
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", (event) => setFile(event.target.files[0]));
dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  setFile(event.dataTransfer.files[0]);
});
$("remove-file").addEventListener("click", () => { fileInput.value = ""; clearFile(); });
document.querySelectorAll(".modality-option").forEach((button) => button.addEventListener("click", () => selectModality(button.dataset.modality)));
analyzeButton.addEventListener("click", analyse);
$("download-report").addEventListener("click", downloadReport);
$("zoom-control").addEventListener("input", (event) => { state.zoom = Number(event.target.value); updatePreviewTransform(); });
$("brightness-control").addEventListener("input", (event) => { state.brightness = Number(event.target.value); updatePreviewTransform(); });
$("contrast-control").addEventListener("input", (event) => { state.contrast = Number(event.target.value); updatePreviewTransform(); });
$("rotate-button").addEventListener("click", () => { state.rotation = (state.rotation + 90) % 360; updatePreviewTransform(); });
$("reset-view-button").addEventListener("click", resetView);
$("year").textContent = new Date().getFullYear();
updatePreviewTransform();
loadCapabilities();
