# Dataset preparation notes

## Classification data

Use a dataset whose licence permits your intended research use. Create a patient-level split before extracting slices or augmenting images. Do not randomly split individual slices from the same patient across train/validation/test.

Folder names become class labels. The starter script assumes a 2D image classification task. Keep the original patient/source identifiers in a separate offline manifest; do not put identifiers into this application.

## Segmentation data

The starter U-Net expects paired 2D PNG images and binary PNG masks. If starting from BraTS NIfTI volumes, first build a documented conversion pipeline that:
1. Reads volumes and segmentation labels.
2. Applies consistent orientation and spacing handling.
3. Selects slices and excludes empty slices according to a predeclared rule.
4. Converts selected tumour regions to a clearly defined binary target.
5. Exports matched image/mask PNG pairs.
6. Keeps all slices from each patient in the same split.

BraTS uses multiple tumour subregions; this starter binary U-Net does not reproduce the full BraTS multi-class segmentation task. Preserve the source labels and write down exactly how they were collapsed.
