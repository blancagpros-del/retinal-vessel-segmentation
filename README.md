# Retinal Blood Vessel Segmentation — Reproducing Ramos-Soto et al. (2021)

Reproduction of the unsupervised retinal vessel segmentation pipeline from Ramos-Soto et al. (2021, *Computer Methods and Programs in Biomedicine*), implemented from scratch in Python using only NumPy, SciPy, Matplotlib and scikit-image — including a custom implementation of the **Harris Hawks Optimization (HHO)** metaheuristic, which was not covered in the course syllabus.

**Authors:** Blanca González-Pros, Maria Amores Cintora
Bachelor's Degree in Physics — Universitat de Barcelona
Image Processing and Computer Vision, June 2026

## Why this project

Retinal fundus imaging is a non-invasive way to detect vascular changes linked to diabetic retinopathy, hypertension and arteriosclerosis. Manual vessel segmentation is slow and inconsistent between observers. This project reproduces an **unsupervised** pipeline — no labeled training set required, unlike deep learning approaches — that combines classical image processing with a bio-inspired optimization algorithm to pick segmentation thresholds automatically.

## Method

The pipeline splits the problem into two branches — thick and thin vessels — later merged:

- **Pre-processing:** green channel extraction, Gaussian smoothing, background fill inside the field of view (FOV)
- **Optimized top-hat transform:** morphological opening/closing with disk-shaped structuring elements
- **Homomorphic filtering:** separates illumination from reflectance in the frequency domain
- **Matched filtering** (thin branch): a rotated Gaussian kernel matched to vessel cross-sections
- **MCET-HHO** (thin branch): multilevel minimum cross-entropy thresholding, with the threshold selected via a **from-scratch implementation of Harris Hawks Optimization** (250 iterations, 30 hawks)
- **Post-processing:** branch merging (logical OR), removal of small connected components, morphological closing

## Results — DRIVE dataset, training subset (20 images)

| Metric | Ours | Original paper (test set) |
|---|---|---|
| Accuracy | 0.854 ± 0.047 | 0.967 |
| Sensitivity | 0.466 ± 0.097 | 0.758 |
| Specificity | 0.910 ± 0.054 | 0.986 |
| Dice | 0.451 ± 0.086 | — |

*Note: the original article reports metrics on DRIVE's test subset (two independent observers); we only had ground-truth for the training subset (single observer), so figures are comparable but not identical in basis.*

## Honest limitations

Two binarization steps are left unspecified in the original paper: which of MCET-HHO's four resulting thresholds separates vessel from background, and the exact thick-branch threshold. Both were resolved empirically, which accounts for most of the sensitivity gap versus the original results. Performance is also uneven across images — those with a strong illumination artifact near the FOV border produce more false positives, which the original authors also list as a known weakness of the method.

## What this project demonstrates

- Reading and faithfully reproducing a peer-reviewed scientific pipeline from a written specification alone, including filling in gaps the paper leaves undocumented
- Implementing a metaheuristic optimization algorithm (HHO) from scratch, outside of any library
- Classical image processing: Fourier-domain filtering, mathematical morphology, adaptive thresholding
- Rigorous, honest quantitative evaluation when a reproduction doesn't fully match the original — including *why* it doesn't

## Real-world relevance

Unsupervised segmentation pipelines like this one are relevant wherever labeled data for deep learning is scarce or interpretability matters, such as scalable diabetic retinopathy screening programs in public healthcare systems. Worth having this framing ready if this project comes up in a technical interview — it's the "why does this matter" layer beyond the algorithm itself.

## Repository structure

```
├── report/              # Full write-up (PDF)
├── src/                 # Python implementation
│   ├── preprocessing.py
│   ├── tophat.py
│   ├── homomorphic.py
│   ├── matched_filter.py
│   ├── hho.py           # Harris Hawks Optimization, from scratch
│   └── pipeline.py      # End-to-end segmentation pipeline
├── figures/             # Output masks and comparison figures
├── requirements.txt
└── README.md
```
*(Adjust file names above to match your actual code files.)*

## Requirements

```
numpy
scipy
matplotlib
scikit-image
tifffile
imagecodecs
```

## References

- Ramos-Soto, O. et al. (2021). "An efficient retinal blood vessel segmentation in eye fundus images by using optimized top-hat and homomorphic filtering." *Computer Methods and Programs in Biomedicine*, 201, 105949.
- Heidari, A.A. et al. (2019). "Harris hawks optimization: Algorithm and applications." *Future Generation Computer Systems*, 97, 849–872.
- Staal, J. et al. (2004). "Ridge-based vessel segmentation in color images of the retina." *IEEE Transactions on Medical Imaging*, 23(4), 501–509.
- Chaudhuri, S. et al. (1989). "Detection of blood vessels in retinal images using two-dimensional matched filters." *IEEE Transactions on Medical Imaging*, 8(3), 263–269.
