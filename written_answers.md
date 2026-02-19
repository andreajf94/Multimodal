# Brain Metastases Tumor Segmentation — Written Answers

## Part 1: Project Planning (25 pts)

### Q1: What goal(s) do you want your model to achieve?

**Primary Goal:** Automated segmentation of brain metastases from multi-sequence MRI scans.

**Objectives:**
- Accurately delineate tumor boundaries in 3D brain MRI volumes
- Leverage complementary information from 4 MRI sequences (FLAIR, T1-pre, T1-GD, BRAVO)
- Produce clinically useful segmentations to assist radiologists in treatment planning

**Why it matters:** Brain metastases affect ~20–40% of cancer patients. Manual segmentation is slow and variable between raters. An automated pipeline would reduce radiologist workload, improve consistency, and enable faster treatment planning (surgery, stereotactic radiosurgery).

---

### Q2: Datasets — relevance and drawbacks

**Dataset:** Stanford BrainMetShare-3

| | Details |
|---|---|
| **Patients** | 156 whole-brain studies, each with ≥1 metastasis |
| **Sequences** | T1-pre, T1-GD (post-contrast), BRAVO (IR-FSPGR), FLAIR — all co-registered, skull-stripped, axial 256×256, 0.94 mm in-plane, 1.0 mm through-plane |
| **Labels** | Voxel-level segmentation masks by expert radiologists |
| **Metadata** | Primary cancer type spreadsheet (lung 99, breast 33, melanoma 7, GU 7, GI 5, other 5) |
| **Split** | Pre-defined train / test |

**Why relevant:**
1. Multiple co-registered MRI modalities → ideal testbed for multimodal fusion
2. Expert ground-truth segmentations → reliable supervision
3. Clinical NIfTI format → standard in medical imaging

**Drawbacks:**
1. Moderate sample size (156 patients) — risk of overfitting
2. Single institution (Stanford) — may not generalise across scanners or protocols
3. Severe class imbalance — tumour voxels are tiny fraction of total brain volume
4. No inter-rater variability data — hard to gauge annotation uncertainty

---

### Q3: Modality selection and rationale

| Modality | What it captures | Why selected |
|---|---|---|
| **FLAIR** | Edema & lesion extent (CSF suppressed) | Highlights peri-tumoral changes invisible on T1 |
| **T1-GD** | Active tumour (BBB breakdown enhances with Gd) | Single strongest signal for metastasis detection |
| **T1-pre** | Baseline anatomy | Needed to compute enhancement by subtracting from T1-GD |
| **BRAVO** | High-res structural detail | Fine anatomical context for precise boundary delineation |

**Modalities NOT used (and why):**
- *Clinical metadata (cancer type)* — planning to add later for multi-task learning; focusing on imaging first
- *T2-weighted / DWI* — not included in this dataset, though they would add useful contrast
- *Patient demographics* — not available; could correlate with tumour phenotype

---

### Q4: Difficulties in obtaining the data

1. **Cloud storage logistics:** Data lives in Azure Blob Storage behind a SAS token — required learning the Azure SDK, constructing proper URLs, and handling token expiry
2. **File sizes:** Each patient has 5 large NIfTI volumes (~20–50 MB total) — network bandwidth and local storage are non-trivial
3. **Specialised format:** NIfTI (.nii.gz) requires `nibabel`; standard image libraries (PIL, OpenCV) cannot read them
4. **Intensity scale variation:** Each sequence has a completely different intensity range → mandatory per-modality normalisation before any comparison or fusion
5. **Missing files:** Some patients in the test set lack segmentation masks (by design), so the code had to handle absent files gracefully

---

### Q5: Six core challenges of multimodal learning

| Challenge | Impact on this project | Planned approach |
|---|---|---|
| **1. Representation** | Each sequence has different intensity distributions and contrasts | Normalise each modality independently to [0,1]; use separate encoder branches |
| **2. Translation** | Could synthesise one sequence from another (e.g. predict T1-GD from T1-pre) | Not primary focus, but could serve as auxiliary task or handle missing modalities |
| **3. Alignment** | All sequences are already co-registered voxel-by-voxel ✅ | Verify dimensions match at load time; no registration algorithms needed |
| **4. Fusion** | Must decide how/when to combine four sequences | Start with early fusion (channel concatenation) since data is perfectly aligned; experiment with late fusion later |
| **5. Co-learning** | Can transfer knowledge across modalities | Shared encoder weights + modality-specific heads; contrastive learning between sequences |
| **6. Missing modalities** | Test set lacks seg masks; some patients may miss a sequence | Modality dropout during training; architecture that accepts variable input channels |

> The biggest advantage of this dataset is that **challenge 3 (alignment) is already solved** — all sequences are co-registered. The biggest risk is **challenge 6 (missing modalities)** in deployment scenarios.

---

## Part 3: Visualization Discussion

### What visualizations were tried and why

1. **Multi-modal slice display** — side-by-side FLAIR, T1-pre, T1-GD, BRAVO, and segmentation at the same axial slice. Shows how each sequence highlights different tissue properties.

2. **Intensity histograms** — overlaid normalised intensity distributions for each modality. Reveals that FLAIR has a bimodal distribution (bright lesions vs dark CSF), while T1 sequences are more uniform.

3. **Cross-modality correlation heatmap** — Pearson correlation between modality intensities at a single slice. T1-pre and BRAVO are highly correlated (both T1-weighted); FLAIR is less correlated with the T1 sequences, confirming it provides complementary information.

4. **t-SNE of voxel features** — each voxel described by its 4-modality intensity vector, coloured by segmentation label. Tumour voxels (label=1) cluster separately from healthy brain tissue, validating that the multi-modal features are discriminative.

---

## Part 4: Evaluation Metrics (20 pts)

### Q1: What metrics are you using and why?

| Metric | Formula | Why chosen |
|---|---|---|
| **Dice coefficient** | 2\|P∩G\| / (\|P\|+\|G\|) | Gold standard for medical segmentation; handles class imbalance well; can also be used as a loss function |
| **IoU (Jaccard)** | \|P∩G\| / \|P∪G\| | Stricter than Dice; widely used in computer vision; complements Dice |
| **Sensitivity (recall)** | TP / (TP+FN) | Clinically critical — missing a tumour is dangerous |
| **Specificity** | TN / (TN+FP) | Ensures we don't over-predict; maintains radiologist trust |
| **Precision** | TP / (TP+FP) | Measures reliability of positive predictions |

### Q2: Other metrics considered?

- **Hausdorff distance** — measures worst-case boundary error; useful but expensive to compute and sensitive to outliers
- **Average surface distance** — more robust than Hausdorff; plan to add in later HWs
- **Volumetric similarity** — simple volume comparison but ignores spatial overlap
- **Pixel accuracy** — not informative here due to severe class imbalance (~99% background)
- **AUC-ROC** — useful for probability outputs, but we're doing binary segmentation

### Q3: Pros and cons

| Metric | Pros | Cons |
|---|---|---|
| **Dice** | ✅ Standard; comparable to literature; differentiable | ❌ Unstable for very small tumours; doesn't separate error types |
| **IoU** | ✅ Intuitively interpretable; stricter than Dice | ❌ Values look lower (cosmetic); same small-object issue |
| **Sensitivity** | ✅ Critical safety metric; measures missed tumours directly | ❌ Can be trivially maximised by predicting everything as tumour |
| **Specificity** | ✅ Controls false alarms | ❌ Inflated when background dominates (as in brain MRI) |
| **Precision** | ✅ Measures prediction trustworthiness | ❌ Low for aggressive models; sensitive to threshold |

> **Together** these metrics give a balanced picture: Dice/IoU for overall quality, sensitivity for safety, precision for reliability, specificity for false-alarm control.

---

## Part 5: Instruction Tuning Prompts (15 pts)

See code file for full prompts. Key design principles:

1. **Constrained output format** — every prompt specifies "respond with EXACTLY ONE word" or "respond with X and nothing else" to prevent verbose, unparseable answers.
2. **Enumerated valid options** — when a closed label set exists (positive/negative/neutral; angry/sad/happy), the prompt lists them explicitly so the model cannot invent labels.
3. **Task-specific framing** — each prompt opens with a role ("You are a sentiment analysis assistant") to prime the model.
4. **Template placeholders** — prompts use `{review_text}` and `{text}` so they can be applied to any input programmatically.

---

## Part 6: Reflection (5 pts)

### Q1: Most interesting topic

The interplay between different MRI sequences was the most fascinating part. Each sequence is essentially a different "view" of the same brain — FLAIR suppresses fluid to reveal lesions, T1-GD lights up active tumour via contrast agent, and BRAVO provides fine structural detail. Seeing these modalities side by side made it viscerally clear why multimodal learning matters: no single sequence tells the full story.

### Q2: Unexpected challenge

I did not expect the **data engineering** aspect to be so involved. Medical imaging data lives in specialised formats (NIfTI), behind cloud storage systems (Azure Blob), with large file sizes and varying intensity ranges. The actual "machine learning" part felt straightforward by comparison — most of the effort went into downloading, parsing, normalising, and visualising the raw data. The key insight was to build a modular pipeline (download → load → normalise → extract slices → visualise) so that each step could be tested and debugged independently.

### Q3: Dataset quality assessment

**Strengths:**
- Perfect spatial alignment across modalities — eliminates the registration problem entirely
- Expert radiologist segmentations — high-quality ground truth
- 4 complementary MRI sequences — rich multimodal signal
- Standard NIfTI format and pre-defined train/test split

**Weaknesses:**
- 156 patients is moderate — will need aggressive augmentation or transfer learning to avoid overfitting
- Single institution → potential bias toward Stanford's scanner/protocol
- Severe class imbalance (tiny tumour vs large brain) → metric and loss function choices matter greatly
- No uncertainty information on annotations (would be useful to have inter-rater agreement)

Overall, this is a **high-quality dataset for learning and initial development**, but would need supplementation with multi-institutional data for any real clinical deployment.
