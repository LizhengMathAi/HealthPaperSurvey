# Mini-survey over NeurIPS 2025 health+imaging papers

---

## 1. MRI / CT / 3D Radiology

### 1.1 Test-time adaptation and robustness

**D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction** ([arXiv][25])

* Problem: MRI reconstruction models trained on one distribution (scanner, hospital, protocol) often fail on new sites. Full retraining is impossible at deployment.
* Idea (from title + TTA literature): D2SA performs *two* levels of adaptation at test time:

  1. **Global distribution alignment** – adapt a small set of normalization or affine parameters to the new test distribution.
  2. **Per-slice refinement** – selectively adapt slice-wise / local features subject to a strict compute budget.
* Likely contributions:

  * A TTA scheme tailored to *reconstruction* (complex-valued inverse problem) rather than classification.
  * Sane deployment story: low test-time overhead, no access to labels or raw k-space from the new site.

**DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging**

* Problem: OOD detection is crucial before we trust models on novel pathologies or scanners.
* Idea: “Decentralized isolation networks” suggests multiple lightweight detectors operating on different feature subspaces (or anatomical regions), whose outputs are combined to estimate OOD scores.
* Likely angle:

  * Emphasis on *decentralized* / local detectors that can be aggregated securely (e.g., across sites) rather than a single central OOD model.
  * Better calibration of OOD scores in clinical regimes.

**Fast MRI for All: Bridging Access Gaps by Training without Raw Data**

* Problem: Many hospitals store only magnitude images, not raw k-space; this blocks training supervised recon models.
* High-level idea:

  * Learn MRI reconstruction *without* access to raw data, e.g., by simulating undersampling on image-space data and/or distilling from teacher recon models.
  * This is directly motivated by access/fairness: enabling low-resource hospitals to benefit from fast MRI even if they lack k-space archives.

### 1.2 3D radiology + temporal reasoning

**3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks** ([arXiv][1])

* Dataset: CT-based Med-VQA with 6 task types: anomaly detection, image observation, medical computation, existence, *static* temporal diagnosis, and *longitudinal* temporal diagnosis.
* Key features:

  * **3D** volumetric inputs, not just 2D slices.
  * Multi-temporal tasks where the model must reason about changes between scans over time.
* Findings: Current vision-language models (including medical ones) struggle especially with multi-temporal reasoning and 3D context, even when fine-tuned.

**Dual-Res Tandem Mamba-3D: Bilateral Breast Lesion Detection and Classification on Non-contrast Chest CT**

* Problem: Opportunistic breast lesion detection on non-contrast chest CT (not dedicated breast imaging).
* Technical ideas:

  * **Dual-resolution 3D backbone** (coarse+fine).
  * **Bilateral modeling** – left/right symmetry is exploited explicitly.
  * Uses *Mamba-style* sequence modeling to handle long 3D token sequences efficiently.

**AOR: Anatomical Ontology-Guided Reasoning for Medical Large Multimodal Model in Chest X-Ray Interpretation** ([NeurIPS][2])

* Idea: Inject an *anatomical ontology graph* (organs, subregions, relations) into the chest-X-ray MLLM.
* Mechanism:

  * Ontology nodes become queries/keys that steer cross-attention.
  * Encourages findings to be localized and explanations to be anatomically structured (“opacity in right lower lobe”) rather than purely pattern-based.

**BrainODE: Neural Shape Dynamics for Age- and Disease-aware Brain Trajectories**

* Problem: Brain morphology evolves over age and disease; static 3D shape models miss the *trajectory*.
* Idea: Use neural ODEs to model continuous dynamics of 3D brain shapes across time / disease progression; enabling simulation and trajectory forecasting.

**AneuG-Flow: A Large-Scale Synthetic Dataset of Diverse Intracranial Aneurysm Geometries and Hemodynamics**

* Contribution:

  * Massive *synthetic* dataset of aneurysm 3D geometries + CFD-derived hemodynamics.
  * Enables training models to predict flow features or risk markers directly from geometry, sidestepping expensive CFD.

**SAM2Flow & Learning to Zoom with Anatomical Relations**

* **SAM2Flow**: interactive optical flow for in-vivo microcirculation videos using dual memory – think of a SAM-like interactive interface that refines flow fields based on user scribbles, with short- and long-term memory states.
* **Learning to Zoom with Anatomical Relations**: dynamic zoom-in strategy for structure detection; the model learns which regions to crop at higher resolution based on anatomical context (e.g., “start at organ-level, then zoom to hilum, then vessel segment”).

---

## 2. Pathology / Whole-Slide Imaging

### 2.1 Foundation models & token efficiency

**PathVQ: Reforming Computational Pathology Foundation Model for Whole Slide Image Analysis via Vector Quantization** ([arXiv][3])

* Problem: Pathology FMs either:

  * use a *few* pooled tokens (cheap but loses spatial detail), or
  * keep *all* patch tokens (very expensive for WSIs).
* PathVQ’s solution:

  * **VQ distillation**: compress dense patch tokens into discrete indices and a small decoder.
  * Achieves ~64× token compression (e.g., 1024→16 dims) while preserving reconstruction and downstream performance.
* Impact: Makes “full-token” FMs practical for WSI classification, retrieval, and survival analysis.

### 2.2 Stain translation & normalization

**D-VST: Diffusion Transformer for Pathology-Correct Tone-Controllable Cross-Dye Virtual Staining of Whole Slide Images** ([OpenReview][4])

* Problem: Virtual staining / stain normalization often fails to preserve subtle histological details and lacks control over tone.
* Method:

  * Diffusion model + Transformer backbone (hence D-VST).
  * Explicitly conditions on target dye and tone; encourages *pathology-correct* translation, not just pretty colors.
* Outcome: Improved staining fidelity and controllable appearance, with strong pathologist-rated quality.

### 2.3 Survival analysis & sparse supervision

**Cancer Survival Analysis via Zero-shot Tumor Microenvironment Segmentation on Low-resolution Whole Slide Pathology Images** ([OpenReview][5])

* Problem:

  * Survival analysis usually needs pixel-level labels or high-res TME annotations.
  * WSIs are huge; annotations are scarce.
* Approach:

  * Use *zero-shot* TME segmentation on low-res WSIs (leveraging large segmentation FMs) to derive region-level features (immune infiltration, stroma, tumor).
  * Plug these into survival models.
* Takeaway: Shows that coarse, weakly supervised TME maps can still carry strong prognostic signal.

### 2.4 Efficient WSI exploration

**Sequential Attention-based Sampling for Histopathological Analysis (SASHA)** ([arXiv][6])

* Problem: WSIs are gigapixel; only a small fraction is diagnostically relevant.
* Method:

  * A hierarchical MIL model learns slide-level features.
  * An RL agent sequentially “zooms and samples” ~10–20% of patches to decide diagnosis.
* Result: Achieves SOTA accuracy comparable to full-slide methods while dramatically reducing compute and memory; clearly aligns with how human pathologists scan a slide.

---

## 3. OCT / Ophthalmology / Retina

**OCTDiff: Bridged Diffusion Model for Portable OCT Super-Resolution and Enhancement**

* Focus: Portable / low-cost OCT devices with noisy, low-res images.
* Idea: A “bridged” diffusion model that learns mapping from low-quality to high-quality OCT volumes, possibly using intermediate anatomical priors.

**Towards Generalizable Retina Vessel Segmentation with Deformable Graph Priors** ([zjukeliu.github.io][7])

* Problem: Vessel segmentation models overfit to specific scanners / datasets; generalization is poor.
* Approach:

  * Represent vascular trees as **deformable graphs** with priors on connectivity and topology (e.g., branching patterns).
  * Combine CNN/Transformer features with graph-based regularization to encourage anatomically plausible vessels.
* Reported effect: Significant robustness across datasets and acquisition conditions, cited also by later OOD segmentation work.([arXiv][8])

---

## 4. Dermatology / Skin Imaging

**DermaCon-IN: A Multiconcept-Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research** ([NeurIPS][9])

* Dataset:

  * Focused on Indian skin tones & disorders.
  * Annotated with multiple *concepts* per image (disease, lesion type, anatomical site, etc.).
* Importance:

  * Addresses geographic and skin-tone bias in dermatology AI (most datasets are Western, lighter skin).
  * Enables multi-task learning (diagnosis + attributes) and fairness analysis.

**Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback** ([arXiv][10])

* Problem: Diffusion-generated skin images often look realistic but are *medically wrong*.
* Method (MAGIC framework):

  * Dermatologists define criteria for correctness.
  * A multimodal LLM acts as an *AI evaluator* of generated images against those criteria.
  * Feedback (not raw labels) is translated into optimization signals for the diffusion model.
* Results:

  * Large gains in clinical realism according to dermatologists.
  * Synthetic-data augmentation boosts diagnostic accuracy by ~9–14% on challenging multi-condition tasks.

---

## 5. Endoscopy / Video Imaging

**EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis** ([arXiv][11])

* Dataset/benchmark:

  * 4 endoscopic scenarios, 12 primary tasks + 12 subtasks, 5 levels of visual prompting.
  * 6,832 VQA pairs from 21 datasets.
* Evaluation:

  * 23 MLLMs (general, medical, proprietary).
  * Proprietary models outperform open models but still lag human experts.
  * Performance is very sensitive to prompt style and task complexity.
* Message: We’re still far from “safe” endoscopy copilots; but EndoBench gives a *standardized* testbed.

---

## 6. Wearable / Motion / Physiological Imaging

**MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition** ([arXiv][12])

* Problem: HAR models on IMU data are often black boxes and don’t generalize across datasets.
* Approach:

  1. Segment multi-channel IMU streams into short windows and quantize them into discrete **motion primitives**.
  2. Feed these primitive tokens into a Transformer with a masked-token pretraining objective.
* Impact:

  * Better interpretability (you can inspect the motion codebook).
  * Improved cross-dataset generalization on six benchmarks.

**PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring** ([arXiv][13])

* Dataset:

  * In-vehicle setting with multimodal remote sensing (video, remote PPG, etc.) for physiological monitoring.
* Goals:

  * Support models that can estimate heart rate, stress, and other vitals in realistic driving scenarios.
  * Comes with benchmark tasks and baselines.

**CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing** ([OpenReview][14])

* Dataset:

  * RGB stereo, NIR, *two* thermal cameras + radar, synchronized with derived biosignals (PPG, respiration, blinks).
  * Participants perform tasks with varying cognitive loads.
* Use cases:

  * Cognitive load estimation, driver attention, human–computer interaction.
  * Highlights cross-modal fusion challenges (radar + imaging + derived waveforms).

**PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation** ([arXiv][15])

* Idea:

  * Combine multi-scale wavelet transforms with Transformer-style attention for EMG/ECG/EEG.
  * Pretrain large models on physiological signals; then build a unified multimodal branch for EMG+ECG+EEG fusion.
* Result: SOTA on multiple downstream signal tasks; points toward “foundation models” for physiological signals.

---

## 7. Medical Vision–Language & Clinical Foundation Models

### 7.1 Large medical VLM datasets & benchmarks

**MedicalNarratives: Connecting Medical Vision and Language with Localized Narratives** ([arXiv][16])

* Data:

  * 4.7M image–text pairs from instructional medical videos and articles.
  * 1M samples with dense spatial grounding (cursor traces, boxes).
  * Covers many modalities (CT, MRI, X-ray, etc.).
* Use:

  * Train GenMedCLIP and other models that jointly learn semantic and *dense* visual tasks.
  * New benchmark across multiple modalities; strong gains vs previous med-CLIP.

**3D-RAD** (already covered): also a key Med-VQA benchmark for 3D radiology. ([arXiv][1])

**RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis** ([arXiv][17])

* Dataset:

  * 600+ wrist radiographs with multi-task labels (joint space narrowing, erosions, etc.).
* Aim:

  * Standard benchmark for RA severity scoring and related tasks; fills a gap in public RA datasets.

**TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine** ([arXiv][18])

* Content:

  * > 52k QA pairs across TCM disciplines (theory, formulas, diagnostics, etc.).
  * Includes text, images, and videos.
* Contribution:

  * First unified multimodal benchmark for TCM LLMs.
  * Introduces Ladder-Score, an evaluation metric tailored to TCM terminology and semantics.

DermaCon-IN, EndoBench, and 3D-RAD are likewise *domain-specific* multi-modal datasets that stress-test MLLMs in dermatology, endoscopy, and 3D radiology, respectively.

### 7.2 Foundation models & versatile processors

**GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images** ([arXiv][19])

* Architecture:

  * Dual encoders for raw ECG time series and 12-lead ECG images.
  * Cross-modal alignment + instruction-style supervision for grounded ECG explanation.
* Contributions:

  * Defines a “Grounded ECG Understanding” benchmark linking diagnoses to waveform evidence (intervals, segments).
  * Improves predictive performance, explainability, and grounding vs prior ECG MLLMs.

**Orochi: Versatile Biomedical Image Processor** ([arXiv][20])

* Aim: “ImageJ/napari but powered by a *single* foundation model instead of many specialist models.”
* Technical ideas:

  * Pretrain on patches/volumes from >100 biomedical imaging datasets, using **Task-related Joint-embedding Pretraining (TJP)** with task-like degradations (denoising, registration, SR) rather than generic MIM.
  * Backbone is a **Multi-head Hierarchy Mamba** (state-space model) for efficiency on large volumes.
  * Three-tier fine-tuning (full / normal / light) for different compute budgets.
* Result: One model that handles diverse low-level biomedical image processing tasks, often matching or beating specialist models, including with PEFT-style light adapters.

**QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training** ([arXiv][21])

* Model: QoQ-Med-7B/32B, a generalist clinical foundation MLLM.
* Inputs: images, time series, and text reports across 9 clinical domains.
* Training:

  * Proposes **Domain-aware Relative Policy Optimization (DRPO)** – a RL objective that scales rewards differently for rare domains/harder modalities, addressing imbalanced data.
* Outcomes:

  * Large macro-F1 gains vs standard GRPO on clinical tasks.
  * Strong segmentation-grounding performance; IoU comparable to high-end proprietary models on some tasks.

**MIRA: Medical Time Series Foundation Model for Real-World Health Data** ([arXiv][22])

* Focus: *time series* (EHR signals, vitals, labs), not images, but important for multimodal clinical agents.
* Architecture:

  * Continuous-Time Rotary Positional Encoding for irregular sampling.
  * Mixture-of-Experts over frequency regimes.
  * Neural-ODE-based block for continuous latent dynamics.
* Trained on >454B time points; improves forecasting errors by ~7–10% vs strong baselines in in- and out-of-distribution settings.

### 7.3 Reasoning, retrieval, and instruction tuning

**RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis** ([arXiv][23])

* Problem: Most clinical MLLMs rely on “implicit” knowledge in weights; little explicit use of guidelines or reference documents.
* Framework:

  1. Retrieve disease-centered knowledge from guidelines and curated sources.
  2. **Guideline-enhanced contrastive loss** ties image/text features to guideline embeddings.
  3. Dual Transformer decoder uses guideline tokens as queries during fusion.
* Benefits:

  * Better diagnostic performance on multiple imaging datasets.
  * Improved interpretability: saliency aligns better with abnormal regions and guideline criteria.

**Chiron-o1: Igniting Multimodal LLMs towards Generalizable Medical Reasoning**

* Problem: Getting *good* chain-of-thought (CoT) for clinical reasoning is hard; naive CoT distillation often yields noisy or brittle reasoning.
* Approach:

  * Proposes a **Mentor-Intern Collaborative Search (MICS)** for generating and screening CoT paths.
  * Builds **MMRP**, a dataset with QA, image-text alignment, and high-quality multimodal CoT for complex clinical VQA.
* Outcome:

  * Chiron-o1 shows strong step-by-step reasoning on in- and out-of-domain medical problems, with robust VQA and explanation quality.

**MedMax: Mixed-Modal Instruction Tuning for Training Biomedical Assistants**

* Dataset:

  * 1.47M mixed-modal instruction instances: interleaved text–image content, biomedical image captioning, visual chatting, report understanding, etc.
  * Spans radiology, histopathology, and more.
* Contributions:

  * Fine-tuning mixed-modal foundation models on MedMax yields big jumps vs both Chameleon and GPT-4o on 12 biomedical VQA tasks.
  * Provides a *unified* evaluation suite for biomedical assistants.

Together with MedicalNarratives + QoQ-Med, MedMax marks a shift toward **instruction-tuned, generalist biomedical agents** that can handle multi-image, multi-step tasks.

---

## 8. Cross-cutting Trends & Open Directions

Across all these NeurIPS 2025 health-imaging papers, some clear themes emerge:

1. **From single-task models to foundation-style systems**

   * Orochi, QoQ-Med, MIRA, GEM, MedMax, and MedicalNarratives all move toward *generalist* models that:

     * cover many modalities (images, time series, text),
     * span multiple organs/domains,
     * and support multi-step reasoning / interaction.
   * This mirrors GPT-style generality, but adapted to stricter data, privacy, and evaluation constraints in medicine.

2. **Datasets and benchmarks that stress realism and multimodality**

   * 3D-RAD (3D radiology + temporal reasoning)
   * EndoBench (full endoscopy workflow)
   * DermaCon-IN (Indian skin, multi-concept)
   * TCM-Ladder (TCM multimodal QA)
   * RAM-W600, PhysDrive, CogPhys, and MedicalNarratives all push for *richer*, more clinically faithful evaluation.

3. **Generalization, OOD robustness, and fairness**

   * D2SA, DIsoN, retina deformable graph priors, and R2-Seg-style works explicitly target cross-dataset robustness and OOD segmentation.
   * DermaCon-IN tackles geographic and skin-tone biases in dermatology.
   * Foundation models (Orochi, QoQ-Med, MIRA) are evaluated in OOD regimes as a core requirement.

4. **Synthetic data & expert-in-the-loop generation**

   * AneuG-Flow (simulated aneurysm flow) and Doctor Approved/MAGIC (skin) show how *structured* synthetic data, guided by physical simulation or expert feedback, can meaningfully improve downstream models—versus naive “generate tons of images.”

5. **Trustworthiness, grounding, and interpretability**

   * RAD, GEM, Chiron-o1, AOR, and MedicalNarrives all build mechanisms to *ground* predictions in:

     * external knowledge (guidelines, retrieval),
     * explicit anatomy graphs,
     * waveform parameters,
     * or spatial traces and bounding boxes.
   * The trend is away from “black-box classifier” toward “evidence-backed, guideline-aligned assistant.”

6. **Computation-aware methods for huge inputs**

   * PathVQ and SASHA directly address the computational burden of WSIs with token compression and sequential sampling.
   * 3D-RAD and Mamba-based models (Dual-Res Mamba-3D, Orochi) use architecture choices that scale better to 3D+time.

[1]: https://arxiv.org/abs/2506.11147?utm_source=chatgpt.com "3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks"
[2]: https://neurips.cc/virtual/2025/loc/san-diego/day/12/5?utm_source=chatgpt.com "NeurIPS 2025 Friday 12/5"
[3]: https://arxiv.org/html/2503.06482v1?utm_source=chatgpt.com "Reforming Computational Pathology Foundation Model for ..."
[4]: https://openreview.net/pdf/01a05d9580a2101283275c60780b243faa2836b0.pdf?utm_source=chatgpt.com "D-VST: Diffusion Transformer for Pathology-Correct Tone- ..."
[5]: https://openreview.net/pdf/e3d333b74e83b9abb52f9706e85b92ef2869a110.pdf?utm_source=chatgpt.com "Cancer Survival Analysis via Zero-shot Tumor ..."
[6]: https://arxiv.org/abs/2507.05077?utm_source=chatgpt.com "Sequential Attention-based Sampling for Histopathological Analysis"
[7]: https://zjukeliu.github.io/?utm_source=chatgpt.com "Ke Liu's Homepage @ Zhejiang University"
[8]: https://arxiv.org/html/2511.12691v1?utm_source=chatgpt.com "R2-Seg: Training-Free OOD Medical Tumor Segmentation ..."
[9]: https://neurips.cc/virtual/2025/poster/121561?utm_source=chatgpt.com "DermaCon-IN: A Multiconcept-Annotated Dermatological ..."
[10]: https://arxiv.org/abs/2506.12323?utm_source=chatgpt.com "Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback"
[11]: https://arxiv.org/abs/2505.23601?utm_source=chatgpt.com "A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis"
[12]: https://arxiv.org/abs/2505.20744?utm_source=chatgpt.com "MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition"
[13]: https://arxiv.org/abs/2507.19172?utm_source=chatgpt.com "PhysDrive: A Multimodal Remote Physiological ..."
[14]: https://openreview.net/forum?id=VJEcCMx16R&referrer=%5Bthe+profile+of+Akane+Sano%5D%28%2Fprofile%3Fid%3D~Akane_Sano1%29&utm_source=chatgpt.com "Assessing Cognitive Load via Multimodal Remote and ..."
[15]: https://arxiv.org/abs/2506.10351?utm_source=chatgpt.com "PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation"
[16]: https://arxiv.org/abs/2501.04184?utm_source=chatgpt.com "MedicalNarratives: Connecting Medical Vision and Language with Localized Narratives"
[17]: https://arxiv.org/abs/2507.05193?utm_source=chatgpt.com "[2507.05193] RAM-W600: A Multi-Task Wrist Dataset and ..."
[18]: https://arxiv.org/abs/2505.24063?utm_source=chatgpt.com "TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine"
[19]: https://arxiv.org/abs/2503.06073?utm_source=chatgpt.com "GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images"
[20]: https://arxiv.org/abs/2509.22583?utm_source=chatgpt.com "Orochi: Versatile Biomedical Image Processor"
[21]: https://arxiv.org/abs/2506.00711?utm_source=chatgpt.com "QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training"
[22]: https://arxiv.org/abs/2506.07584?utm_source=chatgpt.com "MIRA: Medical Time Series Foundation Model for Real-World Health Data"
[23]: https://arxiv.org/abs/2509.19980?utm_source=chatgpt.com "RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis"
[25]: https://arxiv.org/pdf/2503.20815 "D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction"
