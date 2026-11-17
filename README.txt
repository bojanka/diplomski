# Automatic Segmentation and Quantitative 3D Analysis of Mandibular Asymmetry from CT Scans

Bachelor's thesis project, Faculty of Technical Sciences (FTN), University of Novi Sad, 2026.

An end-to-end computer vision pipeline that takes a raw head CT scan and produces a quantitative and visual report on mandibular asymmetry, with no manual landmarking.

> **Research use only.** This software is not a medical device and has not been clinically validated. It must not be used for diagnosis or treatment decisions.

---

# Motivation

Mandibular asymmetry is present to some degree in almost everyone, but in growing children it can progress into permanent deformity if it is not detected early. If it is caught while the mandible is still growing, less invasive options such as orthodontic monitoring and functional appliances are available. If it is detected only after growth has finished, orthognathic surgery is often the only effective option.

Current clinical practice relies on clinicians manually placing anatomical landmarks on CT/CBCT scans and computing asymmetry indices by hand. This is slow, subjective, and hard to reproduce, which makes systematic longitudinal follow-up impractical. This project automates the whole process.

# Pipeline

Raw head CT
   │
   ▼
1. Automatic mandible segmentation        (3D nnU-Net, lowres configuration)
   │
   ▼
2. 3D surface reconstruction              (Marching Cubes + Laplacian smoothing)
   │
   ▼
3. Mirror-plane reflection & distance     (point-to-surface distance, ICP-style)
   │
   ▼
4. Quantification & visualization         (asymmetry index, heat-map, numeric report)
   │
   ▼
5. Additional morphological parameters    (Hausdorff, curvature, surface area, PCA)

The only input is a raw CT scan. The output is a numeric report plus a heat-map of per-vertex deviation (in mm) on the 3D mandible surface.

# What is measured

- Mean, maximum, and standard deviation of left/right deviation (mm)
- Asymmetry index (AI) per anatomical region (condyle/ramus left and right, body left and right)
- Hausdorff distance, surface area, mean curvature
- PCA-based shape descriptors computed directly from the mesh

Parameters that need landmark detection (ramus and body length, gonial angle, per-region volume) are deliberately **not** included.

# Datasets

| | HaN-Seg | PDDCA |
|---|---|---|
| Role | Training / test | Independent external validation |
| Patients | 42 | 48 (40 with mandible mask) |
| In-plane resolution | 1024 × 1024 | 512 × 512 |
| Spacing (x, y) | 0.664 mm | 1.12 mm |
| Spacing (z) | 2.0 mm | 3.0 mm |
| Seen during training | Yes | **No** |

Both datasets come from head-and-neck radiotherapy planning and contain **adult** patients without clinically diagnosed mandibular asymmetry.

**The data is not included in this repository.** Please obtain it from the original sources and respect their licenses:

- HaN-Seg: Podobnik et al., *Med Phys* 2023, [doi:10.1002/mp.16197](https://doi.org/10.1002/mp.16197), data on [Zenodo](https://zenodo.org/records/7442914) (CC BY-ND 4.0)
- PDDCA: Raudaschl et al., *Med Phys* 2017, [doi:10.1002/mp.12197](https://doi.org/10.1002/mp.12197)

# Results

Segmentation accuracy (Dice similarity coefficient) of the final model, trained for 100 epochs:

| Evaluation set | Patients | Mean Dice | Min | Max |
|---|---|---|---|---|
| HaN-Seg held-out test set | 7 | 0.8979 | 0.8865 | 0.9044 |
| PDDCA (external, unseen) | 40 | 0.9061 | 0.7519 | 0.9497 |

- Replacing the manual mask with the automatic one changes the asymmetry measurements by less than 0.12 mm on the hardest test case.
- Asymmetry results on all 42 HaN-Seg patients are consistent with natural asymmetry reported in healthy adults (condyle/ramus more asymmetric than the mandibular body).
- Hausdorff and ICP-style distances correlate strongly with the asymmetry index (r = 0.82 and r = 0.98), so they act as independent consistency checks.

# Training length: 100 vs 1000 epochs

| Model | HaN-Seg test (mean Dice) | PDDCA (mean Dice) | PDDCA std |
|---|---|---|---|
| 100 epochs | **0.8979** | **0.9061** | 0.0370 |
| 1000 epochs | 0.8855 | 0.9038 | 0.0226 |

Validation pseudo-Dice saturated after roughly 50–100 epochs while training loss kept decreasing. On the held-out test set the 1000-epoch model was slightly worse, mostly on the most asymmetric and atypical cases. This is consistent with mild overfitting on a small training set (28 training scans). The 100-epoch model was therefore kept as the final one.

# Limitations

- Training and validation data are adult patients; **pediatric data is not yet covered**, and that is the clinically most relevant population.
- The mirror plane is approximated by the mesh center of mass along the X axis, which can introduce systematic error on strongly asymmetric mandibles.
- Analysis quality depends on segmentation quality (Dice ≈ 0.90).
- Single time point per patient: no longitudinal data, so growth-related progression cannot be modeled yet.
- Atypical anatomy (e.g. after surgery) remains the hardest case (lowest PDDCA Dice: 0.7519).

# Future work

Applying the system to pediatric CBCT data, collecting longitudinal measurements of the same child, and training a sequence model (LSTM or Transformer) to predict how asymmetry will progress during growth.

# Repository structure

> TODO: adjust to your actual layout.
.
├── 01_visualize.py              # load CT + mask, slice previews
├── 02_reconstruction.py         # Marching Cubes surface extraction
├── 03_smoothing.py              # Laplacian smoothing
├── 04_reconstruction_final.py   # final mesh + normals
├── 05_symmetry.py               # mirror plane
├── 06_asymmetry.py              # deviation computation + heat-map
├── 07_report.py                 # numeric report
├── 08_batch.py                  # run on all patients, write CSV
├── 04_predict.py                # nnU-Net inference (raw CT -> mandible mask)

# Installation

Tested with Python 3.11.

bash
pip install SimpleITK vtk numpy matplotlib open3d scikit-learn scipy
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

A CUDA-capable GPU is recommended for segmentation. Training was done on a laptop RTX 4060 (8 GB VRAM). Inference on 8 GB can run out of memory on large scans. If that happens, set `perform_everything_on_device=False` in `nnUNetPredictor`, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and process one case per process.

# Usage

1. Set the nnU-Net environment variables:
   ```bash
   export nnUNet_raw=/path/to/nnunet/raw
   export nnUNet_preprocessed=/path/to/nnunet/preprocessed
   export nnUNet_results=/path/to/nnunet/results
   ```
2. Segment the mandible:
   ```bash
   python 04_predict.py
   ```
   Use the checkpoint name **with** the extension, e.g. `checkpoint_best.pth`. A name without `.pth` can silently load a stale file.
3. Reconstruct the mesh, measure asymmetry, and generate the report and heat-map:
   ```bash
   python 08_batch.py
   ```

Trained weights are not stored in this repository because of file size (~240 MB).

# Tech stack

Python · SimpleITK · VTK · NumPy · SciPy · scikit-learn · Matplotlib · Open3D · PyTorch · nnU-Net v2

# References

1. Vasković J., Mandible, Kenhub, 2023 [На мрежи], Доступно на: https://www.kenhub.com/en/library/anatomy/the-mandible [Последњи приступ: 20. август 2026]
2. Smartt JM, Low DW, Bartlett SP. The pediatric mandible: I. A primer on growth and development. Plast Reconstr Surg., 2005 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/15988242/ [Последњи приступ: 20. август 2026]
3. Evangelista K, et al. Prevalence of mandibular asymmetry in different skeletal sagittal patterns: A systematic review, Angle Orthod., 2022 [На мрежи], Доступно на: https://pmc.ncbi.nlm.nih.gov/articles/PMC8691477/pdf/i1945-7103-92-1-118.pdf [Последњи приступ: 20. август 2026] 
4. Kovač A, Janošević P, Todorović A, Miljković M, Ranđelović K, Ilić S, Jocić S. Mandibular Asymmetries in Patients with Skeletal Classes I and II: An Analysis and Comparison, Acta Stomatologica Naissi, 2025 [На мрежи], Доступно на: https://publisher.medfak.ni.ac.rs/ASN_1/2025/JUN%202025/7.Andrea.pdf [Последњи приступ: 20. август 2026] 
5. Melnik AK. A cephalometric study of mandibular asymmetry in a longitudinally followed sample of growing children, Am J Orthod Dentofacial Orthop, 1992 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/1558065/ [Последњи приступ: 20. август 2026] 
6. Liukkonen M, Sillanmäki L, Peltomäki T. Mandibular asymmetry in healthy children, Acta Odontol Scand, 2005 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/16191911/
[Последњи приступ: 20. август 2026] 
7. Vespasiano V, Klop C, Mulder CS, Koolstra JH, Lobé NHJ, Beenen LFM, Nolte JW, Becking AG. Normal variation of mandibular asymmetry in children, Orthod Craniofac Res, 2023 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/36760175/ [Последњи приступ: 20. август 2026] 
8. Babczyńska A, Kawala B, Sarul M. Genetic factors that affect asymmetric mandibular growth – a systematic review, Symmetry, 2022 [На мрежи], Доступно на: https://www.mdpi.com/2073-
8994/14/3/490/pdf [Последњи приступ: 20. август 2026] 
9. Almășan O, Leucuța DC, Buduru S. Disc Displacement of the Temporomandibular Joint and Facial Asymmetry in Children and Adolescents: A Systematic Review and Meta-Analysis, Children (Basel), 2022 [На мрежи], Доступно на: https://www.mdpi.com/2227-9067/9/9/1297
[Последњи приступ: 20. август 2026] 
10. Verhelst PJ, Smolders A, Beznik T, et al. Layered deep learning for automatic mandibular segmentation in cone-beam computed tomography. J Dent, 2021 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/34425172/ [Последњи приступ: 20. август 2026] 
11. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015. [На мрежи], Доступно на: https://arxiv.org/pdf/1505.04597 [Последњи приступ: 20. август 2026] 
12. Dot G, Schouman T, Chang S, et al. Fully automatic segmentation of craniomaxillofacial CT scans for computer-assisted orthognathic surgery planning using the nnU-Net framework, Eur Radiol, 2022 [На мрежи], Доступно на: https://www.medrxiv.org/content/10.1101/2021.07.22.21260825v2.full.pdf
[Последњи приступ: 20. август 2026]
13. Lorensen WE, Cline HE. Marching cubes: A high resolution 3D surface construction algorithm, ACM SIGGRAPH Computer Graphics, 1987 [На мрежи], Доступно на: https://dl.acm.org/doi/10.1145/37402.37422 [Последњи приступ: 20. август 2026] 
14. Besl PJ, McKay ND. A method for registration of 3-D shapes, IEEE Trans Pattern Anal Mach Intell., 1992 [На мрежи], Доступно на: https://ieeexplore.ieee.org/document/121791 [Последњи приступ: 20. август 2026]
15. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nat Methods, 2021 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/33288961/ [Последњи приступ: 20. август 2026] 
16. Cootes TF, Taylor CJ, Cooper DH, Graham J. Active shape models — their training and application, Comput Vis Image Underst, 1995 [На мрежи], Доступно на: https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/papers/cviu95.pdf
[Последњи приступ: 20. август 2026] 
17. Podobnik G, Strojan P, Peterlin P, Ibragimov B, Vrtovec T. HaN-Seg: The head and neck organ-at-risk CT and MR segmentation dataset, Med Phys., 2023 [На мрежи], Доступно на: https://lit.fe.uni-lj.si/data/research/resources/HaN-Seg/HaN-Seg_MedPhys_2023.pdf [Последњи приступ: 20. август 2026] 
18. Raudaschl PF, Zaffino P, Sharp GC, Spadea MF, Chen A, Dawant BM, et al. Evaluation of segmentation methods on head and neck CT: auto-segmentation challenge 2015, Med Phys., 2017 [На мрежи], Доступно на: https://pubmed.ncbi.nlm.nih.gov/28273355/ [Последњи приступ: 20. август 2026] 
19. Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance, IEEE Trans Pattern Anal Mach Intell, 1993 [На мрежи], Доступно на: https://www.inf.fu-berlin.de/lehre/SS10/SP-Par/download/vd1.pdf [Последњи приступ: 20. август 2026] 
20. Podobnik G, Strojan P, Peterlin P, Ibragimov B, Vrtovec T. HaN-Seg: The head and neck organ-at-risk CT and MR segmentation dataset, Medical Psych., 2023 [На мрежи], Доступно на: https://zenodo.org/records/7442914 [Последњи приступ: 20. јул 2026.] 
21. Raudaschl, P. F., Zaffino, P., Sharp, G. C., Spadea, M. F., Chen, A., Dawant, B. M., … & Jung, F. A Public Domain Database for Computational Anatomy (PDDCA), Medical Physics, 2017 [На мрежи], Доступно на: http://www.imagenglab.com/newsite/pddca/ [Последњи приступ: 08. август 2026.] 
22. Љубица Стојшић Џуња, Љиља Мијатов Укропина, Даница Обрадовић, Oстеологија,
Треће издање, Медицински факултет Нови Сад, 2021The full thesis, with the complete literature review and all 20 references, is available [TODO: link or "on request"].

# Author
Iva Bojanic · FTN Novi Sad · iva.bojanicc@gmail.com