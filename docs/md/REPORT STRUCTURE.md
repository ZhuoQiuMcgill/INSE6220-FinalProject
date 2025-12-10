# REPORT STRUCTURE

Think of your LaTeX files as different “modules” in a system: each has a clear responsibility, and you route specific pieces of your RESULTs into the right place.

Below is a **section-by-section design**, including **what story to tell** and **which figures/tables to place where**.

(When I say “table” for something that’s currently just text in `RESULTS.pdf`, you can typeset it as a LaTeX table.)

---

## 0. Title / Authors / Keywords

**Goal:** make it clear this is a PCA + ML quality-classification study on Kepler KOIs.

- Title: mention *Kepler Objects of Interest*, *quality*, and *PCA + machine learning*.
- Keywords: “Kepler Objects of Interest, PCA, Dimensionality Reduction, Classification, Neural Networks, Random Forest, Quality Assessment”.

*No figures here.*

---

## 1. `sections/abstract.tex`

**Goal:** 1 paragraph (150–200 words) summarizing the entire pipeline and key numbers.

Content to include:

- Problem: binary quality classification of KOIs, derived from the continuous `koi_score`.
- Data: total 7,994 KOIs; after thresholding you keep 7,211 labeled examples with roughly balanced classes (3753 low / 3458 high).
- Methods: log/linear feature transforms + PCA (retain 4 PCs) + comparison of several classifiers (LR, RF, MLP, gradient boosting, etc.).
- Main PCA result: first 3 PCs explain ~73% of variance, 4 PCs >80%.
- Main ML result: best model is **MLP on original 12 features**, with test Accuracy ≈ 0.893 and F1 ≈ 0.890.
- One sentence of interpretation: PCA gives compact structure but loses some predictive power vs using original features; a nonlinear model captures complex relations between astrophysical features and quality.

*No figures or tables in abstract.*

---

## 2. `sections/introduction.tex`

**Goal:** Set up *why the problem matters* and *what you contribute*; don’t dive into numbers yet.

Structure:

1. **Context (1–2 paragraphs)**
    - Brief intro to *Kepler* and *Kepler Objects of Interest* (KOIs).
    - Explain that KOIs are candidate exoplanets; their `koi_score` reflects confidence/quality of the candidate vetting.
2. **Problem Statement (1 paragraph)**
    - Frame this as a **statistical quality assessment**: given a set of astrophysical and observational features, can we automatically classify KOIs into low vs high quality labels derived from `koi_score`?
3. **Challenges (short paragraph)**
    - Many correlated features (see later correlation matrix).
    - High dimensionality; some features span several orders of magnitude (motivates log transforms and PCA).
4. **Contributions (bullet list)**
    
    Something like:
    
    - Construct a clean binary quality label from `koi_score` and perform detailed exploratory analysis.
    - Apply PCA to 12 transformed features and show that a small set of PCs captures most variance while revealing underlying physical axes.
    - Benchmark a range of classifiers on original and PCA-reduced features.
    - Provide an in-depth analysis of the best-performing model (MLP) and feature importance / decision boundaries.

*No figures here; just refer forward (“as shown in Section X”).*

---

## 3. `sections/data_description.tex`

**Goal:** Document the dataset and EDA. This is where most of the **early figures** go.

### 3.1 Dataset overview

- Describe:
    - Total number of KOIs, which columns are used as **features** (list the 12) and which is the **source label** (`koi_score`).
    - Mention any removed rows (missing values, invalid scores) qualitatively.
- Explain label construction:
    - Describe in words how continuous `koi_score` is converted into binary “low vs high quality” (thresholding and discarding ambiguous middle region if that’s what you did).
    - Quote the final label counts (3753 low, 3458 high).

**Figure 1 – KOI score distribution**

- `Distribution_of_KOI_score.png`
- Caption idea: *“Histogram of koi_score with labeling thresholds indicated; low vs high quality samples are obtained by thresholding this distribution.”*

Include the **summary statistics table** (count, mean, std, min, quartiles, max) as **Table 1** right next to this figure.

### 3.2 Feature transformations

- Explain the motivation for transforming features that span orders of magnitude or are highly skewed.
- Summarize the transformation rules:
    - log₁₀: `koi_period`, `koi_duration`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_srad`
    - log₁₀(1+x): `koi_depth`, `koi_model_snr`
    - no transform: `koi_impact`, `koi_steff`, `koi_slogg`, `koi_kepmag`

**Figure 2 – Transformed feature distributions**

- `Distribution_of_Transformed_Features.png`
- Caption: *“Boxplots of transformed features after applying log/linear transformations; outliers are trimmed visually for clarity.”*

### 3.3 Correlation structure

- Explain in text that the features show moderate-to-strong correlations (e.g. between planetary radius, depth, and duration), motivating dimensionality reduction.

**Figure 3 – Correlation matrix**

- `Correlation_Matrix_of_Transformed_Features.png`
- Caption: *“Correlation matrix of the 12 transformed features, showing several groups of strongly correlated variables.”*

Optional:

**Figure 4 – Pairplot (appendix or main)**

- `Pairplot_of_Transformed_Features.png` if you keep it.
- Because it’s visually heavy, you can either put it at the end of Data Description or in an appendix; in the main text just mention that it shows approximate separability (or lack thereof) in raw feature space.

---

## 4. `sections/methodology.tex`

**Goal:** Explain **how** you model the problem in a general way (math & algorithms), not yet specific numerical results.

Suggested sub-sections:

### 4.1 Problem formulation

- Define input vector **x** (12 transformed features), label **y** ∈ {0,1}.
- Briefly state that you treat this as a supervised binary classification problem.

### 4.2 Principal Component Analysis

- Give a compact explanation of PCA:
    - Standardize features, compute covariance matrix, eigenvalues/eigenvectors, PCs as linear combinations.
    - Mention that you consider the first few PCs as new features, and you choose 4 PCs based on cumulative variance > 80% (you’ll show the numbers later in Results).
- Optionally include the **PCA loading matrix notation** (A) but keep the big numeric matrix for the Results section.

*No figures yet; keep all PCA plots for Results.*

### 4.3 Classification models

Describe each family conceptually:

- Logistic Regression (linear decision boundary, interpretable coefficients).
- Random Forest (ensemble of trees, handles nonlinear interactions / variable importance).
- MLP Classifier (feed-forward neural network; can model complex boundaries; mention basic architecture if you know it).
- Briefly note that other models (LightGBM, etc.) are benchmarked but not analysed in depth.

### 4.4 Evaluation metrics

Define Accuracy, Precision, Recall, F1, AUC, Kappa, MCC; describe why F1 and AUC are important for imbalanced or cost-sensitive classification.

*No figures here; equations only.*

---

## 5. `sections/experimental_setup.tex`

**Goal:** Specify **how** you applied those methods on this dataset.

Describe:

1. **Preprocessing pipeline**
    - Train/test split (e.g., 80/20) – mention it clearly.
    - Standardization fitted on training data only, applied to test data.
    - How PCA is fit: on training data transformed features, then applied to validation/test.
2. **Cross-validation**
    - Number of folds (e.g., 10-fold) and random seeds.
    - For each model, performance reported as mean ± std across folds (you later show these as tables).
3. **Hyperparameter tuning**
    - Briefly state that you tuned LR, RF, MLP on CV (grid search / PyCaret’s tuning); mention which hyperparameters at a high level (e.g. RF: number of trees, depth; MLP: hidden layers, activation).
4. **Model selection protocol**
    - Explain that:
        - You first compare many models on the training set via cross-validation on **original 12-D features** and **4-PC features**.
        - Then you pick **three representative models** (LR, RF, MLP) for deeper analysis and evaluation on the held-out test set.

*No figures needed; maybe a simple pipeline diagram if you want to create one.*

---

## 6. `sections/results.tex`

**Goal:** Present the numerical & visual evidence. This is where **almost all plots** from `RESULTS.pdf` live, organized into logical subsections.

### 6.1 PCA results

Start with the eigenvalues/variance table:

**Table 2 – PCA eigenvalues and explained variance**

- Use the numbers from “Explained variance by component”.
- Emphasize that PC1–PC3 explain ~73.5% and PC1–PC4 ~81.8% of total variance.

**Figure 5 – Scree + Pareto plot**

- `ParetoPlot.png` (if you have a separate ScreePlot, you can combine as subfigures).
- Caption: *“Scree and cumulative variance plot; dashed horizontal line highlights the 80% variance threshold achieved by the first 4 PCs.”*

Then move to structure in PC space:

**Figure 6 – PCA score plot (PC1 vs PC2)**

- `PCA_Score_Plot.png` (or similar).
- Describe how high vs low quality KOIs distribute; note any visible separation or overlap.

**Figure 7 – Biplot / loadings visualization**

- `PCA_Biplot.png` and/or `Feature_Loadings_Plot.png / PCA_Loading_Matrix.png`.
- Use text to explain which features contribute strongly to PC1, PC2, PC3 (you already have “Top contributors by component”).

If space is tight, use **one** loadings figure (either the heatmap or the scatter plot).

### 6.2 Model benchmarking (cross-validation)

Here you show the big PyCaret style tables.

**Table 3 – Model comparison on original 12-D features**

- Use the block with Accuracy, AUC, Recall, Precision, F1, Kappa, MCC, TT(Sec).
- In the text, briefly highlight the top performers (LightGBM, GBC, RF, MLP all around 0.88–0.89 accuracy).

**Table 4 – Model comparison on 4-PC features**

- Use the similar table for PCA features.
- Emphasize the performance drop relative to original features but still good accuracy (~0.80).

Then focus on the three chosen models:

**Table 5 – CV performance of LR, RF, MLP (original vs PCA)**

- Build from the “Training on Original Features” and “Training on PCA Features” blocks.
- For each model, show mean ± std of accuracy, AUC, recall, precision, F1, etc.
- Use the text to show:
    - MLP (original) has the highest CV accuracy and F1.
    - RF is close but slightly lower; PCA versions of all models lose some performance.

### 6.3 Test-set performance of selected models

**Table 6 – Final test performance**

- Use the “Test Set Performance” table (LR, RF, MLP with original vs PCA features).
- Highlight:
    - Best accuracy: MLP original (0.8926) and F1 (0.8898).
    - RF original also strong; PCA variants systematically lower but still competitive.

### 6.4 Confusion matrices and ROC curves

**Figure 8 – Confusion matrices**

- Panel (a): `Confusion_Matrix_Original.png` (3 models).
- Panel (b): `Confusion_Matrix_PCA.png`.
- Text: describe patterns of errors — e.g., which class is more often misclassified, how this differs across models/feature sets.

**Figure 9 – ROC curves (original features)**

- `ROC_Curve_Original.png`.

**Figure 10 – ROC curves (PCA features)**

- `ROC_Curve_PCA.png`.

In the text, comment on AUC values, even if they look strangely low compared with CV (you can acknowledge potential reasons and leave deeper discussion to the Discussion section).

### 6.5 Decision boundaries in PCA space

**Figure 11 – Decision boundaries on PC1–PC2 plane**

- `Decision_Boundaries.png` with LR, RF, MLP panels.
- Explain qualitatively:
    - LR: mostly linear boundary.
    - RF & MLP: more complex, nonlinear regions; discuss how these boundaries track clusters of high/low quality KOIs.

### 6.6 Feature importance

**Figure 12 – Random Forest feature importance (original 12-D)**

- `RF_Feature_Importance_Original.png`.
- Text: highlight top features (`koi_prad`, `koi_depth`, `koi_period`, `koi_duration`, `koi_insol`) and give intuitive astrophysical interpretations (e.g., larger planetary radius / deeper transit likely correlates with more robust detections).

**Figure 13 – Random Forest feature importance (PCA space)**

- `RF_Feature_Importance_PCA.png`.
- Use the “Top 3 most important PCs” list to explain which PCs drive predictions.

---

## 7. `sections/discussion.tex`

**Goal:** Interpret and connect the results; answer “so what?” in words.

Organize around themes:

1. **Dimensionality reduction vs prediction performance**
    - Discuss how PCA reveals underlying structure (PC1 ~ stellar environment, PC2 ~ transit depth-radius-snr, PC3 ~ size/brightness mix), but using only 4 PCs leads to lower accuracy than using all 12 features.
    - Interpret why: information loss, PCA being unsupervised, class separation not fully aligned with maximum variance directions.
2. **Model performance trade-offs**
    - Compare LR, RF, MLP:
        - LR is interpretable but underfits complex structure.
        - RF and MLP significantly improve F1 and accuracy, especially on original features.
    - Comment on training time (TT(Sec) from model comparison tables) if relevant.
3. **Error analysis**
    - Use confusion matrices to argue:
        - Whether the model tends to misclassify low quality as high (risky) or high as low (conservative).
    - Relate these patterns back to practical implications in KOI vetting (e.g., false positives waste follow-up, false negatives may miss good candidates).
4. **AUC discrepancy / limitations**
    - Reflect on the difference between CV AUC (~0.94 for RF) and test ROC curves which show AUC < 0.5 in your plots; mention possible causes (probability extraction, label encoding, or plotting bug) and acknowledge this as a limitation that should be addressed in future work.
5. **Interpretation of feature importance**
    - Connect the most important features to astrophysical reasoning:
        - larger radius / deeper transit = stronger signal,
        - longer durations and periods maybe correlate with cleaner transit shapes, etc.
    - Discuss how this matches or challenges expectations from exoplanet detection.

No new figures; this is mostly cross-referencing the figures already shown.

---

## 8. `sections/conclusion_future_work.tex`

**Goal:** Short, high-signal summary + specific next steps.

### 8.1 Conclusion

- Re-state objective in one sentence.
- Summarize key results in 3–4 bullet points:
    - Successful construction of binary KOI quality labels from `koi_score` with balanced dataset.
    - PCA shows strong dimensionality reduction potential (first 3 PCs ≈73.5% variance) and interpretable axes.
    - Nonlinear classifiers (RF, MLP, boosting) on original features achieve ~0.88–0.89 accuracy and F1, outperforming linear baselines.
    - Best model (MLP) offers a practical automatic quality screener for KOIs.

### 8.2 Future work

List 3–5 concrete directions:

- Fix and further investigate ROC/AUC behaviour and probability calibration.
- Explore alternative label definitions (different `koi_score` thresholds or multi-class labels).
- Incorporate additional features (e.g., time-series metrics) or domain knowledge.
- Try other dimensionality reduction methods (LDA, autoencoders) and gradient boosting models tuned more carefully.
- Consider cost-sensitive learning to explicitly penalize the more dangerous error type (decide which based on scientific context).

*No figures here.*

---

## 9. `sections/references.tex`

- Cite:
    - PCA / ML textbooks or standard papers.
    - Papers on Kepler KOIs and `koi_score`.
    - Libraries used (scikit-learn, PyCaret etc., in a concise way).

No figures; just standard bibliography.

---

If you follow this architecture, you basically “wire” every element of `RESULTS.pdf` into an appropriate place in the report, with **Data Description + Results** holding the visual evidence, **Methodology + Experimental Setup** explaining how you got there, and **Discussion + Conclusion** explaining what it all means.