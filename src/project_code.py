#=======================================================================================================================
# Setup: imports, global config, and paths
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pca import pca

pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (7, 5)

# Global color palette for binary quality label (0=low, 1=high)
QUALITY_LABEL_PALETTE = {0: '#ff7f0e', 1: '#1f77b4'}

# Resolve raw and clean paths robustly relative to working directory
RAW_CANDIDATES = [
    os.path.join('../data', 'koi.csv'),
    os.path.join('../..', 'data', 'koi.csv'),
]
RAW_PATH = next((p for p in RAW_CANDIDATES if os.path.isfile(p)), RAW_CANDIDATES[0])
DATA_DIR = os.path.dirname(RAW_PATH) if os.path.basename(RAW_PATH) else os.path.join('../data')
CLEAN_PATH = os.path.join(DATA_DIR, 'koi_clean.csv')
#=======================================================================================================================

#=======================================================================================================================
# Column groups (kept consistent with the framework)
IDENTIFIER_COLS = [
    'kepid',            # Kepler Catalog ID
    'kepoi_name',       # KOI Name
    'kepler_name',      # Official Kepler Planet Name (if any)
]

TARGET_COL = 'koi_score'

LABEL_COLS = [
    'koi_disposition',     # Exoplanet Archive Disposition
    'koi_pdisposition',    # Disposition Using Kepler Data
    'koi_fpflag_nt',       # Not Transit-Like FP flag
    'koi_fpflag_ss',       # Stellar Eclipse FP flag
    'koi_fpflag_co',       # Centroid Offset FP flag
    'koi_fpflag_ec',       # Ephemeris Match FP flag
]

FEATURE_COLS = [
    # 1) Transit geometry & signal quality
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_model_snr',
    # 2) Planet properties & irradiation
    'koi_prad', 'koi_teq', 'koi_insol',
    # 3) Stellar properties
    'koi_steff', 'koi_slogg', 'koi_srad',
    # 4) Brightness (observation quality)
    'koi_kepmag',
]

# Always drop from feature matrix (metadata), per framework
DROP_ALWAYS = ['koi_tce_plnt_num', 'koi_tce_delivname']

ALL_KEEP_COLS = IDENTIFIER_COLS + [TARGET_COL] + LABEL_COLS + FEATURE_COLS
#=======================================================================================================================

#=======================================================================================================================
# Data cleaning function and build clean CSV
def clean_koi(raw_path: str, out_path: str) -> pd.DataFrame:
    """
    Load raw KOI CSV, select relevant columns, enforce basic validity of koi_score,
    and save a clean CSV for downstream use.
    """
    # Load raw CSV; ignore NASA header comments starting with '#'
    df = pd.read_csv(raw_path, comment='#', low_memory=False)
    orig_rows = len(df)
    df.columns = df.columns.str.strip()

    # Keep only the specified columns if present
    keep_cols = [c for c in ALL_KEEP_COLS if c in df.columns]
    df = df.loc[:, keep_cols].copy()

    # Drop exact duplicate rows
    before_dupes = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before_dupes - len(df)

    # Coerce numeric columns
    numeric_cols = set(FEATURE_COLS + [TARGET_COL, 'kepid'])
    numeric_cols.update([c for c in LABEL_COLS if c.startswith('koi_fpflag_')])
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove rows with missing target
    before_drop_y_na = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped_y_na = before_drop_y_na - len(df)

    # Enforce koi_score within [0, 1]
    before_range = len(df)
    mask_valid = (df[TARGET_COL] >= 0.0) & (df[TARGET_COL] <= 1.0)
    df = df.loc[mask_valid].copy()
    dropped_out_of_range = before_range - len(df)

    # Drop rows missing any of the 12 features used downstream
    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    before_drop_feat_na = len(df)
    df = df.dropna(subset=present_feats)
    dropped_feat_na = before_drop_feat_na - len(df)

    # Normalize string columns
    for c in ['kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition']:
        if c in df.columns:
            df[c] = df[c].astype('string').str.strip()

    # Save clean CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f'Loaded rows: {orig_rows}')
    print(f'Dropped exact duplicates: {dropped_dupes}')
    print(f'Dropped missing koi_score: {dropped_y_na}')
    print(f'Dropped out-of-range koi_score: {dropped_out_of_range}')
    print(f'Dropped rows missing any of the 12 features: {dropped_feat_na}')
    print(f'Kept columns ({len(keep_cols)}): {keep_cols}')
    print(f'Final shape: {df.shape}')
    print(f'Saved to: {out_path}')
    return df

# Run cleaning once to produce data/koi_clean.csv
FORCE_REBUILD = False  # set True to overwrite existing clean file if needed
if FORCE_REBUILD or (not os.path.exists(CLEAN_PATH)):
    _df_clean = clean_koi(RAW_PATH, CLEAN_PATH)
else:
    print(f'Clean file already exists at {CLEAN_PATH}. Set FORCE_REBUILD=True and re-run this cell to rebuild.')
#=======================================================================================================================

#=======================================================================================================================
# Load the cleaned dataset and perform basic EDA + binary label construction
CLEAN_CANDIDATES = [
    os.path.join('../data', 'koi_clean.csv'),
    os.path.join('../..', 'data', 'koi_clean.csv'),
]
CLEAN_PATH = next((p for p in CLEAN_CANDIDATES if os.path.isfile(p)), CLEAN_CANDIDATES[0])
df = pd.read_csv(CLEAN_PATH, low_memory=False)

print(f'Loaded clean dataset: {df.shape[0]} rows x {df.shape[1]} columns')

# Basic inspection
print('\nDataFrame info:')
df.info()
print('\nSummary statistics for koi_score and selected features:')
print(df[['koi_score'] + FEATURE_COLS].describe())
print(f"Number of exact duplicate rows: {df.duplicated().sum()}")

# koi_score distribution with thresholds 0.1 and 0.9
plt.figure(figsize=(6.5, 4.0))
sns.histplot(df['koi_score'].dropna(), bins=40, kde=True, color='#1f77b4')
plt.axvline(0.1, color='red', linestyle='--', linewidth=1.3, label='0.1 threshold')
plt.axvline(0.9, color='green', linestyle='--', linewidth=1.3, label='0.9 threshold')
plt.title('Distribution of koi_score')
plt.xlabel('koi_score')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

# Binary quality label construction based on koi_score (per FRAMEWORK)
df['quality_label'] = np.nan
df.loc[df['koi_score'] >= 0.9, 'quality_label'] = 1
df.loc[df['koi_score'] <= 0.1, 'quality_label'] = 0
df_labeled = df.dropna(subset=['quality_label']).copy()
df_labeled['quality_label'] = df_labeled['quality_label'].astype(int)

print('\nAfter applying koi_score thresholds:')
print(f'  Labeled samples: {df_labeled.shape[0]}')
print('  Label distribution (0=low quality, 1=high quality):')
print(df_labeled['quality_label'].value_counts().sort_index())

# Pre-PCA transform specification (fixed per FRAMEWORK)
LOG10_COLS = ['koi_period','koi_duration','koi_prad','koi_teq','koi_insol','koi_srad']
LOG1P_COLS = ['koi_depth','koi_model_snr']
LINEAR_COLS = ['koi_impact','koi_steff','koi_slogg','koi_kepmag']
#=======================================================================================================================

#=======================================================================================================================
# EDA boxplots of transformed numerical features (using labeled subset)
# Use labeled subset and fixed transform lists per FRAMEWORK
feat_cols = LOG10_COLS + LOG1P_COLS + LINEAR_COLS
plot_df = df_labeled.loc[:, feat_cols].copy()

# Apply the same transforms that will be used for PCA
for col in LOG10_COLS:
    plot_df[col] = np.log10(np.clip(plot_df[col], 1e-12, None))
for col in LOG1P_COLS:
    plot_df[col] = np.log10(np.clip(plot_df[col] + 1.0, 1e-12, None))

print('Using fixed transforms per FRAMEWORK:')
print('  Log10 columns:', LOG10_COLS)
print('  Log10(1+x) columns:', LOG1P_COLS)
print('  Linear columns:', LINEAR_COLS)

cols_to_plot = feat_cols
ncols = 4
nrows = math.ceil(len(cols_to_plot) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows), squeeze=False)

for i, col in enumerate(cols_to_plot):
    r, c = divmod(i, ncols)
    ax = axes[r][c]
    sns.boxplot(x=plot_df[col].dropna(), ax=ax, color='#9467bd', orient='h', whis=1.5, showfliers=False)
    tr = ' (log10)' if col in LOG10_COLS else (' (log10(1+x))' if col in LOG1P_COLS else '')
    ax.set_title(col + tr)
    ax.grid(True, axis='x', linestyle=':', alpha=0.35)

# Hide unused axes
total_axes = nrows * ncols
for j in range(len(cols_to_plot), total_axes):
    r, c = divmod(j, ncols)
    axes[r][c].set_visible(False)

plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# Build the 12-feature matrix (order as in FEATURE_COLS) with fixed pre-PCA transforms on labeled data
X_cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
X_12 = df_labeled.loc[:, X_cols].copy()

# Apply transforms in-place, keeping original column order
for c in LOG10_COLS:
    if c in X_12.columns:
        X_12[c] = np.log10(np.clip(X_12[c], 1e-12, None))
for c in LOG1P_COLS:
    if c in X_12.columns:
        X_12[c] = np.log10(np.clip(X_12[c] + 1.0, 1e-12, None))

print(f'X_12 shape (labeled, transformed): {X_12.shape}')
X_12.head(10)
#=======================================================================================================================

#=======================================================================================================================
# Correlation heatmap of X_12
# Guard: require X_12 exists
if 'X_12' not in globals():
    raise RuntimeError('X_12 not found. Run the Pre-PCA 12-feature table cell first.')

corr = X_12.corr(numeric_only=True)

# 1) Correlation heatmap (lower triangle)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(9.5, 7.5))
sns.heatmap(corr, mask=mask, cmap='vlag', vmin=-1, vmax=1, center=0,
            annot=True, fmt='.2f', linewidths=0.5, cbar_kws=dict(shrink=0.8))
plt.title('Correlation matrix of transformed features (X_12)')
plt.tight_layout()
plt.show()

# 2) Pairplot to visualize bivariate relations between attributes (sampled for efficiency)
sample_size = min(1000, len(X_12))
sample_idx = X_12.sample(n=sample_size, random_state=6220).index
pair_df = X_12.loc[sample_idx].copy()
pair_df['quality_label'] = df_labeled.loc[sample_idx, 'quality_label'].astype(int)

sns.pairplot(
    pair_df,
    vars=X_12.columns.tolist(),
    hue='quality_label',
    corner=True,
    diag_kind='hist',
    palette=QUALITY_LABEL_PALETTE,
    plot_kws={'s': 12, 'alpha': 0.5},
)
plt.suptitle('Pairwise relationships of transformed features (sample)', y=1.02)
plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# Train-test split, standardization, and PCA (using pca package)
# Guard: require X_12 (transformed 12-D features) and df_labeled (with quality_label)
if 'X_12' not in globals() or 'df_labeled' not in globals():
    raise RuntimeError('X_12 and df_labeled not found. Run the previous preprocessing cells first.')

# Prepare feature matrix and label vector
X = X_12.copy()
y = df_labeled.loc[X.index, 'quality_label'].astype(int)

# Train-test split with stratification on the binary label (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=6220, stratify=y
)

# Standardize features based on training set only
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train.astype(float))
X_test_std = scaler.transform(X_test.astype(float))

X_train_std = np.asarray(X_train_std, dtype=float)
X_test_std = np.asarray(X_test_std, dtype=float)

X_train_std_df = pd.DataFrame(X_train_std, columns=X.columns, index=X_train.index)
X_test_std_df = pd.DataFrame(X_test_std, columns=X.columns, index=X_test.index)

print(f'Training set (standardized): {X_train_std_df.shape}')
print(f'Test set (standardized):      {X_test_std_df.shape}')
print('Training label distribution:')
print(y_train.value_counts().sort_index())
print('Test label distribution:')
print(y_test.value_counts().sort_index())

# Fit PCA on standardized training set (all 12 PCs)
pca_model = pca(n_components=None, normalize=False, detect_outliers=None, random_state=6220)
pca_results = pca_model.fit_transform(X_train_std_df)

# Explained variance information from pca model results
explained_cum = np.asarray(pca_model.results['explained_var'], dtype=float)
if 'variance_ratio' in pca_model.results:
    expl_var = np.asarray(pca_model.results['variance_ratio'], dtype=float)
else:
    # Derive per-component variance from cumulative curve if variance_ratio is not available
    expl_var = np.diff(np.concatenate([[0.0], explained_cum]))

cum_var = np.cumsum(expl_var)
pc_labels = [f'PC{i+1}' for i in range(len(expl_var))]

# Approximate eigenvalues: total variance equals number of standardized features
total_var = float(X_train_std_df.shape[1])
eigvals = expl_var * total_var

ev_table = pd.DataFrame({
    'eigenvalue': eigvals,
    'explained_%': expl_var * 100.0,
    'cumulative_%': cum_var * 100.0,
}, index=pc_labels)

formatters = {
    'eigenvalue': lambda v: f'{v:.6f}',
    'explained_%': lambda v: f'{v:.2f}',
    'cumulative_%': lambda v: f'{v:.2f}',
}
print('Eigenvalues and explained variance (PCA on training set):')
print(ev_table.to_string(formatters=formatters))

# Loadings matrix from pca (orientation may differ by version)
loadings_raw = pca_model.results['loadings']

# Normalize orientation: ensure rows are PCs and columns are original features
idx_str = [str(v) for v in loadings_raw.index]
col_str = [str(v) for v in loadings_raw.columns]
has_pc_in_index = any(s.startswith('PC') for s in idx_str)
has_pc_in_cols = any(s.startswith('PC') for s in col_str)

if has_pc_in_index and not has_pc_in_cols:
    # Rows are PCs, columns are features (desired orientation)
    loadings_pc = loadings_raw.copy()
elif has_pc_in_cols and not has_pc_in_index:
    # Columns are PCs, rows are features -> transpose
    loadings_pc = loadings_raw.T.copy()
else:
    # Ambiguous; fall back to treating columns as PCs
    loadings_pc = loadings_raw.T.copy()

print('\nLoadings matrix A (first 5 PCs, rows=PCs, columns=features):')
print(loadings_pc.iloc[: min(5, loadings_pc.shape[0]), :].to_string(float_format=lambda x: f'{x:.4f}'))

# Choose number of PCs to retain (k) to reach at least ~80% cumulative variance
k_default = np.searchsorted(cum_var, 0.80) + 1
k = int(min(max(k_default, 2), X.shape[1]))
print(f'\nRetaining k={k} principal components, covering {cum_var[k-1]*100:.2f}% of variance.')

# Build PCA feature matrices for train and test using the fitted model
PC_train_df = pca_model.transform(X_train_std_df)
PC_test_df = pca_model.transform(X_test_std_df)

# Ensure consistent column order and keep only first k PCs
pc_cols = [c for c in PC_train_df.columns if isinstance(c, str) and c.startswith('PC')]
PC_train_df = PC_train_df[pc_cols]
PC_test_df = PC_test_df[pc_cols]

X_train_pca = PC_train_df.iloc[:, :k].copy()
X_test_pca = PC_test_df.iloc[:, :k].copy()

print(f'PCA feature matrices -> X_train_pca: {X_train_pca.shape}, X_test_pca: {X_test_pca.shape}')
#=======================================================================================================================

#=======================================================================================================================
# Scree and Pareto plots (explained variance from PCA)
if 'eigvals' not in globals():
    raise RuntimeError('eigvals not found. Run the PCA cell first.')

pcs = np.arange(1, len(eigvals) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree plot (eigenvalues)
axes[0].plot(pcs, eigvals, 'o-', color='#1f77b4')
axes[0].set_title('Scree Plot (Eigenvalues)')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Eigenvalue')
axes[0].grid(True, linestyle=':', alpha=0.4)

# Pareto plot (explained and cumulative variance)
axes[1].bar(pcs, expl_var * 100.0, color='#1f77b4', alpha=0.7, label='Individual %')
axes[1].plot(pcs, cum_var * 100.0, 'o-', color='#d62728', label='Cumulative %')
axes[1].set_title('Pareto Plot (Explained Variance)')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('% Variance Explained')
axes[1].set_ylim(0, 110)
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# PCA loadings heatmap and top contributors
# 1) Loadings heatmap for first k PCs (rows=features, columns=PCs)
if 'loadings_pc' not in globals():
    if 'pca_model' not in globals():
        raise RuntimeError('loadings_pc not found and pca_model missing. Run the PCA cell first.')
    loadings_raw = pca_model.results['loadings']
    idx_str = [str(v) for v in loadings_raw.index]
    col_str = [str(v) for v in loadings_raw.columns]
    has_pc_in_index = any(s.startswith('PC') for s in idx_str)
    has_pc_in_cols = any(s.startswith('PC') for s in col_str)
    if has_pc_in_index and not has_pc_in_cols:
        loadings_pc = loadings_raw.copy()
    elif has_pc_in_cols and not has_pc_in_index:
        loadings_pc = loadings_raw.T.copy()
    else:
        loadings_pc = loadings_raw.T.copy()

# loadings_pc: rows = PCs, columns = features
loadings_k = loadings_pc.iloc[:k, :].copy()        # shape (k, n_features)
loadings_plot = loadings_k.T                       # shape (n_features, k)

plt.figure(figsize=(8.2, 6.2))
ax = sns.heatmap(
    loadings_plot,
    cmap='vlag',
    center=0,
    annot=True,
    fmt='.2f',
    linewidths=0.4,
    cbar_kws=dict(shrink=0.8),
)
ax.set_title('Loadings (A): rows = features, columns = first k PCs')
ax.set_xlabel('Principal components')
ax.set_ylabel('Original features')
plt.tight_layout()
plt.show()

# Print top-contributing features for PC1..PC3 (by absolute loading)
max_pc_to_report = min(3, k)
for i in range(1, max_pc_to_report + 1):
    pc_name = f'PC{i}'
    if pc_name in loadings_k.index:
        s = loadings_k.loc[pc_name]
        top_features = s.abs().sort_values(ascending=False).head(5).index.tolist()
        print(f'Top contributors to {pc_name} (by |loading|): {", ".join(top_features)}')
#=======================================================================================================================

#=======================================================================================================================
# PCA coefficient scatter plot: A1 vs A2 for original features
# Each point is a feature; coordinates given by loadings on PC1 and PC2
if 'loadings_pc' not in globals():
    raise RuntimeError('loadings_pc not found. Run the PCA cell first to compute loadings.')

# Build A matrix for plotting (n_features x 4)
feature_names = list(X_12.columns)
A1 = loadings_pc.loc['PC1', feature_names].to_numpy()
A2 = loadings_pc.loc['PC2', feature_names].to_numpy()
if 'PC3' in loadings_pc.index:
    A3 = np.abs(loadings_pc.loc['PC3', feature_names].to_numpy())
else:
    A3 = np.sqrt(A1 ** 2 + A2 ** 2)
A4 = np.sqrt(A1 ** 2 + A2 ** 2)  # overall contribution in PC1–PC2 plane
A = np.column_stack([A1, A2, A3, A4])

plt.figure(figsize=(7.2, 5.6))
sc = plt.scatter(
    A[:, 0],
    A[:, 1],
    marker='o',
    c=A[:, 2],
    s=(np.abs(A[:, 3]) * 600 + 80),
    cmap=plt.get_cmap('Spectral'),
    alpha=0.9,
    edgecolors='black',
    linewidths=0.5,
)
plt.xlabel(r'$A_1$ (loading on PC1)')
plt.ylabel(r'$A_2$ (loading on PC2)')

for label, x, y in zip(feature_names, A[:, 0], A[:, 1]):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(-20, 20),
        textcoords='offset points',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='olive', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='white', lw=0.8),
    )

cbar = plt.colorbar(sc)
cbar.set_label('Color ~ |loading on PC3| (or overall magnitude)')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# PCA biplot: PC1 vs PC2 with eigenvectors (feature loadings)
# Combine PC score scatter and projected eigenvectors of original features
if 'pca_model' not in globals():
    raise RuntimeError('pca_model not found. Run the PCA cell first.')

# Build or reuse PC coordinates for all labeled samples
need_build_pc_all = True
if 'X_std_all' in globals() and 'PC_all_df' in globals():
    pc_all_tmp = globals()['PC_all_df']
    if isinstance(pc_all_tmp, pd.DataFrame) and 'PC1' in pc_all_tmp.columns:
        need_build_pc_all = False

if need_build_pc_all:
    X_std_all = pd.concat([X_train_std_df, X_test_std_df], axis=0).loc[X.index]
    PC_all_tmp = pca_model.transform(X_std_all)
    pc_cols_all = [c for c in PC_all_tmp.columns if isinstance(c, str) and str(c).startswith('PC')]
    PC_all_df = PC_all_tmp[pc_cols_all].iloc[:, :max(2, k)].copy()
    y_all = y.loc[X_std_all.index].astype(int).to_numpy()
    PC_all_df['quality_label'] = y_all

# Eigenvectors in the PC1–PC2 plane (from loadings)
if 'loadings_pc' not in globals():
    loadings_raw = pca_model.results['loadings']
    idx_str = [str(v) for v in loadings_raw.index]
    col_str = [str(v) for v in loadings_raw.columns]
    has_pc_in_index = any(s.startswith('PC') for s in idx_str)
    has_pc_in_cols = any(s.startswith('PC') for s in col_str)
    if has_pc_in_index and not has_pc_in_cols:
        loadings_pc = loadings_raw.copy()
    elif has_pc_in_cols and not has_pc_in_index:
        loadings_pc = loadings_raw.T.copy()
    else:
        loadings_pc = loadings_raw.T.copy()

if 'PC1' not in loadings_pc.index or 'PC2' not in loadings_pc.index:
    raise RuntimeError('PC1 or PC2 not found in loadings matrix.')

vecs_2d = loadings_pc.loc[['PC1', 'PC2'], :]  # shape (2, n_features)

# Scale eigenvectors to match PC score scatter scale
scores = PC_all_df[['PC1', 'PC2']].to_numpy()
score_radius = np.sqrt((scores ** 2).sum(axis=1))
score_scale = np.quantile(score_radius, 0.9)
vec_lengths = np.sqrt((vecs_2d.values ** 2).sum(axis=0))
max_vec_len = float(vec_lengths.max()) if vec_lengths.size > 0 else 1.0
arrow_scale = (score_scale / max_vec_len) * 0.8 if max_vec_len > 0 else 1.0

plt.figure(figsize=(7.2, 6.2))
ax = sns.scatterplot(
    data=PC_all_df,
    x='PC1',
    y='PC2',
    hue='quality_label',
    palette=QUALITY_LABEL_PALETTE,
    alpha=0.6,
    s=18,
)

# Draw eigenvector arrows from origin
origin_x, origin_y = 0.0, 0.0
for feat in X_12.columns:
    vx = float(vecs_2d.loc['PC1', feat]) * arrow_scale
    vy = float(vecs_2d.loc['PC2', feat]) * arrow_scale
    ax.arrow(
        origin_x,
        origin_y,
        vx,
        vy,
        color='black',
        alpha=0.7,
        width=0.01,
        head_width=0.18,
        head_length=0.28,
        length_includes_head=True,
    )
    ax.text(vx * 1.08, vy * 1.08, feat, fontsize=8, ha='center', va='center', color='white')

ax.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
ax.axvline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
ax.set_title('PCA biplot: PC1 vs PC2 with feature eigenvectors')
ax.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# Feature matrices for ML: original standardized 12-D space and PCA-k space
# Create DataFrames for downstream ML usage (PyCaret)
x_train_original = X_train_std_df.copy()
x_test_original = X_test_std_df.copy()
x_train_pca = X_train_pca.copy()
x_test_pca = X_test_pca.copy()

# Also provide labeled DataFrames convenient for PyCaret setup()
train_original_with_y = x_train_original.copy()
train_original_with_y['quality_label'] = y_train.values

test_original_with_y = x_test_original.copy()
test_original_with_y['quality_label'] = y_test.values

train_pca_with_y = x_train_pca.copy()
train_pca_with_y['quality_label'] = y_train.values

test_pca_with_y = x_test_pca.copy()
test_pca_with_y['quality_label'] = y_test.values

print(f'Prepared matrices -> x_train_original: {x_train_original.shape}, x_test_original: {x_test_original.shape}')
print(f'Prepared matrices -> x_train_pca: {x_train_pca.shape}, x_test_pca: {x_test_pca.shape}')

# Reset indexes and ensure clean column names for PyCaret compatibility
x_train_original = x_train_original.reset_index(drop=True)
x_test_original = x_test_original.reset_index(drop=True)
x_train_pca = x_train_pca.reset_index(drop=True)
x_test_pca = x_test_pca.reset_index(drop=True)

# Rename PCA columns to simple names (avoid any special characters)
pca_col_names = [f'PC{i+1}' for i in range(x_train_pca.shape[1])]
x_train_pca.columns = pca_col_names
x_test_pca.columns = pca_col_names

# Reset y_train and y_test indexes to match
y_train_reset = y_train.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# Rebuild labeled DataFrames with reset indexes
train_original_with_y = x_train_original.copy()
train_original_with_y['quality_label'] = y_train_reset.values

test_original_with_y = x_test_original.copy()
test_original_with_y['quality_label'] = y_test_reset.values

train_pca_with_y = x_train_pca.copy()
train_pca_with_y['quality_label'] = y_train_reset.values

test_pca_with_y = x_test_pca.copy()
test_pca_with_y['quality_label'] = y_test_reset.values

print(f'PCA column names: {list(x_train_pca.columns)}')
#=======================================================================================================================

#=======================================================================================================================
# PyCaret Classification: Original Features Setup and Model Comparison
# Use PyCaret to benchmark classifiers on the original 12-D standardized features
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, predict_model, pull, get_config

# Setup PyCaret experiment on original features
print('Setting up PyCaret classification on ORIGINAL features...')
clf_setup_orig = setup(
    data=train_original_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

# Compare all available models to get a leaderboard
print('\nComparing models on ORIGINAL features:')
best_models_orig = compare_models(n_select=3, sort='F1')
comparison_orig = pull()
print(comparison_orig)
#=======================================================================================================================

#=======================================================================================================================
# PyCaret Classification: PCA Features Setup and Model Comparison
# Use PyCaret to benchmark classifiers on PCA-transformed features
print('\nSetting up PyCaret classification on PCA features...')
clf_setup_pca = setup(
    data=train_pca_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

# Compare all available models on PCA features
print('\nComparing models on PCA features:')
best_models_pca = compare_models(n_select=3, sort='F1')
comparison_pca = pull()
print(comparison_pca)
#=======================================================================================================================

#=======================================================================================================================
# Train and Tune Models on Original Features: LR, RF, MLP
# Focus on the three models specified in FRAMEWORK: Logistic Regression, Random Forest, MLP

# Re-setup for original features (ensure we're in the right experiment context)
clf_setup_orig = setup(
    data=train_original_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('Training and tuning models on ORIGINAL features...\n')

# Logistic Regression
print('--- Logistic Regression (Original) ---')
lr_orig = create_model('lr', verbose=False)
lr_orig_tuned = tune_model(lr_orig, verbose=False, optimize='F1')
lr_orig_results = pull()
print(lr_orig_results.tail(1))

# Random Forest
print('\n--- Random Forest (Original) ---')
rf_orig = create_model('rf', verbose=False)
rf_orig_tuned = tune_model(rf_orig, verbose=False, optimize='F1')
rf_orig_results = pull()
print(rf_orig_results.tail(1))

# MLP Classifier
print('\n--- MLP Classifier (Original) ---')
mlp_orig = create_model('mlp', verbose=False)
mlp_orig_tuned = tune_model(mlp_orig, verbose=False, optimize='F1')
mlp_orig_results = pull()
print(mlp_orig_results.tail(1))

# Store tuned models for original features
tuned_models_orig = {
    'LR': lr_orig_tuned,
    'RF': rf_orig_tuned,
    'MLP': mlp_orig_tuned,
}
print('\nTuned models on original features stored.')
#=======================================================================================================================

#=======================================================================================================================
# Train and Tune Models on PCA Features: LR, RF, MLP

# Re-setup for PCA features
clf_setup_pca = setup(
    data=train_pca_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('Training and tuning models on PCA features...\n')

# Logistic Regression
print('--- Logistic Regression (PCA) ---')
lr_pca = create_model('lr', verbose=False)
lr_pca_tuned = tune_model(lr_pca, verbose=False, optimize='F1')
lr_pca_results = pull()
print(lr_pca_results.tail(1))

# Random Forest
print('\n--- Random Forest (PCA) ---')
rf_pca = create_model('rf', verbose=False)
rf_pca_tuned = tune_model(rf_pca, verbose=False, optimize='F1')
rf_pca_results = pull()
print(rf_pca_results.tail(1))

# MLP Classifier
print('\n--- MLP Classifier (PCA) ---')
mlp_pca = create_model('mlp', verbose=False)
mlp_pca_tuned = tune_model(mlp_pca, verbose=False, optimize='F1')
mlp_pca_results = pull()
print(mlp_pca_results.tail(1))

# Store tuned models for PCA features
tuned_models_pca = {
    'LR': lr_pca_tuned,
    'RF': rf_pca_tuned,
    'MLP': mlp_pca_tuned,
}
print('\nTuned models on PCA features stored.')
#=======================================================================================================================

#=======================================================================================================================
# Confusion Matrices for All Tuned Models
# Generate confusion matrices for each model on both feature sets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Setup for original features
clf_setup_orig = setup(
    data=train_original_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('Confusion Matrices - Original Features:')
fig_cm_orig, axes_cm_orig = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(tuned_models_orig.items()):
    # Get predictions on holdout set
    preds = predict_model(model, verbose=False)
    y_true = preds['quality_label']
    y_pred = preds['prediction_label']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low (0)', 'High (1)'])
    disp.plot(ax=axes_cm_orig[idx], cmap='Blues', colorbar=False)
    axes_cm_orig[idx].set_title(f'{name} (Original)')

plt.tight_layout()
plt.show()

# Setup for PCA features
clf_setup_pca = setup(
    data=train_pca_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('\nConfusion Matrices - PCA Features:')
fig_cm_pca, axes_cm_pca = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(tuned_models_pca.items()):
    # Get predictions on holdout set
    preds = predict_model(model, verbose=False)
    y_true = preds['quality_label']
    y_pred = preds['prediction_label']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low (0)', 'High (1)'])
    disp.plot(ax=axes_cm_pca[idx], cmap='Blues', colorbar=False)
    axes_cm_pca[idx].set_title(f'{name} (PCA)')

plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# ROC Curves for All Tuned Models
# Generate ROC curves with AUC for each model on both feature sets
from sklearn.metrics import roc_curve, auc

# Setup for original features
clf_setup_orig = setup(
    data=train_original_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('ROC Curves - Original Features:')
fig_roc_orig, ax_roc_orig = plt.subplots(figsize=(8, 6))

for name, model in tuned_models_orig.items():
    preds = predict_model(model, verbose=False)
    y_true = preds['quality_label']
    y_prob = preds['prediction_score']
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc_orig.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax_roc_orig.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
ax_roc_orig.set_xlabel('False Positive Rate')
ax_roc_orig.set_ylabel('True Positive Rate')
ax_roc_orig.set_title('ROC Curves - Original Features')
ax_roc_orig.legend(loc='lower right')
ax_roc_orig.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Setup for PCA features
clf_setup_pca = setup(
    data=train_pca_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

print('\nROC Curves - PCA Features:')
fig_roc_pca, ax_roc_pca = plt.subplots(figsize=(8, 6))

for name, model in tuned_models_pca.items():
    preds = predict_model(model, verbose=False)
    y_true = preds['quality_label']
    y_prob = preds['prediction_score']
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc_pca.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax_roc_pca.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
ax_roc_pca.set_xlabel('False Positive Rate')
ax_roc_pca.set_ylabel('True Positive Rate')
ax_roc_pca.set_title('ROC Curves - PCA Features')
ax_roc_pca.legend(loc='lower right')
ax_roc_pca.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# Decision Boundaries on PC1-PC2 Plane
# Visualize how each model separates the classes in the first two principal components
from sklearn.inspection import DecisionBoundaryDisplay

# Use only PC1 and PC2 for visualization
X_train_2d = x_train_pca.iloc[:, :2].values
X_test_2d = x_test_pca.iloc[:, :2].values
y_train_arr = y_train_reset.values
y_test_arr = y_test_reset.values

# Train simple versions of each model on 2D PCA data for decision boundary visualization
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models_2d = {
    'Logistic Regression': LogisticRegression(random_state=6220, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=6220, n_estimators=100),
    'MLP': MLPClassifier(random_state=6220, max_iter=1000, hidden_layer_sizes=(50, 25)),
}

fig_db, axes_db = plt.subplots(1, 3, figsize=(16, 5))

for idx, (name, model) in enumerate(models_2d.items()):
    model.fit(X_train_2d, y_train_arr)
    ax = axes_db[idx]
    
    # Create decision boundary display
    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train_2d,
        ax=ax,
        alpha=0.4,
        cmap='RdYlBu',
        response_method='predict_proba' if hasattr(model, 'predict_proba') else 'predict',
    )
    
    # Scatter plot of training data
    scatter = ax.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train_arr,
        cmap='RdYlBu',
        edgecolors='black',
        s=20,
        alpha=0.6,
    )
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Decision Boundary: {name}')
    
    # Calculate and display accuracy
    train_acc = model.score(X_train_2d, y_train_arr)
    test_acc = model.score(X_test_2d, y_test_arr)
    ax.text(0.02, 0.98, f'Train Acc: {train_acc:.3f}\nTest Acc: {test_acc:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.suptitle('Decision Boundaries on PC1-PC2 Plane', y=1.02)
plt.show()
#=======================================================================================================================

#=======================================================================================================================
# Model Comparison Summary Table
# Consolidate all metrics for Original vs PCA across LR, RF, MLP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Function to evaluate a PyCaret model on test data
def evaluate_model_metrics(model, test_data, target_col='quality_label'):
    """Evaluate model and return key metrics."""
    predictions = predict_model(model, data=test_data, verbose=False)
    y_true = predictions[target_col]
    y_pred = predictions['prediction_label']
    
    # Handle prediction_score for AUC (probability of class 1)
    if 'prediction_score' in predictions.columns:
        y_prob = predictions['prediction_score']
    else:
        y_prob = y_pred  # fallback
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
    }
    
    # Try to compute AUC if probabilities available
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['AUC'] = None
    
    return metrics

# Evaluate all models
print('Evaluating models on test sets...\n')

# Setup for original features evaluation
clf_setup_orig = setup(
    data=train_original_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

results_orig = {}
for name, model in tuned_models_orig.items():
    results_orig[name] = evaluate_model_metrics(model, test_original_with_y)

# Setup for PCA features evaluation
clf_setup_pca = setup(
    data=train_pca_with_y,
    target='quality_label',
    session_id=6220,
    verbose=False,
    html=False,
    log_experiment=False,
)

results_pca = {}
for name, model in tuned_models_pca.items():
    results_pca[name] = evaluate_model_metrics(model, test_pca_with_y)

# Build comparison DataFrame
comparison_data = []
for name in ['LR', 'RF', 'MLP']:
    # Original features
    row_orig = {'Model': name, 'Features': 'Original (12-D)'}
    row_orig.update(results_orig[name])
    comparison_data.append(row_orig)
    
    # PCA features
    row_pca = {'Model': name, 'Features': f'PCA ({k} PCs)'}
    row_pca.update(results_pca[name])
    comparison_data.append(row_pca)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df[['Model', 'Features', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC']]

print('='*80)
print('MODEL COMPARISON SUMMARY (Test Set Performance)')
print('='*80)
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A'))
print('='*80)

# Identify best model
best_idx = comparison_df['F1'].idxmax()
best_model_info = comparison_df.loc[best_idx]
print(f"\nBest Model by F1 Score: {best_model_info['Model']} on {best_model_info['Features']}")
print(f"  F1 = {best_model_info['F1']:.4f}, Accuracy = {best_model_info['Accuracy']:.4f}, AUC = {best_model_info['AUC']:.4f}")
#=======================================================================================================================

#=======================================================================================================================
# Feature Importance Analysis using Random Forest
# Visualize which features (original or PCs) are most important for classification

print('Feature Importance Analysis...\n')

# --- Feature Importance for Original Features (RF) ---
# Re-fit RF on original features to get feature importances
from sklearn.ensemble import RandomForestClassifier as RF_sklearn

rf_orig_for_importance = RF_sklearn(n_estimators=100, random_state=6220)
rf_orig_for_importance.fit(x_train_original.values, y_train_reset.values)

importances_orig = rf_orig_for_importance.feature_importances_
feature_names_orig = list(x_train_original.columns)

# Sort by importance
sorted_idx_orig = np.argsort(importances_orig)[::-1]
sorted_importances_orig = importances_orig[sorted_idx_orig]
sorted_features_orig = [feature_names_orig[i] for i in sorted_idx_orig]

fig_imp, axes_imp = plt.subplots(1, 2, figsize=(14, 5))

# Plot original features importance
axes_imp[0].barh(range(len(sorted_features_orig)), sorted_importances_orig[::-1], color='steelblue')
axes_imp[0].set_yticks(range(len(sorted_features_orig)))
axes_imp[0].set_yticklabels(sorted_features_orig[::-1])
axes_imp[0].set_xlabel('Feature Importance')
axes_imp[0].set_title('Random Forest Feature Importance\n(Original 12 Features)')
axes_imp[0].grid(True, axis='x', alpha=0.3)

# --- Feature Importance for PCA Features (RF) ---
rf_pca_for_importance = RF_sklearn(n_estimators=100, random_state=6220)
rf_pca_for_importance.fit(x_train_pca.values, y_train_reset.values)

importances_pca = rf_pca_for_importance.feature_importances_
feature_names_pca = list(x_train_pca.columns)

# Sort by importance
sorted_idx_pca = np.argsort(importances_pca)[::-1]
sorted_importances_pca = importances_pca[sorted_idx_pca]
sorted_features_pca = [feature_names_pca[i] for i in sorted_idx_pca]

# Plot PCA features importance
axes_imp[1].barh(range(len(sorted_features_pca)), sorted_importances_pca[::-1], color='darkorange')
axes_imp[1].set_yticks(range(len(sorted_features_pca)))
axes_imp[1].set_yticklabels(sorted_features_pca[::-1])
axes_imp[1].set_xlabel('Feature Importance')
axes_imp[1].set_title(f'Random Forest Feature Importance\n(PCA - {len(feature_names_pca)} Components)')
axes_imp[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# Print top features
print('Top 5 Most Important Original Features:')
for i in range(min(5, len(sorted_features_orig))):
    print(f'  {i+1}. {sorted_features_orig[i]}: {sorted_importances_orig[i]:.4f}')

print(f'\nTop 3 Most Important Principal Components:')
for i in range(min(3, len(sorted_features_pca))):
    print(f'  {i+1}. {sorted_features_pca[i]}: {sorted_importances_pca[i]:.4f}')

print('\nFeature importance analysis complete.')
#=======================================================================================================================
