# =============================================================================
# Rainfall Prediction with Binary PSO Feature Selection + SVM
# Dataset: Rain in Australia (weatherAUS.csv) | Target: RainTomorrow
#
# Methodology anchored in the literature:
#   - PSO + SVM for rainfall             -> Du et al. (2017)
#   - Weighted FS fitness (acc + count)  -> Xue et al. (2012)
#   - Binary PSO w/ sigmoid transfer     -> Kennedy & Eberhart; Ji et al. (2020)
#   - Decaying inertia (anti prem. conv) -> Putri et al. (2024); Elshewey (2025)
#   - class_weight balanced for imbalance-> Rain-in-Australia ML studies
#
# Leakage controls:
#   1. train/test split BEFORE any preprocessing
#   2. PSO fitness cross-validated on TRAIN only (never touches test)
#   3. RISK_MM dropped if present (encodes next-day rainfall = the answer)
#   4. Final grid search tunes BOTH baseline and PSO model (fair comparison)
#
# Author: Mustopha | KWASU CSC FYP
# =============================================================================

# %% [markdown]
# ## 0. Config
# %%
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report)


@dataclass
class Config:
    data_path: str = "weatherAUS.csv"
    target: str = "RainTomorrow"
    test_size: float = 0.20
    random_state: int = 42

    # PSO fitness is evaluated on a stratified subsample for speed
    fitness_subsample: int = 15000
    fitness_cv_folds: int = 3
    fitness_metric: str = "roc_auc"    # imbalance-friendly, not accuracy

    # Binary PSO hyper-parameters
    n_particles: int = 20
    n_iterations: int = 30
    w_max: float = 0.9                 # inertia start  (decays linearly...)
    w_min: float = 0.4                 # inertia end    (...to here)
    c1: float = 1.5                    # cognitive
    c2: float = 1.5                    # social
    alpha: float = 0.90                # fitness weight: accuracy vs. fewer feats
    n_pso_runs: int = 5                # independent seeds -> mean +/- std

    # Final model: "fast" = Nystroem RBF + LinearSVC (scales, recommended)
    #              "exact"= real SVC(kernel='rbf')   (slow on 100k+ rows)
    model_mode: str = "fast"
    nystroem_components: int = 300

    # Small, fair grid search for the FINAL models (applied to baseline + PSO)
    grid_C: tuple = (0.1, 1.0, 10.0)
    grid_gamma: tuple = (0.01, 0.1, 1.0)   # gamma for Nystroem/RBF

CFG = Config()


# %% [markdown]
# ## 1. Load + clean (drops the RISK_MM leak)
# %%
def load_data(cfg=CFG):
    df = pd.read_csv(cfg.data_path)

    # ---- CRITICAL: drop leakage column if present ----
    # RISK_MM = next-day rainfall in mm, used to derive RainTomorrow.
    # Leaving it in inflates accuracy to ~99% (fake). Kill it.
    for leak in ["RISK_MM", "Risk_MM", "risk_mm"]:
        if leak in df.columns:
            df = df.drop(columns=[leak])
            print(f"[guard] dropped leakage column: {leak}")

    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    # Yes/No -> 1/0 for target and RainToday
    for col in [cfg.target, "RainToday"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype("float")

    # Date -> Month feature (seasonality), then drop raw date
    if "Date" in df.columns:
        df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
        df = df.drop(columns=["Date"])
    return df


# %% [markdown]
# ## 2. Preprocessing (fit on TRAIN only via ColumnTransformer)
# %%
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return pre


# %% [markdown]
# ## 3. Fast RBF-SVM approx used INSIDE the PSO loop
# %%
def make_fast_svm(gamma=None, C=1.0, cfg=CFG, seed=0):
    ny = Nystroem(kernel="rbf", n_components=cfg.nystroem_components,
                  gamma=gamma, random_state=seed)
    return Pipeline([
        ("rbf", ny),
        ("svm", LinearSVC(class_weight="balanced", C=C, dual="auto",
                          max_iter=5000)),
    ])


# %% [markdown]
# ## 4. Binary PSO for feature selection
#   fitness (MINIMISE) = alpha*(1 - CV_AUC) + (1-alpha)*(n_selected/n_total)
# %%
def _fitness(mask, Xf, y, cfg=CFG, seed=0):
    if mask.sum() == 0:
        return 1.0
    cols = np.where(mask == 1)[0]
    model = make_fast_svm(cfg=cfg, seed=seed)
    skf = StratifiedKFold(n_splits=cfg.fitness_cv_folds, shuffle=True,
                          random_state=seed)
    auc = cross_val_score(model, Xf[:, cols], y, cv=skf,
                          scoring=cfg.fitness_metric, n_jobs=-1).mean()
    return cfg.alpha * (1 - auc) + (1 - cfg.alpha) * (mask.sum() / mask.size)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))


def binary_pso(Xf, y, cfg=CFG, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = Xf.shape[1]
    X = rng.integers(0, 2, size=(cfg.n_particles, n))
    V = rng.uniform(-1, 1, size=(cfg.n_particles, n))

    pbest = X.copy()
    pbest_fit = np.array([_fitness(X[i], Xf, y, cfg, seed)
                          for i in range(cfg.n_particles)])
    g = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    history = [gbest_fit]

    for it in range(cfg.n_iterations):
        # linearly decaying inertia weight (curbs premature convergence)
        w = cfg.w_max - (cfg.w_max - cfg.w_min) * it / max(1, cfg.n_iterations - 1)
        r1 = rng.random((cfg.n_particles, n))
        r2 = rng.random((cfg.n_particles, n))
        V = w * V + cfg.c1 * r1 * (pbest - X) + cfg.c2 * r2 * (gbest - X)
        X = (rng.random((cfg.n_particles, n)) < _sigmoid(V)).astype(int)

        for i in range(cfg.n_particles):
            fit = _fitness(X[i], Xf, y, cfg, seed)
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, X[i].copy()
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit:
            gbest_fit, gbest = pbest_fit[g], pbest[g].copy()
        history.append(gbest_fit)
        if verbose:
            print(f"  seed {seed} | iter {it+1:02d}/{cfg.n_iterations} "
                  f"| w={w:.2f} | fitness {gbest_fit:.4f} "
                  f"| feats {int(gbest.sum())}")
    return gbest, gbest_fit, history


# %% [markdown]
# ## 5. Final tuned model + evaluation (fair grid search on BOTH)
# %%
def tune_and_eval(Xtr, ytr, Xte, yte, cfg=CFG, label="", seed=0):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if cfg.model_mode == "exact":
        base = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        grid = {"C": cfg.grid_C, "gamma": cfg.grid_gamma}
        gs = GridSearchCV(base, grid, scoring="roc_auc", cv=skf, n_jobs=-1)
    else:
        pipe = make_fast_svm(cfg=cfg, seed=seed)
        grid = {"rbf__gamma": cfg.grid_gamma, "svm__C": cfg.grid_C}
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=skf, n_jobs=-1)

    t0 = time.time()
    gs.fit(Xtr, ytr)
    train_time = time.time() - t0
    model = gs.best_estimator_

    pred = model.predict(Xte)
    try:
        auc = roc_auc_score(yte, model.decision_function(Xte))
    except Exception:
        auc = roc_auc_score(yte, pred)
    return {
        "model": label,
        "n_features": Xtr.shape[1],
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "auc": auc,
        "best_params": gs.best_params_,
        "train_time_s": round(train_time, 2),
        "_pred": pred,
    }


# %% [markdown]
# ## 6. Orchestration
# %%
def run(cfg=CFG):
    df = load_data(cfg)
    y = df[cfg.target].astype(int).values
    X = df.drop(columns=[cfg.target])

    # 1) SPLIT FIRST (stratified: imbalance-safe)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state)

    # 2) Preprocessing fit on TRAIN only
    pre = build_preprocessor(X_tr)
    Xtr = pre.fit_transform(X_tr)
    Xte = pre.transform(X_te)
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
    Xte = Xte.toarray() if hasattr(Xte, "toarray") else Xte
    feat_names = list(pre.get_feature_names_out())
    n_total = Xtr.shape[1]
    print(f"\nFeatures after encoding: {n_total}")
    print(f"Train class balance: {np.bincount(y_tr)} "
          f"({y_tr.mean()*100:.1f}% rain)")

    # 3) Stratified subsample for the PSO fitness loop
    if Xtr.shape[0] > cfg.fitness_subsample:
        idx, _ = train_test_split(np.arange(Xtr.shape[0]),
                                  train_size=cfg.fitness_subsample,
                                  stratify=y_tr, random_state=cfg.random_state)
    else:
        idx = np.arange(Xtr.shape[0])
    Xf, yf = Xtr[idx], y_tr[idx]

    # 4) BASELINE (all features, tuned)
    print("\n=== Baseline: all features (tuned) ===")
    base = tune_and_eval(Xtr, y_tr, Xte, y_te, cfg, "Baseline (all features)")
    print({k: v for k, v in base.items() if not k.startswith("_")})

    # 5) PSO feature selection over multiple seeds
    print("\n=== PSO feature selection ===")
    masks, fits, histories = [], [], []
    for r in range(cfg.n_pso_runs):
        seed = cfg.random_state + r
        gbest, gfit, hist = binary_pso(Xf, yf, cfg, seed=seed, verbose=True)
        masks.append(gbest); fits.append(gfit); histories.append(hist)
        print(f"  -> run {r+1}: {int(gbest.sum())} feats | fitness {gfit:.4f}")

    counts = [int(m.sum()) for m in masks]
    print(f"\nFeature count over {cfg.n_pso_runs} runs: "
          f"{np.mean(counts):.1f} +/- {np.std(counts):.1f}")

    best = int(np.argmin(fits))
    mask = masks[best]
    cols = np.where(mask == 1)[0]
    selected = [feat_names[i] for i in cols]
    print(f"Best run: {int(mask.sum())}/{n_total} features "
          f"({(1 - mask.sum()/n_total)*100:.1f}% reduction)")

    # 6) FINAL: PSO-selected features (tuned)
    print("\n=== PSO + SVM: selected features (tuned) ===")
    sel = tune_and_eval(Xtr[:, cols], y_tr, Xte[:, cols], y_te, cfg, "PSO + SVM")
    print({k: v for k, v in sel.items() if not k.startswith("_")})

    # Report
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (PSO + SVM)")
    print(confusion_matrix(y_te, sel["_pred"]))
    print("\nCLASSIFICATION REPORT (PSO + SVM)")
    print(classification_report(y_te, sel["_pred"],
                                target_names=["No Rain", "Rain"]))

    results = pd.DataFrame([
        {k: v for k, v in base.items() if not k.startswith("_")},
        {k: v for k, v in sel.items() if not k.startswith("_")},
    ])
    return {
        "results": results, "selected_features": selected,
        "feature_counts": counts, "histories": histories, "mask": mask,
    }


if __name__ == "__main__":
    out = run()
    print("\n================ SUMMARY ================")
    cols = ["model", "n_features", "accuracy", "precision",
            "recall", "f1", "auc", "train_time_s"]
    print(out["results"][cols].to_string(index=False))
    print(f"\nSelected features ({len(out['selected_features'])}):")
    print(", ".join(out["selected_features"]))
