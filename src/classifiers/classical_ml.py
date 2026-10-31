"""Classical ML on the feature table: SVM (RBF) and Random Forest, with proper preprocessing and tuning."""
import os
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

GROUPS = ("shape", "stat", "glcm", "hog")


def select_columns(df, groups):
    return [c for c in df.columns if c.split("_")[0] in groups and c != "mask_ok"]


def usable_columns(train_df, cols):
    """Drop columns that are all-NaN or constant ON THE TRAIN SPLIT (e.g. shape_* with 'full' masks)."""
    keep = []
    for c in cols:
        x = train_df[c]
        if x.notna().any() and x.nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def build_pipeline(kind, seed=42, pca=0):
    """Imputation (median, fitted on train only) -> [scaling (+ optional PCA) for SVM] -> classifier.
    pca = number of components (0 = off). PCA shrinks the 300+ correlated HOG/GLCM dims and makes the
    RBF-SVM several times faster. 'probability' is deliberately NOT passed to SVC (deprecated in
    scikit-learn 1.9); calibrated probabilities are added at save time in experiment.py."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == "svm":
        steps += [("scale", StandardScaler())]
        if pca:
            steps += [("pca", PCA(n_components=pca, random_state=seed))]
        steps += [("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed, cache_size=300))]
    elif kind == "rf":
        steps += [("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=seed))]
    else:
        raise ValueError(kind)
    return Pipeline(steps)


PARAM_DISTS = {
    "svm": {"clf__C": loguniform(0.1, 100), "clf__gamma": loguniform(1e-4, 1e-1)},
    "rf": {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 10, 20, 40],
           "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.1, 0.3]},
}


def tune(kind, X, y, n_iter=12, tune_n=4000, cv=3, seed=42, pca=0):
    """Randomised search with stratified CV on the TRAIN split only (optionally on a subsample,
    because RBF-SVM is slow). Scoring = ROC-AUC. Returns the best parameter dict."""
    if len(X) > tune_n:
        X, _, y, _ = train_test_split(X, y, train_size=tune_n, stratify=y, random_state=seed)
    search = RandomizedSearchCV(build_pipeline(kind, seed=seed, pca=pca), PARAM_DISTS[kind], n_iter=n_iter,
                                scoring="roc_auc", cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                n_jobs=min(4, os.cpu_count() or 1) if kind == "svm" else 1, random_state=seed, refit=False)
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def get_scores(pipe, X):
    """Continuous score for ROC-AUC: probability if available, else the SVM decision function."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)
