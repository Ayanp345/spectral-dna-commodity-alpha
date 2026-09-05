

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def evaluate_feature_set(X_feat, y, sev, test_size=0.25, seed=0, n_pca=None):
    if n_pca is not None:
        X_feat = PCA(n_components=n_pca, random_state=seed).fit_transform(X_feat)

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size,
                                       random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  n_jobs=-1, random_state=seed)
    clf.fit(X_feat[idx_tr], y[idx_tr])
    y_pred = clf.predict(X_feat[idx_te])

    acc_all = accuracy_score(y[idx_te], y_pred)
    f1_all = f1_score(y[idx_te], y_pred, average="macro")

    low_sev_mask = sev[idx_te] < 0.3
    if low_sev_mask.sum() > 5:
        acc_low = accuracy_score(y[idx_te][low_sev_mask], y_pred[low_sev_mask])
        f1_low = f1_score(y[idx_te][low_sev_mask], y_pred[low_sev_mask],
                           average="macro")
    else:
        acc_low, f1_low = None, None

    return dict(acc_all=acc_all, f1_all=f1_all, acc_low_severity=acc_low,
                f1_low_severity=f1_low, n_test=len(idx_te),
                n_low_severity_test=int(low_sev_mask.sum()))


def run_comparison(X_raw, Z_sae, y, sev, n_pca=32, seed=0):
    results = {}
    results["raw_bands"] = evaluate_feature_set(X_raw, y, sev, seed=seed)
    results["pca"] = evaluate_feature_set(X_raw, y, sev, n_pca=n_pca, seed=seed)
    results["sae_codes"] = evaluate_feature_set(Z_sae, y, sev, seed=seed)
    return results


def print_comparison(results):
    print(f"{'Feature set':<14}{'Acc (all)':<12}{'F1 (all)':<12}"
          f"{'Acc (low-sev)':<16}{'F1 (low-sev)':<14}")
    for name, r in results.items():
        acc_low = f"{r['acc_low_severity']:.3f}" if r['acc_low_severity'] is not None else "-"
        f1_low = f"{r['f1_low_severity']:.3f}" if r['f1_low_severity'] is not None else "-"
        print(f"{name:<14}{r['acc_all']:<12.3f}{r['f1_all']:<12.3f}"
              f"{acc_low:<16}{f1_low:<14}")
