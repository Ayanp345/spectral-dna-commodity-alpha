import numpy as np
from collections import Counter

try:
    from spectral_physics_simulator import USGS_LIKE_LIBRARY, WAVELENGTHS
except ImportError:
    from .spectral_physics_simulator import USGS_LIKE_LIBRARY, WAVELENGTHS


def match_absorption(center_wl, tolerance=15):
    best_name, best_dist = None, np.inf
    for name, (wl0, fwhm) in USGS_LIKE_LIBRARY.items():
        d = abs(center_wl - wl0)
        if d < best_dist:
            best_dist, best_name = d, name
    if best_dist <= tolerance:
        return best_name, best_dist
    return None, best_dist


def build_feature_dashboard(Z, X_raw, wavelengths, labels=None, top_n=40,
                             min_active_pixels=15, max_features_report=25):
    """
    Z         : (n_pixels, latent_dim) SAE activations (numpy)
    X_raw     : (n_pixels, n_bands) raw spectra used to produce Z
    wavelengths: (n_bands,) wavelength grid, NaNs allowed to be masked out
    labels    : optional (n_pixels,) ground-truth class strings, for purity only
    """
    n_pixels, latent_dim = Z.shape
    valid_bands = ~np.isnan(X_raw).any(axis=0)
    wl_valid = wavelengths[valid_bands]
    mean_spectrum = np.nanmean(X_raw, axis=0)[valid_bands]

    active_counts = (Z > 0).sum(axis=0)
    candidate_features = np.argsort(-active_counts)  # most-used features first

    dashboard = []
    for j in candidate_features:
        n_active = int(active_counts[j])
        if n_active < min_active_pixels:
            continue
        top_idx = np.argsort(-Z[:, j])[:min(top_n, n_active)]
        avg_spec = np.nanmean(X_raw[top_idx], axis=0)[valid_bands]
        deviation = avg_spec - mean_spectrum
        center_idx = np.argmin(deviation)  # strongest absorption dip
        center_wl = wl_valid[center_idx]
        depth = -deviation[center_idx]

        match_name, match_dist = match_absorption(center_wl)

        purity, top_label = None, None
        if labels is not None:
            lbl_counts = Counter(labels[top_idx])
            top_label, cnt = lbl_counts.most_common(1)[0]
            purity = cnt / len(top_idx)

        dashboard.append(dict(
            feature_id=int(j),
            n_active_pixels=n_active,
            center_wavelength_nm=float(center_wl),
            absorption_depth=float(depth),
            matched_signature=match_name,
            match_distance_nm=float(match_dist),
            dominant_class=top_label,
            purity=purity,
        ))
        if len(dashboard) >= max_features_report:
            break

    return dashboard


def print_dashboard(dashboard, k=15):
    print(f"{'MSF ID':<10}{'Wavelength':<12}{'Matched signature':<38}{'Class':<22}{'Purity':<8}")
    for row in dashboard[:k]:
        msf = f"MSF_{row['feature_id']:04d}"
        wl = f"{row['center_wavelength_nm']:.0f}nm"
        sig = row['matched_signature'] or "(unmatched)"
        cls = row['dominant_class'] or "-"
        pur = f"{row['purity']:.2f}" if row['purity'] is not None else "-"
        print(f"{msf:<10}{wl:<12}{sig:<38}{cls:<22}{pur:<8}")
