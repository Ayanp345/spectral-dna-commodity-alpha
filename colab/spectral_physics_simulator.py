import numpy as np

BAND_START, BAND_END, BAND_STEP = 400, 2500, 5
WAVELENGTHS = np.arange(BAND_START, BAND_END + BAND_STEP, BAND_STEP).astype(float)
N_BANDS = len(WAVELENGTHS)

# Known absorption-feature reference library (for later interpretation matching)
USGS_LIKE_LIBRARY = {
    "Chlorophyll-a absorption":        (680,  25),
    "Chlorophyll-b / carotenoid":      (480,  20),
    "Red-edge inflection":             (712,  15),
    "Leaf water (weak)":               (970,  30),
    "Leaf water (moderate)":           (1200, 30),
    "Leaf water (strong)":             (1450, 40),
    "Leaf water (strong, SWIR)":       (1940, 50),
    "Cellulose/lignin (dry matter)":   (2100, 35),
    "Cellulose/lignin (dry matter) 2": (2280, 35),
    "Ferric iron oxide charge-transfer": (870, 60),
    "Kaolinite Al-OH doublet (a)":     (2160, 12),
    "Kaolinite Al-OH doublet (b)":     (2208, 12),
    "Alunite OH":                      (1480, 20),
    "Alunite Al-OH":                   (2170, 15),
    "Alunite SO4":                     (2320, 15),
    "Li-bearing clay Al-OH shift":     (2200, 15),
    "Li-bearing clay OH combination":  (2340, 20),
    "Methane overtone (SWIR)":         (2300, 25),
    "Methane overtone (weak)":         (1650, 20),
}


def _gaussian_dip(wl, center, fwhm, depth):
    sigma = fwhm / 2.3548
    return depth * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def _base_continuum(wl, slope=0.00006, intercept=0.25):
    """Smooth soil/rock/vegetation continuum baseline."""
    return intercept + slope * (wl - 400)


def _red_edge(wl, inflection=712, steepness=0.05, low=0.05, high=0.45):
    return low + (high - low) / (1 + np.exp(-steepness * (wl - inflection)))


def simulate_spectrum(label, severity=0.0, rng=None):
    """
    Generate one pixel spectrum for a given class.

    label options:
      'healthy_crop', 'diseased_crop', 'water_stressed_crop',
      'bare_soil', 'feldspar', 'kaolinite_alteration', 'alunite_alteration',
      'li_bearing_clay', 'urban_concrete'

    severity in [0,1]: for stress/disease classes, controls how deep the
    early-stage signal is buried (0 = nearly healthy, 1 = fully expressed).
    This is what lets us test EARLY detection — the entire financial thesis
    depends on catching severity ~0.1-0.3 before it's visible to the eye
    or to a simple NDVI threshold.
    """
    rng = rng or np.random.default_rng()
    wl = WAVELENGTHS
    spec = np.zeros_like(wl)

    if label in ("healthy_crop", "diseased_crop", "water_stressed_crop"):
        # vegetation base: red edge + chlorophyll dip + water dips
        veg_vigor = 1.0
        chl_depth = 0.35
        water_depth_scale = 1.0
        cellulose_scale = 0.3  # senescence / dry matter exposure

        if label == "diseased_crop":
            # Early-stage disease: subtle RE-680 chlorophyll degradation +
            # a faint red-shift in the red edge position + slight cellulose
            # rise (necrosis) — all BELOW what a human eye or plain NDVI
            # threshold would flag at low severity. This is the exact
            # "diseased vs healthy-stressed" ambiguity NDVI can't resolve.
            chl_depth = 0.35 * (1 - 0.6 * severity)
            cellulose_scale = 0.3 + 0.5 * severity
            re_shift = -8 * severity  # red edge blue-shift with disease
        elif label == "water_stressed_crop":
            # Water stress: water-band deepening dominates, chlorophyll
            # holds up much longer -> distinguishable from disease ONLY
            # via joint band structure, not any single index.
            water_depth_scale = 1.0 + 1.2 * severity
            chl_depth = 0.35 * (1 - 0.15 * severity)
            re_shift = 0
        else:
            re_shift = 0

        spec += _red_edge(wl, inflection=712 + re_shift)
        spec += _gaussian_dip(wl, 480, 20, chl_depth * 0.6)
        spec += _gaussian_dip(wl, 680, 25, chl_depth)
        for center, fwhm in [(970, 30), (1200, 30), (1450, 40), (1940, 50)]:
            spec -= _gaussian_dip(wl, center, fwhm, 0.18 * water_depth_scale)
        spec -= _gaussian_dip(wl, 2100, 35, 0.05 * cellulose_scale)
        spec -= _gaussian_dip(wl, 2280, 35, 0.04 * cellulose_scale)
        spec += 0.05  # NIR plateau lift
        spec = np.clip(spec, 0.02, 0.9)

    elif label == "bare_soil":
        spec = _base_continuum(wl, slope=0.00009, intercept=0.15)
        spec -= _gaussian_dip(wl, 1450, 40, 0.03)
        spec -= _gaussian_dip(wl, 1940, 50, 0.03)

    elif label == "feldspar":
        # Relatively featureless mineral — the "boring" negative class
        # that a naive band-ratio miner will confuse with altered minerals.
        spec = _base_continuum(wl, slope=0.00003, intercept=0.35)
        spec -= _gaussian_dip(wl, 1400, 60, 0.02)

    elif label == "kaolinite_alteration":
        spec = _base_continuum(wl, slope=0.00002, intercept=0.40)
        spec -= _gaussian_dip(wl, 2160, 12, 0.10 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 2208, 12, 0.14 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 1400, 40, 0.05)

    elif label == "alunite_alteration":
        spec = _base_continuum(wl, slope=0.00002, intercept=0.42)
        spec -= _gaussian_dip(wl, 1480, 20, 0.08 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 2170, 15, 0.12 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 2320, 15, 0.09 * (0.4 + 0.6 * severity))

    elif label == "li_bearing_clay":
        # This is the economically interesting one: pegmatite-adjacent
        # Li-bearing clay alteration halo used as a lithium exploration proxy.
        spec = _base_continuum(wl, slope=0.000025, intercept=0.38)
        spec -= _gaussian_dip(wl, 2200, 15, 0.11 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 2340, 20, 0.09 * (0.4 + 0.6 * severity))
        spec -= _gaussian_dip(wl, 1480, 20, 0.03)

    elif label == "urban_concrete":
        spec = _base_continuum(wl, slope=0.00001, intercept=0.30)
        # deliberately near-flat / low spectral contrast

    else:
        raise ValueError(f"unknown label {label}")

    # Sensor / atmosphere realism: multiplicative illumination jitter,
    # additive electronic noise, and water-vapor gap masking near 1350-1420nm
    # and 1800-1950nm (bands typically dropped in real L2A products).
    illum = rng.normal(1.0, 0.03)
    noise = rng.normal(0, 0.004, size=N_BANDS)
    spec = np.clip(spec * illum + noise, 0.0, 1.0)
    return spec


VALID_LABELS = [
    "healthy_crop", "diseased_crop", "water_stressed_crop", "bare_soil",
    "feldspar", "kaolinite_alteration", "alunite_alteration",
    "li_bearing_clay", "urban_concrete",
]

WATER_VAPOR_MASK = ((WAVELENGTHS >= 1350) & (WAVELENGTHS <= 1420)) | \
                   ((WAVELENGTHS >= 1800) & (WAVELENGTHS <= 1950))


def simulate_dataset(n_per_class=2000, seed=42, severity_dist="uniform",
                      drop_water_vapor_bands=True):
    """Build a labeled synthetic pixel dataset spanning all classes with
    varying severity for the stress/alteration classes (the hard, realistic
    cases that matter for early detection).

    drop_water_vapor_bands: real L2A products physically CANNOT measure
    reflectance inside the deep atmospheric water-vapor absorption windows
    (~1350-1420nm, ~1800-1950nm) - the sensor gets pure noise there, so
    those bands are dropped entirely rather than NaN-masked-then-imputed
    (there is no valid signal to impute FROM).
    """
    rng = np.random.default_rng(seed)
    X, y, sev = [], [], []
    for label in VALID_LABELS:
        for _ in range(n_per_class):
            if label in ("diseased_crop", "water_stressed_crop",
                         "kaolinite_alteration", "alunite_alteration",
                         "li_bearing_clay"):
                s = rng.uniform(0, 1) if severity_dist == "uniform" else rng.beta(2, 5)
            else:
                s = 0.0
            X.append(simulate_spectrum(label, severity=s, rng=rng))
            y.append(label)
            sev.append(s)
    X = np.array(X)
    wl = WAVELENGTHS.copy()
    if drop_water_vapor_bands:
        keep = ~WATER_VAPOR_MASK
        X = X[:, keep]
        wl = wl[keep]
    return X, np.array(y), np.array(sev), wl


def load_real_envi_cube(hdr_path, img_path):
    """
    STUB for swapping in real data later.
    With the `spectral` (SPy) package + rasterio, a real Pixxel L2A GeoTIFF
    or ENVI cube loads as:

        import spectral as sp
        cube = sp.open_image(hdr_path)   # (rows, cols, bands)
        pixels = cube.load().reshape(-1, cube.nbands)

    Everything downstream (SAE training, interpretation, classifier,
    finance layer) consumes a flat (n_pixels, n_bands) float array —
    identical shape to what simulate_dataset() returns — so no other
    code changes are needed to go from synthetic to real.
    """
    raise NotImplementedError(
        "Requires real Pixxel/AVIRIS/Hyperion cube + `spectral`/`rasterio` "
        "packages and file access not available in this sandbox."
    )


if __name__ == "__main__":
    X, y, sev, wl = simulate_dataset(n_per_class=50)
    print("Dataset shape:", X.shape, "classes:", set(y))
