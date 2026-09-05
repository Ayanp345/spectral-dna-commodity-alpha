# Spectral DNA Fingerprinting → Commodity Alpha Generation

End-to-end pipeline: unsupervised Sparse Autoencoder (SAE) decomposition of
hyperspectral pixel spectra → interpretable "monosemantic spectral features"
(MSFs) → early crop-stress / mineral-alteration detection → commodity supply
forecast → options Greeks re-pricing → Jensen's Alpha portfolio backtest.

## Two execution tracks, and why

| Track | Where it runs | Why |
|---|---|---|
| `colab/spectral_dna_fingerprinting_colab.ipynb` | **Google Colab** (free, GPU optional) | The real PyTorch SAE. Your laptop has no GPU and this sandbox has no internet to install torch — Colab solves both for free in one click, and GPU isn't actually required at this scale (see below). |
| `src/sae_numpy_reference.py` | **Here, already run** | A hand-derived NumPy re-implementation of the identical SAE math (same architecture, same loss, same Adam update, same decoder-renormalization trick) — used to validate the entire pipeline and produce every number/plot in this repo *right now*, with zero unverifiable claims. |

**I will not claim a PyTorch run I can't actually perform in this sandbox.**
Everything reported below was genuinely executed (in NumPy); the Colab
notebook is the literal same algorithm in real PyTorch, ready for you to run
in ~2 minutes with a "Run all."

Compute reality check: the SAE here is <1M parameters, trained on ~15-20k
tiny 375-dimensional vectors. This is not a vision transformer — it fits
comfortably in Colab's free CPU tier in well under a minute. GPU only starts
mattering once you scale to millions of real pixels and a 10,000+-dim latent
space, which is a one-line change (`latent_dim=10000`) once you have real data.

## What's genuinely novel here vs. what's standard, published research

Being direct about this so nothing here is oversold:

- **Standard, published, and correctly attributed:** the Sparse Autoencoder
  architecture, L1 sparsity penalty, and TopK variant are from Anthropic's
  "Towards Monosemanticity" (Bricken et al., 2023) and OpenAI's TopK-SAE
  follow-up (Gao et al., 2024) — both developed to decompose *language model
  activations* into interpretable features.
- **The actually novel part:** applying that exact toolkit to *raw
  hyperspectral reflectance vectors* instead of neural activations, and
  wiring the resulting interpretable features into a commodity-trading
  pipeline end to end. That combination is genuinely underexplored — but
  it's an application of solid, real methods, not a fabricated algorithm.

## Why simulated data (and how that's handled honestly)

This sandbox has no internet egress, so a real Pixxel L2A cube, AVIRIS/EMIT
scene, or the USGS Spectral Library can't be downloaded here. Rather than
generate arbitrary random noise, `src/spectral_physics_simulator.py` builds
spectra from **real, documented absorption-feature wavelengths** (chlorophyll
680nm, red-edge ~712nm, leaf water 970/1200/1450/1940nm, kaolinite Al-OH
doublet 2160/2208nm, alunite 1480/2170/2320nm, a lithium-bearing-clay
alteration proxy at 2200/2340nm used in real pegmatite lithium exploration,
etc.) so the SAE has genuine physical structure to decompose — an SAE trained
on pure noise learns nothing interpretable, and that would have been a
worthless demo. `load_real_envi_cube()` in the same file is a ready stub:
point it at a real `.hdr`/ENVI or GeoTIFF cube via the `spectral` (SPy) +
`rasterio` packages and every downstream module (SAE, classifier, finance
layer) works unchanged, since they only ever see a flat `(n_pixels, n_bands)`
array.

## Results actually produced (NumPy reference run, this sandbox)

**1. Unsupervised feature discovery works.** Training on 2,700 synthetic
pixels across 9 classes (healthy/diseased/water-stressed crop, bare soil,
feldspar, kaolinite/alunite/Li-clay alteration, urban concrete), the SAE
discovered latent features that — with **no label supervision at all** —
correspond to real absorption lines with high class purity, e.g.:

```
MSF_0056  2170nm  -> Alunite Al-OH        (dominant class: alunite_alteration, purity=0.97)
MSF_0105  1450nm  -> Leaf water (strong)  (dominant class: water_stressed_crop, purity=1.00)
MSF_0372  2208nm  -> Kaolinite Al-OH (b)  (dominant class: kaolinite_alteration, purity=0.53)
```

See `outputs/03_feature_dashboard.png`.

**2. Downstream classification — reported honestly, not cherry-picked.**
Random Forest accuracy at low severity (<0.3, the hard early-detection
regime that matters financially):

| Feature set | Acc (all) | Acc (low-severity) |
|---|---|---|
| Raw bands (375) | 0.983 | 0.972 |
| PCA (32 components) | **0.996** | **0.992** |
| SAE codes (512-dim, sparse) | 0.981 | 0.968 |

**PCA wins on raw classification accuracy here.** This is an honest result,
not a setback to hide: PCA is near-optimal for cleanly-separated classes
under additive Gaussian noise, which is exactly this synthetic corpus. SAE's
value isn't "beats PCA on accuracy" — it's **interpretability and
auditability**: every SAE feature maps to a named, physically verifiable
wavelength (a geologist/agronomist can check "does MSF_0056 really fire on
alunite?" independent of any labels); PCA components are opaque linear
combinations of all 375 bands and can't be individually validated the same
way, which matters a lot in a regulated fund context.

**3. Superposition test (the actual motivating case for SAEs).** Sub-pixel
linear mixtures of healthy + early-disease spectra, recovering the mixing
fraction via linear probe:

| Feature set | R² | MAE |
|---|---|---|
| Raw bands | 0.598 | 0.149 |
| PCA (32) | 0.663 | 0.138 |
| SAE codes | 0.567 | 0.154 |

Also close — PCA remains a strong baseline even here with a simple linear
probe. I'm reporting this straight rather than only showing favorable
numbers. The likely reason SAE doesn't clearly dominate on *linear* mixing
tasks is that PCA is literally the optimal linear basis for linear mixtures;
SAE's disentangling advantage is expected to show up more under nonlinear
superposition (e.g. multiplicative illumination effects, true spectral
non-linear mixing at mineral grain boundaries) and under distribution shift
to unseen material combinations — worth testing with real data.

**4. Finance layer (fully wired, plausible calibration, clearly labeled
assumptions).**

- Stress score 0.20 (early-stage, still classifiable per above) →
  forecast yield anomaly **-7.9%** → forecast futures price impact **+22.7%**
  (using a 0.35 short-run supply elasticity, within the published -0.3 to
  -0.5 range for grains).
- Options Greeks re-pricing (see `outputs/04_options_greeks.png`): at the
  460 strike, call Delta moves from 0.554 → 0.977 and Vega collapses from
  0.737 → 0.124 once the underlying and IV are re-priced off the forecast —
  showing exactly how an early spectral signal should reprice a derivatives
  book *before* the market reacts to an eventual USDA report.
- Portfolio backtest (750 days, Newey-West HAC-robust CAPM alpha):

  | | Ann. Alpha | t-stat | Sharpe | Cum. Return |
  |---|---|---|---|---|
  | Baseline (no signal) | 3.93% | 0.45 | 0.56 | 32.0% |
  | + Overlay, signal has genuine predictive power | **10.42%** | 1.11 | 0.95 | 59.7% |
  | + Overlay, signal is pure noise (null/sanity check) | 3.63% | 0.41 | 0.53 | 30.7% |

  The null-signal row is there deliberately: if you feed the exact same
  overlay mechanics a signal with **zero** real information, alpha collapses
  back to baseline — confirming the uplift isn't an artifact of the backtest
  construction. The t-stat of 1.11 is *not* strong statistical significance
  on a single 3-year simulated path — that's expected and stated plainly;
  real deployment needs many independent historical periods or a
  block-bootstrap, not one path.

## File map

```
spectral_sae_alpha/
├── README.md                         <- this file
├── colab/
│   └── spectral_dna_fingerprinting_colab.ipynb   <- run the real PyTorch SAE here
├── src/
│   ├── spectral_physics_simulator.py  <- physically-informed synthetic data + real-data loader stub
│   ├── spectral_sae.py                <- PyTorch L1-SAE + TopK-SAE (for Colab)
│   ├── sae_numpy_reference.py         <- NumPy re-implementation, actually run in this sandbox
│   ├── feature_interpretation.py      <- unsupervised feature -> absorption-line matching, feature dashboard
│   ├── downstream_classifier.py       <- raw vs PCA vs SAE-code classification, incl. low-severity split
│   └── commodity_alpha_pipeline.py    <- yield anomaly, supply-shock pricing, Black-Scholes Greeks, Jensen's Alpha
└── outputs/
    ├── 01_example_spectra.png
    ├── 02_sae_training.png
    ├── 03_feature_dashboard.png
    ├── 04_options_greeks.png
    ├── 05_portfolio_backtest.png
    └── finance_results.pkl
```

## Honest limitations / what to fix before trading real money on this

1. All spectra are simulated from documented absorption physics, not a real
   Pixxel/AVIRIS cube — validate on real data before trusting the numbers.
2. The yield-anomaly and price-elasticity calibrations are set to plausible
   published ranges, not fit to a live feed — recalibrate against real
   historical (spectral-index, USDA yield, futures-price) triples.
3. The portfolio backtest is a single simulated 750-day path — statistical
   significance needs many independent historical periods or a
   block-bootstrap, not one draw.
4. PCA was a genuinely strong baseline throughout — don't deploy SAE features
   over PCA on accuracy alone; the case for SAE here is interpretability and
   (untested-on-real-data) robustness to nonlinear superposition, not raw
   predictive lift.
#   s p e c t r a l - d n a - c o m m o d i t y - a l p h a  
 