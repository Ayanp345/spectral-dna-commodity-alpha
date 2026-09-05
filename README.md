# Spectral DNA Fingerprinting → Commodity Alpha Generation

An experimental research pipeline using **Sparse Autoencoders (SAEs)** to decompose hyperspectral pixel spectra into interpretable **monosemantic spectral features (MSFs)**, connect those features to early crop-stress and mineral-alteration detection, and explore their downstream use in commodity supply forecasting, options repricing, and portfolio alpha.

```text
Hyperspectral Pixel Spectra
          ↓
Sparse Autoencoder (SAE)
          ↓
Interpretable Spectral Features
          ↓
Crop Stress / Mineral Alteration Detection
          ↓
Commodity Supply Forecast
          ↓
Commodity Price Impact
          ↓
Options Greeks Repricing
          ↓
Portfolio Alpha Backtest

Research prototype: Current experiments use physically informed synthetic hyperspectral spectra. The commodity and portfolio layers are experimental simulations and should not be interpreted as live trading signals.

Project Overview

The central research question is:

Can Sparse Autoencoders decompose hyperspectral reflectance vectors into physically interpretable spectral features that remain useful for downstream scientific and financial analysis?

Most SAE research focuses on decomposing neural-network activations, particularly language-model representations.

This project explores the same interpretability framework on raw hyperspectral reflectance vectors.

The resulting spectral features are connected to a complete experimental pipeline:

spectral signal → crop/mineral information → commodity supply signal → market repricing → portfolio analysis

The emphasis is not simply on predictive accuracy. The primary motivation is interpretability and auditability: whether individual learned features can be connected to physically meaningful absorption regions that a geologist, agronomist, or researcher can independently inspect.

Two Execution Tracks
Track	Location	Purpose
PyTorch SAE	colab/spectral_dna_fingerprinting_colab.ipynb	Full PyTorch implementation designed for Google Colab
NumPy Reference	src/sae_numpy_reference.py	Independently executed reference implementation reproducing the SAE mathematics

The NumPy implementation uses the same underlying SAE architecture, loss, Adam optimization update, and decoder-renormalization procedure as the PyTorch implementation.

Why Google Colab?

The PyTorch implementation is provided through Google Colab so it can be run without requiring a local GPU setup.

At the current experimental scale, GPU acceleration is not essential. GPU becomes more relevant when scaling toward millions of real hyperspectral pixels and substantially larger latent spaces.

Methodology
1. Physically Informed Spectral Data

The current experiments use synthetic spectra constructed from documented spectral absorption regions rather than arbitrary random noise.

Examples include:

Chlorophyll absorption around 680 nm
Red-edge region around 712 nm
Leaf-water absorption around 970 / 1200 / 1450 / 1940 nm
Kaolinite Al-OH features around 2160 / 2208 nm
Alunite-related features around 1480 / 2170 / 2320 nm
Lithium-bearing clay alteration proxies around 2200 / 2340 nm

The purpose is to provide the SAE with structured spectral signals that correspond to physically meaningful phenomena.

The repository also includes a real-data loading interface for ENVI/GeoTIFF hyperspectral cubes.

2. Sparse Autoencoder

The SAE learns a sparse latent representation of the hyperspectral spectra.

Conceptually:

Input spectrum
     ↓
Encoder
     ↓
Sparse latent representation
     ↓
Decoder
     ↓
Reconstructed spectrum

The sparse latent representation contains more features than the original representation, while sparsity encourages individual features to activate selectively.

This provides a mechanism for investigating whether individual latent features correspond to interpretable spectral structures.

3. Spectral Feature Interpretation

After training, individual latent features are analyzed against wavelength regions.

Example discovered features:

Feature	Wavelength	Interpretation	Dominant Class	Purity
MSF_0056	2170 nm	Alunite Al-OH	Alunite alteration	0.97
MSF_0105	1450 nm	Leaf water	Water-stressed crop	1.00
MSF_0372	2208 nm	Kaolinite Al-OH	Kaolinite alteration	0.53

These features were discovered without using class labels during SAE training.

See:

outputs/03_feature_dashboard.png

What Is Novel?

The underlying Sparse Autoencoder methodology is established research.

The project builds on published SAE work, including:

Towards Monosemanticity — Bricken et al., 2023
TopK Sparse Autoencoders — Gao et al., 2024

These approaches were developed primarily to study interpretable features in neural-network activations.

The exploratory research direction here is:

Applying SAE-based feature decomposition to raw hyperspectral reflectance vectors and connecting the resulting interpretable features to a downstream commodity-intelligence pipeline.

This should be viewed as an underexplored application and research direction, not as a claim of inventing a new SAE algorithm.

Experimental Results
1. Unsupervised Feature Discovery

The SAE was trained on 2,700 synthetic pixels across 9 classes:

Healthy crop
Diseased crop
Water-stressed crop
Bare soil
Feldspar
Kaolinite alteration
Alunite alteration
Lithium-clay alteration
Urban concrete

Several latent features aligned with physically meaningful absorption regions despite the SAE being trained without class supervision.

Example:

MSF_0056  → 2170 nm  → Alunite Al-OH
MSF_0105  → 1450 nm  → Leaf water
MSF_0372  → 2208 nm  → Kaolinite Al-OH

The corresponding feature dashboard is available at:

outputs/03_feature_dashboard.png

2. Downstream Classification

A Random Forest was used to compare raw spectral bands, PCA representations, and SAE representations.

Feature Set	Accuracy	Low-Severity Accuracy
Raw bands (375)	0.983	0.972
PCA (32 components)	0.996	0.992
SAE codes (512-dim)	0.981	0.968
Interpretation

PCA performs best on classification accuracy in this synthetic experiment.

This result is intentionally reported rather than hidden.

The purpose of the SAE in this project is not to claim that it automatically outperforms PCA.

The potential advantage is interpretability:

SAE features can be inspected individually.
Features can be mapped to physical wavelength regions.
Researchers can independently evaluate whether a feature corresponds to a meaningful spectral phenomenon.
PCA components are linear combinations of many bands and are less directly interpretable as individual physical features.
3. Superposition Experiment

A sub-pixel linear mixture experiment was performed using healthy and early-disease spectra.

A linear probe was used to estimate the mixing fraction.

Feature Set	R²	MAE
Raw bands	0.598	0.149
PCA (32)	0.663	0.138
SAE codes	0.567	0.154

PCA remains competitive in this experiment.

This is expected to some extent because PCA is designed around linear variance-preserving representations, while the current experiment uses linear spectral mixtures.

A more interesting future test is whether SAE representations provide advantages under:

Nonlinear spectral mixing
Multiplicative illumination effects
Mineral grain boundaries
Unseen material combinations
Distribution shift
Commodity Intelligence Layer

The project connects the spectral signal to an experimental commodity-forecasting pipeline.

The conceptual chain is:

Spectral Stress Signal
        ↓
Crop Stress Estimate
        ↓
Yield Anomaly
        ↓
Commodity Supply Shock
        ↓
Forecast Price Impact
        ↓
Options Repricing
        ↓
Portfolio Overlay
Example Simulation

For an experimental stress score of 0.20:

Stress score:             0.20
Forecast yield anomaly:  -7.9%
Forecast futures impact: +22.7%

The price-impact calculation uses a short-run supply elasticity assumption of 0.35, within the broad range used in the project's calibration.

These are experimental model outputs, not market forecasts.

Options Greeks Repricing

The pipeline includes an experimental Black-Scholes-based options repricing layer.

For the simulated 460 strike example:

Call Delta: 0.554 → 0.977
Vega:       0.737 → 0.124

The underlying and implied volatility are repriced based on the simulated commodity forecast.

See:

outputs/04_options_greeks.png

Portfolio Backtest

The portfolio layer uses a simulated 750-day path and calculates annualized alpha using a Newey-West HAC-robust CAPM framework.

Strategy	Annual Alpha	t-stat	Sharpe	Cumulative Return
Baseline	3.93%	0.45	0.56	32.0%
Signal Overlay	10.42%	1.11	0.95	59.7%
Noise Overlay	3.63%	0.41	0.53	30.7%

The noise-overlay experiment is included as a sanity check.

When the same portfolio mechanics are supplied with a signal containing no useful information, the resulting performance returns close to the baseline.

Important Statistical Caveat

The reported signal-overlay t-stat = 1.11 is not strong statistical significance.

This is a single simulated 750-day path.

It should not be interpreted as evidence of deployable trading alpha.

A real study would require:

Multiple independent historical periods
Real hyperspectral observations
Historical commodity prices
Robust out-of-sample evaluation
Block bootstrap or equivalent time-series statistical testing

See:

outputs/05_portfolio_backtest.png

Why Synthetic Data?

The current environment does not provide direct access to real Pixxel, AVIRIS, EMIT, or USGS hyperspectral datasets.

Instead of training the model on arbitrary random noise, the simulator generates spectra using documented absorption-feature wavelengths.

This gives the experiment physically structured signals while keeping the current implementation reproducible.

The repository contains a loader interface designed to accept real hyperspectral data in formats such as:

ENVI .hdr / raster data
GeoTIFF hyperspectral cubes

Once real data are supplied, the downstream modules operate on the same basic representation:

(n_pixels, n_bands)
Repository Structure
spectral-dna-commodity-alpha/
│
├── README.md
│
├── colab/
│   ├── spectral_dna_fingerprinting_colab.ipynb
│   ├── spectral_sae.py
│   ├── spectral_physics_simulator.py
│   ├── sae_numpy_reference.py
│   ├── feature_interpretation.py
│   ├── downstream_classifier.py
│   └── commodity_alpha_pipeline.py
│
├── src/
│   ├── spectral_sae.py
│   ├── spectral_physics_simulator.py
│   ├── sae_numpy_reference.py
│   ├── feature_interpretation.py
│   ├── downstream_classifier.py
│   └── commodity_alpha_pipeline.py
│
└── outputs/
    ├── 01_example_spectra.png
    ├── 02_sae_training.png
    ├── 03_feature_dashboard.png
    ├── 04_options_greeks.png
    ├── 05_portfolio_backtest.png
    └── finance_results.pkl
Running the Project
Google Colab

Open:

colab/spectral_dna_fingerprinting_colab.ipynb

The notebook contains the PyTorch implementation of the SAE pipeline.

Run the notebook cells from top to bottom.

NumPy Reference Implementation

The NumPy implementation can be executed locally:

python src/sae_numpy_reference.py

The NumPy implementation was used to validate the pipeline and generate the reported experimental outputs.

Limitations

This repository is an experimental research prototype, not a production trading system.

1. Synthetic Spectra

The current experiments use physically informed synthetic spectra rather than real Pixxel, AVIRIS, EMIT, or other operational hyperspectral imagery.

Real-data validation is required.

2. Commodity Calibration

The yield-anomaly and price-impact relationships use assumed calibration parameters.

They are not fitted to a live market feed.

A real deployment would require historical triples such as:

Hyperspectral observation
        +
Agricultural yield
        +
Commodity price
3. Simulated Portfolio Path

The portfolio backtest is based on a single simulated 750-day path.

Statistical significance requires multiple independent historical periods and appropriate time-series validation.

4. PCA Is a Strong Baseline

PCA performs strongly in the current synthetic experiments.

The SAE should therefore not be justified solely through classification accuracy.

The primary research motivation is interpretability, physical feature attribution, and potential robustness under more difficult distribution shifts.

Future Research

The most important next step is validation on real hyperspectral observations.

Potential experiments include:

Real satellite and airborne hyperspectral cubes
Pixxel hyperspectral data
AVIRIS / EMIT datasets
Real crop-stress monitoring
Mineral exploration
Nonlinear spectral mixing
Illumination variation
Unseen material combinations
Temporal crop monitoring
Distribution shift
SAE feature stability
Cross-region generalization
Comparison against PCA
Comparison against ICA
Supervised representation learning
Causal validation of spectral signals against agricultural outcomes

A particularly important research question is:

Do SAE-derived spectral features remain physically interpretable and stable when evaluated on unseen materials, locations, seasons, illumination conditions, and real hyperspectral observations?

Research Position

This project does not claim that SAE representations universally outperform PCA.

Instead, it investigates whether sparse latent representations can provide something that conventional dimensionality-reduction methods may not provide as directly:

physically interpretable and auditable spectral features.

The long-term research direction is to connect those interpretable features to real-world observations and evaluate whether they can provide reliable early information about agricultural and commodity-market conditions.

Outputs

The repository currently contains:

Output	Description
01_example_spectra.png	Example generated spectral observations
02_sae_training.png	SAE training behavior
03_feature_dashboard.png	Interpretable spectral feature analysis
04_options_greeks.png	Simulated options Greek repricing
05_portfolio_backtest.png	Portfolio backtest results
finance_results.pkl	Saved finance-layer results
References

The SAE methodology builds on prior work in mechanistic interpretability and sparse representation learning, including:

Bricken et al. (2023)
Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.

Gao et al. (2024)
Scaling and Evaluating Sparse Autoencoders.

The present project explores transferring this general methodology from neural-network activations to hyperspectral reflectance representations.

Disclaimer

This repository is for research and educational purposes.

The commodity forecasts, options repricing results, portfolio returns, and alpha estimates are generated from experimental/simulated assumptions and should not be interpreted as financial advice, live trading signals, or evidence of real-world investment performance.

Real-world deployment requires validation using independent real hyperspectral datasets, historical agricultural data, market data, rigorous statistical testing, and appropriate out-of-sample evaluation.

Core Idea
                    SPECTRAL DNA
                         │
                         ▼
              Sparse Autoencoder
                         │
                         ▼
           Monosemantic Spectral Features
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Crop Stress             Mineral Alteration
             │                       │
             └───────────┬───────────┘
                         ▼
                 Supply Forecast
                         │
                         ▼
                 Commodity Pricing
                         │
                         ▼
                 Options Repricing
                         │
                         ▼
                  Portfolio Alpha

Spectral information → interpretable features → real-world signals → quantitative decision framework.
