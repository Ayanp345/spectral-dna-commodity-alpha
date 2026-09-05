import numpy as np
from scipy.stats import norm

-

def stress_score_to_yield_anomaly(stress_score, calib_slope=-0.42, calib_intercept=0.0,
                                   noise_sigma=0.015, rng=None):
    """
    stress_score in [0,1] (aggregated SAE disease/water-stress MSF activation
    over a field/region). calib_slope calibrated so that a fully-expressed
    stress signal (score=1) corresponds to roughly a -42% yield anomaly
    (severe blight/drought event scale) while score=0.15 (early, subtle
    detection - the regime our classifier showed still separable) implies
    roughly -6% yield anomaly, consistent with published crop-loss studies
    for early-detected fungal/water stress vs a healthy baseline.
    """
    rng = rng or np.random.default_rng()
    yield_anomaly = calib_intercept + calib_slope * stress_score
    yield_anomaly += rng.normal(0, noise_sigma)
    return yield_anomaly


def yield_anomaly_to_price_impact(yield_anomaly, supply_elasticity=0.35):
    """
    Short-run price elasticity of supply for grains is well-documented in the
    -0.3 to -0.5 range (inelastic: small supply changes -> large price moves).
    price_pct_change = -yield_anomaly / supply_elasticity
    (a -6% yield anomaly with elasticity 0.35 -> +17% expected futures move)
    """
    return -yield_anomaly / supply_elasticity



def bs_price_and_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    vega = S * norm.pdf(d1) * np.sqrt(T)  # per 1.0 (100%) change in sigma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) / 365.0
    return dict(price=price, delta=delta, vega=vega / 100, gamma=gamma, theta=theta)


def greeks_scenario_table(S0, price_impact_pct, strikes, T=60/365, r=0.05,
                           sigma_before=0.22, iv_shock=0.06, option_type="call"):
    """
    Compares Greeks under the market's current (pre-signal) pricing vs the
    repriced scenario implied by the spectral supply-shock forecast:
      - underlying moves by price_impact_pct
      - implied vol rises by iv_shock (grain option IV historically jumps
        on confirmed supply-disruption news; using a conservative flat bump)
    """
    S1 = S0 * (1 + price_impact_pct)
    sigma_after = sigma_before + iv_shock
    rows = []
    for K in strikes:
        before = bs_price_and_greeks(S0, K, T, r, sigma_before, option_type)
        after = bs_price_and_greeks(S1, K, T, r, sigma_after, option_type)
        rows.append(dict(
            strike=K, S_before=S0, S_after=S1,
            delta_before=before["delta"], delta_after=after["delta"],
            vega_before=before["vega"], vega_after=after["vega"],
            price_before=before["price"], price_after=after["price"],
            delta_change=after["delta"] - before["delta"],
            vega_change=after["vega"] - before["vega"],
        ))
    return rows



def newey_west_ols(y, X, lags=4):
    """OLS with Newey-West HAC standard errors, pure numpy (no statsmodels
    available in this sandbox). X should already include an intercept column."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    S = np.zeros((k, k))
    for t in range(n):
        S += resid[t] ** 2 * np.outer(X[t], X[t])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        for t in range(lag, n):
            term = resid[t] * resid[t - lag] * np.outer(X[t], X[t - lag])
            S += w * (term + term.T)

    cov_beta = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    tstats = beta / se
    return dict(beta=beta, se=se, tstats=tstats, resid=resid)


def simulate_commodity_portfolio(n_days=750, seed=11, use_signal_overlay=False,
                                  signal_series=None, signal_predictive_power=0.35,
                                  signal_lag=5):
    """
    Simulates a commodity-heavy portfolio (corn, soybean, iron-ore-proxy
    equity) against a broad commodity benchmark index, computes Jensen's
    Alpha via CAPM regression, and optionally overlays the spectral signal
    as a tactical tilt (increase weight ahead of a positive supply-shock
    forecast, i.e. long the commodity ahead of the price-supportive
    yield-shortfall signal).

    IMPORTANT / HONEST MODELING NOTE: the whole premise being tested is
    "IF the spectral signal has genuine forward-looking information about
    corn returns (which is the financial thesis - early stress detection
    predates the market-moving USDA report), THEN does tilting on it add
    alpha net of the CAPM benchmark exposure?" So corn_ret is explicitly
    given a component driven by the LAGGED signal (signal known at t-lag,
    return realizes at t -> no lookahead bias). signal_predictive_power
    controls how much of corn's variance the signal actually explains -
    0.35 here is a deliberately modest, not-too-generous assumption; set
    it to 0 to see the (honest) null result with a signal that has no
    real information, as in the earlier uncorrelated-signal run.
    """
    rng = np.random.default_rng(seed)
    mkt_ret = rng.normal(0.0003, 0.011, n_days)          # benchmark commodity index
    idio_corn = rng.normal(0.0, 0.014, n_days)
    idio_soy = rng.normal(0.0, 0.013, n_days)
    idio_iron = rng.normal(0.0, 0.017, n_days)

    beta_corn, beta_soy, beta_iron = 0.55, 0.6, 0.8
    corn_ret = 0.55 * mkt_ret * beta_corn + idio_corn
    soy_ret = 0.55 * mkt_ret * beta_soy + idio_soy
    iron_ret = 0.55 * mkt_ret * beta_iron + idio_iron

    if use_signal_overlay and signal_series is not None:
        # Embed genuine (lagged, no-lookahead) predictive signal into corn's
        # realized return, scaled so it explains a modest, stated fraction
        # of corn return variance -- this represents the "if the early
        # spectral detection is real, here's the payoff" scenario.
        lagged_signal = np.roll(signal_series, signal_lag)
        lagged_signal[:signal_lag] = 0
        signal_component = signal_predictive_power * np.std(idio_corn) * lagged_signal
        corn_ret = corn_ret + signal_component

    base_weights = np.array([0.4, 0.35, 0.25])  # corn, soy, iron-proxy
    asset_rets = np.vstack([corn_ret, soy_ret, iron_ret])  # (3, n_days)

    if use_signal_overlay:
        assert signal_series is not None and len(signal_series) == n_days
        # signal_series in [-1, 1]: positive = bullish supply-shock forecast
        # (spectral stress detected -> expect price up) -> tilt toward corn.
        # Also lagged by the same amount to avoid using the same-day signal
        # to trade the same-day return it was used to construct above.
        lagged_signal = np.roll(signal_series, signal_lag)
        lagged_signal[:signal_lag] = 0
        tilt = 0.15 * lagged_signal  # max +/-15% weight tilt
        w_corn = np.clip(base_weights[0] + tilt, 0.05, 0.75)
        w_soy = np.clip(base_weights[1] - tilt / 2, 0.05, 0.6)
        w_iron = 1 - w_corn - w_soy
        weights = np.vstack([w_corn, w_soy, w_iron])
        port_ret = np.sum(weights * asset_rets, axis=0)
    else:
        port_ret = base_weights @ asset_rets

    rf = 0.00008  # ~2%/yr risk-free, daily
    excess_port = port_ret - rf
    excess_mkt = mkt_ret - rf

    X = np.column_stack([np.ones(n_days), excess_mkt])
    ols = newey_west_ols(excess_port, X, lags=5)
    alpha_daily, beta_mkt = ols["beta"]
    alpha_tstat = ols["tstats"][0]
    alpha_annualized = alpha_daily * 252

    sharpe = np.mean(excess_port) / np.std(excess_port) * np.sqrt(252)
    cum_ret = np.cumprod(1 + port_ret) - 1
    max_dd = np.min(cum_ret - np.maximum.accumulate(cum_ret))

    return dict(
        alpha_daily=alpha_daily, alpha_annualized=alpha_annualized,
        alpha_tstat=alpha_tstat, beta_mkt=beta_mkt, sharpe=sharpe,
        cum_return=cum_ret[-1], max_drawdown=max_dd,
        port_ret=port_ret, mkt_ret=mkt_ret, cum_ret_series=cum_ret,
    )


def build_signal_series_from_stress(stress_scores):
    """Turn a raw [0,1] SAE stress-score time series into a [-1,1] tactical
    tilt signal via a simple zscore + tanh squashing (bounded, robust to
    outliers -- a standard signal-processing step before feeding into a
    portfolio overlay)."""
    z = (stress_scores - stress_scores.mean()) / (stress_scores.std() + 1e-8)
    return np.tanh(z)
