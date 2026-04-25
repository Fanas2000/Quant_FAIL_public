from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BermudanPutParams:
    s0: float = 100.0
    strike: float = 110.0
    maturity: float = 10.0
    risk_free_rate: float = 0.10
    volatility: float = 0.25
    n_paths: int = 100_000
    n_steps: int = 100
    n_exercise_dates: int = 10
    regression_degree: int = 2
    seed: int = 42

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put_price(params: BermudanPutParams) -> float:
    d1 = (
        log(params.s0 / params.strike)
        + (params.risk_free_rate + 0.5 * params.volatility**2) * params.maturity
    ) / (params.volatility * sqrt(params.maturity))
    d2 = d1 - params.volatility * sqrt(params.maturity)

    return params.strike * exp(-params.risk_free_rate * params.maturity) * normal_cdf(-d2) - params.s0 * normal_cdf(-d1)


def put_payoff(spot: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(strike - spot, 0.0)


def simulate_gbm_paths(params: BermudanPutParams) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)
    paths = np.empty((params.n_paths, params.n_steps + 1), dtype=float)
    paths[:, 0] = params.s0

    drift = (params.risk_free_rate - 0.5 * params.volatility**2) * params.dt
    diffusion = params.volatility * sqrt(params.dt)
    shocks = rng.standard_normal((params.n_paths, params.n_steps))
    paths[:, 1:] = params.s0 * np.exp(np.cumsum(drift + diffusion * shocks, axis=1))

    return times, paths


def bermudan_exercise_indices(params: BermudanPutParams) -> np.ndarray:
    return np.linspace(
        params.n_steps // params.n_exercise_dates,
        params.n_steps,
        params.n_exercise_dates,
        dtype=int,
    )


def american_exercise_indices(params: BermudanPutParams) -> np.ndarray:
    return np.arange(1, params.n_steps + 1, dtype=int)


def fit_polynomial_continuation(
    spot: np.ndarray,
    discounted_value: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, LinearRegression, PolynomialFeatures]:
    x = spot.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly, discounted_value)
    continuation = model.predict(x_poly)
    return continuation, model, poly


def price_put_lsm(
    params: BermudanPutParams,
    paths: np.ndarray,
    exercise_indices: np.ndarray,
) -> dict:
    maturity_index = exercise_indices[-1]
    values = put_payoff(paths[:, maturity_index], params.strike)
    exercise_time = np.full(params.n_paths, params.maturity)
    diagnostics = []
    next_index = maturity_index

    for current_index in reversed(exercise_indices[:-1]):
        discount = exp(-params.risk_free_rate * (next_index - current_index) * params.dt)
        discounted_values = values * discount

        spot = paths[:, current_index]
        intrinsic = put_payoff(spot, params.strike)
        itm = intrinsic > 0.0
        current_values = discounted_values.copy()

        if np.any(itm):
            continuation, model, poly = fit_polynomial_continuation(
                spot[itm],
                discounted_values[itm],
                params.regression_degree,
            )
            exercise = intrinsic[itm] > continuation
            itm_indices = np.flatnonzero(itm)
            exercise_indices_now = itm_indices[exercise]

            current_values[exercise_indices_now] = intrinsic[exercise_indices_now]
            exercise_time[exercise_indices_now] = current_index * params.dt

            diagnostics.append(
                {
                    "time_index": current_index,
                    "time": current_index * params.dt,
                    "n_itm": int(np.sum(itm)),
                    "n_exercised": int(np.sum(exercise)),
                    "model": model,
                    "poly": poly,
                }
            )

        values = current_values
        next_index = current_index

    initial_discount = exp(-params.risk_free_rate * next_index * params.dt)
    discounted_path_values = values * initial_discount

    return {
        "price": float(np.mean(discounted_path_values)),
        "standard_error": float(np.std(discounted_path_values, ddof=1) / sqrt(params.n_paths)),
        "discounted_path_values": discounted_path_values,
        "exercise_time": exercise_time,
        "diagnostics": diagnostics,
    }


def price_bermudan_put_lsm(params: BermudanPutParams, paths: np.ndarray) -> dict:
    return price_put_lsm(params, paths, bermudan_exercise_indices(params))


def price_american_put_lsm(params: BermudanPutParams, paths: np.ndarray) -> dict:
    return price_put_lsm(params, paths, american_exercise_indices(params))


def fit_neural_continuation(
    spot: np.ndarray,
    discounted_value: np.ndarray,
    random_state: int,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 500,
) -> tuple[np.ndarray, object]:
    x = spot.reshape(-1, 1)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            random_state=random_state,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, discounted_value)
    continuation = model.predict(x)
    return continuation, model


def price_put_lsm_neural(
    params: BermudanPutParams,
    paths: np.ndarray,
    exercise_indices: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 500,
) -> dict:
    maturity_index = exercise_indices[-1]
    values = put_payoff(paths[:, maturity_index], params.strike)
    exercise_time = np.full(params.n_paths, params.maturity)
    diagnostics = []
    next_index = maturity_index

    for current_index in reversed(exercise_indices[:-1]):
        discount = exp(-params.risk_free_rate * (next_index - current_index) * params.dt)
        discounted_values = values * discount

        spot = paths[:, current_index]
        intrinsic = put_payoff(spot, params.strike)
        itm = intrinsic > 0.0
        current_values = discounted_values.copy()

        if np.any(itm):
            continuation, model = fit_neural_continuation(
                spot[itm],
                discounted_values[itm],
                random_state=params.seed + current_index,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
            )
            exercise = intrinsic[itm] > continuation
            itm_indices = np.flatnonzero(itm)
            exercise_indices_now = itm_indices[exercise]

            current_values[exercise_indices_now] = intrinsic[exercise_indices_now]
            exercise_time[exercise_indices_now] = current_index * params.dt

            diagnostics.append(
                {
                    "time_index": current_index,
                    "time": current_index * params.dt,
                    "n_itm": int(np.sum(itm)),
                    "n_exercised": int(np.sum(exercise)),
                    "model": model,
                }
            )

        values = current_values
        next_index = current_index

    initial_discount = exp(-params.risk_free_rate * next_index * params.dt)
    discounted_path_values = values * initial_discount

    return {
        "price": float(np.mean(discounted_path_values)),
        "standard_error": float(np.std(discounted_path_values, ddof=1) / sqrt(params.n_paths)),
        "discounted_path_values": discounted_path_values,
        "exercise_time": exercise_time,
        "diagnostics": diagnostics,
    }


def price_bermudan_put_lsm_neural(
    params: BermudanPutParams,
    paths: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 500,
) -> dict:
    return price_put_lsm_neural(
        params,
        paths,
        bermudan_exercise_indices(params),
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
    )


def bermudan_regression_snapshot(
    params: BermudanPutParams,
    paths: np.ndarray,
    target_index: int | None = None,
) -> dict:
    exercise_grid = bermudan_exercise_indices(params)
    if target_index is None:
        target_index = int(exercise_grid[-2])
    if target_index not in set(exercise_grid[:-1]):
        raise ValueError("target_index must be one of the non-maturity Bermudan exercise indices.")

    maturity_index = exercise_grid[-1]
    values = put_payoff(paths[:, maturity_index], params.strike)
    next_index = maturity_index

    for current_index in reversed(exercise_grid[:-1]):
        discount = exp(-params.risk_free_rate * (next_index - current_index) * params.dt)
        discounted_values = values * discount

        spot = paths[:, current_index]
        intrinsic = put_payoff(spot, params.strike)
        itm = intrinsic > 0.0

        if not np.any(itm):
            values = discounted_values
            next_index = current_index
            continue

        continuation, model, poly = fit_polynomial_continuation(
            spot[itm],
            discounted_values[itm],
            params.regression_degree,
        )

        if current_index == target_index:
            return {
                "time_index": current_index,
                "time": current_index * params.dt,
                "spot_itm": spot[itm],
                "discounted_values_itm": discounted_values[itm],
                "continuation_itm": continuation,
                "model": model,
                "poly": poly,
            }

        current_values = discounted_values.copy()
        exercise = intrinsic[itm] > continuation
        itm_indices = np.flatnonzero(itm)
        current_values[itm_indices[exercise]] = intrinsic[itm_indices[exercise]]
        values = current_values
        next_index = current_index

    raise ValueError("No snapshot was generated for the requested target_index.")
