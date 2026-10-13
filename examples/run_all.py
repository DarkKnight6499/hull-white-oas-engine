"""Run every synthetic example and write reviewable tables without network access."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw_oas.calibration import calibrate, synthetic_swaption_quotes
from hw_oas.curves import DiscountCurve, bootstrap_swaps, par_swap_rate, synthetic_curve
from hw_oas.instruments import FixedRateBond
from hw_oas.mbs import MBSEngine, static_psa_price
from hw_oas.model import HullWhite
from hw_oas.oas import effective_risk, key_rate_durations, solve_oas, z_spread
from hw_oas.prepayment import MortgagePool, RefinanceCPR
from hw_oas.tree import HullWhiteTree

OUTPUT = ROOT / "examples" / "output"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render Markdown without the optional tabulate dependency."""
    def cell(value: object) -> str:
        return f"{value:.8f}" if isinstance(value, (float, np.floating)) else str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join(":-" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows]) + "\n"


def save_table(name: str, frame: pd.DataFrame) -> str:
    """Write matching CSV and Markdown tables."""
    frame.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.12g")
    table = markdown_table(frame)
    (OUTPUT / f"{name}.md").write_text(table, encoding="utf-8")
    return table


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curve = synthetic_curve()
    times = curve.times[1:]
    save_table("synthetic_curve", pd.DataFrame({"time": times, "zero_rate": curve.zero_rate(times),
                                               "discount": curve.discount(times)}))
    swap_rates = np.array([par_swap_rate(curve, t) for t in range(1, 31)])
    bootstrapped = bootstrap_swaps(swap_rates)
    repriced = np.array([par_swap_rate(bootstrapped, t) for t in range(1, 31)])
    save_table("swap_bootstrap", pd.DataFrame({"tenor": np.arange(1, 31), "input_rate": swap_rates,
                                              "repriced_rate": repriced, "error": repriced - swap_rates}))
    quotes = synthetic_swaption_quotes(curve)
    fitted = calibrate(curve, quotes)
    if not fitted.converged:
        raise RuntimeError("Synthetic calibration failed to converge")
    model = fitted.model
    calibration = pd.DataFrame([{"expiry": q.expiry, "tenor": q.tenor,
                                  "normal_vol_bp": q.normal_vol * 10000,
                                  "target_price": q.price,
                                  "model_price": model.swaption(q.expiry, q.tenor),
                                  "price_error": error}
                                 for q, error in zip(quotes, fitted.residuals)])
    calibration_table = save_table("calibration", calibration)
    exact_fit = calibrate(curve, synthetic_swaption_quotes(curve, noise=0))

    expiry, maturity = 2.0, 10.0
    strike = curve.discount(maturity) / curve.discount(expiry)
    analytic = model.zero_bond_option(expiry, maturity, strike)
    convergence = []
    curve_errors = []
    for steps in (40, 80, 160, 320, 640):
        tree = HullWhiteTree(model, maturity, steps)
        price = tree.coupon_bond_option(np.array([maturity]), np.array([1.0]), strike, [expiry])
        curve_errors.extend(abs(tree.zero_bond(t) / curve.discount(t) - 1) for t in (2.0, 5.0, 10.0))
        convergence.append({"steps": steps, "tree_price": price, "analytic_price": analytic,
                            "absolute_error": abs(price - analytic)})
    convergence_table = save_table("tree_convergence", pd.DataFrame(convergence))

    bond = FixedRateBond(10, 0.045, calls={t: 100.0 for t in np.arange(2, 10, 0.5)})
    steps = 400
    tree = HullWhiteTree(model, bond.maturity, steps)
    true_oas = 0.0025
    market = tree.bond_price(bond, true_oas)
    solved = solve_oas(lambda s: tree.bond_price(bond, s), market)
    z = z_spread(bond, curve, market)
    def bond_pricer(bumped_curve: DiscountCurve, spread: float) -> float:
        return HullWhiteTree(HullWhite(bumped_curve, model.a, model.sigma), bond.maturity, steps).bond_price(bond, spread)
    risk = effective_risk(bond_pricer, curve, solved)
    key_rates = key_rate_durations(bond_pricer, curve, solved, np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0]))
    save_table("key_rate_durations", pd.DataFrame({"key_years": list(key_rates), "duration": list(key_rates.values())}))
    callable_metrics = {"straight_price_at_oas": bond.straight_price(curve, solved),
                        "callable_market_price": market, "callable_oas_bp": solved * 10000,
                        "z_spread_bp": z * 10000, "effective_duration": risk.duration,
                        "effective_convexity": risk.convexity, "down_price": risk.down_price,
                        "up_price": risk.up_price}
    callable_table = save_table("callable_bond", pd.DataFrame(callable_metrics.items(), columns=["metric", "value"]))

    pool = MortgagePool()
    prepayment = RefinanceCPR()
    reference = MBSEngine(model, pool, paths=8192, prepayment=prepayment)
    mbs_market = reference.estimate(0.003).price
    mc_rows = []
    for paths in (1024, 4096, 8192):
        engine = reference if paths == 8192 else MBSEngine(model, pool, paths=paths, prepayment=prepayment)
        solution = engine.solve_oas(mbs_market)
        estimate = engine.estimate(solution.oas)
        reference_estimate = engine.estimate(0.003)
        mc_rows.append({"paths": paths, "independent_pairs": estimate.independent_samples,
                        "price_at_30bp": reference_estimate.price,
                        "price_standard_error": reference_estimate.standard_error,
                        "oas_bp": solution.oas * 10000,
                        "oas_standard_error_bp": solution.oas_standard_error * 10000,
                        "wal_years": solution.wal})
    mc_table = save_table("mbs_convergence", pd.DataFrame(mc_rows))
    def mbs_pricer(bumped_curve: DiscountCurve, spread: float) -> float:
        engine = MBSEngine(HullWhite(bumped_curve, model.a, model.sigma), pool,
                           paths=8192, prepayment=prepayment)
        return engine.estimate(spread).price
    mbs_risk = effective_risk(mbs_pricer, curve, 0.003)
    def psa_pricer(bumped_curve: DiscountCurve, spread: float) -> float:
        return static_psa_price(pool, bumped_curve, spread)[0]
    psa_risk = effective_risk(psa_pricer, curve, 0.003)
    psa_price, psa_wal = static_psa_price(pool, curve, 0.003)
    comparison = pd.DataFrame([
        {"method": "Rate-dependent refinance MC", "price_at_30bp": mbs_market,
         "standard_error": reference.estimate(0.003).standard_error,
         "duration": mbs_risk.duration, "convexity": mbs_risk.convexity,
         "wal_years": reference.estimate(0.003).wal},
        {"method": "Static PSA 100", "price_at_30bp": psa_price, "standard_error": 0.0,
         "duration": psa_risk.duration, "convexity": psa_risk.convexity, "wal_years": psa_wal}])
    comparison_table = save_table("mbs_comparison", comparison)
    configuration = {
        "data_kind": "synthetic", "curve_seed": 1729, "vol_seed": 2718,
        "simulation_seed": 31415, "reference_a": 0.075, "reference_sigma": 0.012,
        "quote_log_vol_noise": 0.025, "bond_maturity_years": bond.maturity,
        "bond_coupon": bond.coupon, "bond_face": bond.face, "bond_frequency": bond.frequency,
        "first_call_years": min(bond.calls), "call_redemption": bond.calls[min(bond.calls)],
        "bond_tree_steps": steps, "bond_oas": true_oas, "risk_bump": 0.001,
        "pool_balance": pool.balance, "pool_gross_coupon": pool.coupon,
        "pool_servicing_fee": pool.servicing_fee, "pool_guarantee_fee": pool.guarantee_fee,
        "pool_net_coupon": pool.net_coupon, "pool_months": pool.term_months,
        "mbs_reference_paths": 8192, "mbs_reference_oas": 0.003,
        "base_cpr": prepayment.base_cpr, "incentive_cpr": prepayment.incentive_cpr,
        "incentive_scale": prepayment.incentive_scale, "seasoning_months": prepayment.seasoning_months,
        "burnout": prepayment.burnout, "mortgage_proxy_spread": 0.015,
        "mortgage_proxy_loading": 1.0, "option_expiry": expiry, "option_maturity": maturity,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__}
    (OUTPUT / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    save_table("static_psa_cashflows", pool.static_cashflows())
    metrics = {"a": model.a, "sigma": model.sigma, "calibration_rmse": fitted.rmse,
               "calibration_relative_rmse": fitted.relative_rmse, "calibration_iterations": fitted.iterations,
               "roundtrip_a": exact_fit.model.a, "roundtrip_sigma": exact_fit.model.sigma,
               "roundtrip_relative_rmse": exact_fit.relative_rmse,
               "swap_repricing_max_error": float(np.max(np.abs(repriced - swap_rates))),
               "tree_curve_max_relative_error": max(curve_errors),
               "analytic_option_price": analytic, **callable_metrics,
               "mbs_market_price": mbs_market, "mbs_price_se": reference.estimate(0.003).standard_error,
               "mbs_duration": mbs_risk.duration, "mbs_convexity": mbs_risk.convexity,
               "mbs_wal": reference.estimate(0.003).wal,
               "static_psa_price": psa_price, "static_psa_wal": psa_wal}
    (OUTPUT / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = ("# Synthetic validation report\n\nAll prices and targets use seeded synthetic inputs.\n\n"
              + "## Calibration\n\n" + calibration_table + "\n## Tree convergence\n\n" + convergence_table
              + "\n## Callable bond\n\n" + callable_table + "\n## MBS convergence\n\n" + mc_table
              + "\n## MBS comparison at constant spread\n\n" + comparison_table)
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
