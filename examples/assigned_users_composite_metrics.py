"""Offline synthetic assigned-user/order workflow; development API, not 0.5.1.

Run from the checkout: PYTHONPATH=src python examples/assigned_users_composite_metrics.py
Requires the pandas extra. Revenue/user is the primary example metric; the
other outputs are exploratory/descriptive, not simultaneous inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bootstrapx import RatioOfSums, bootstrap_two_sample


def build_tables() -> tuple[Any, Any]:
    """Generate complete orders within a fixed window for all assigned users."""
    rng = np.random.default_rng(707)
    variants = np.array(["control"] * 240 + ["treatment"] * 280)
    rng.shuffle(variants)
    assignments = pd.DataFrame(
        {
            "user_id": np.arange(520),
            "variant": variants,
        }
    )
    counts = rng.poisson(np.where(assignments["variant"] == "control", 1.8, 2.1))
    users = np.repeat(assignments["user_id"].to_numpy(), counts)
    order_variants = assignments["variant"].to_numpy()[users]
    prices = rng.lognormal(np.where(order_variants == "control", 2.0, 2.08), 0.45)
    orders = pd.DataFrame({"order_id": np.arange(len(users)), "user_id": users, "revenue": prices})
    return assignments, orders


def prepare_user_metrics(assignments: Any, orders: Any) -> Any:
    """Keep all assigned users; fill zeros only for verified absence of orders.

    This example assumes complete order capture over the same fixed window in
    both variants, not telemetry loss, delayed events, refunds, or currency
    conversion. A production workflow must establish those policies separately.
    """
    if not {"user_id", "variant"}.issubset(assignments.columns):
        raise ValueError("Assignments require user_id and variant columns.")
    if not {"order_id", "user_id", "revenue"}.issubset(orders.columns):
        raise ValueError("Orders require order_id, user_id and revenue columns.")
    if {"revenue", "orders"}.intersection(assignments.columns):
        raise ValueError("Assignment columns must not conflict with derived revenue/orders.")
    if assignments[["user_id", "variant"]].isna().any().any():
        raise ValueError("Assignments must not contain missing IDs or variants.")
    if assignments["user_id"].duplicated().any():
        raise ValueError("Each assigned user must have exactly one assignment row.")
    if not assignments["variant"].isin(["control", "treatment"]).all():
        raise ValueError("The example requires control/treatment variants.")
    if orders[["order_id", "user_id", "revenue"]].isna().any().any():
        raise ValueError("Missing order fields must not be interpreted as no orders.")
    if orders["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs would double-count revenue and orders.")
    if not orders["user_id"].isin(assignments["user_id"]).all():
        raise ValueError("Orders contain users absent from the assignment table.")
    if not pd.api.types.is_numeric_dtype(orders["revenue"].dtype):
        raise ValueError("Order revenue must already be numeric, not numeric-looking strings.")
    revenue = orders["revenue"].to_numpy(dtype=float)
    if not np.isfinite(revenue).all() or (revenue < 0).any():
        raise ValueError("This example requires finite non-negative order revenue.")
    aggregates = orders.groupby("user_id").agg(
        revenue=("revenue", "sum"), orders=("order_id", "size")
    )
    users = assignments.merge(aggregates, on="user_id", how="left", validate="one_to_one")
    # No missing event values remain: NaNs here mean no matching captured order.
    users[["revenue", "orders"]] = users[["revenue", "orders"]].fillna(0)
    return users


def analyze_example(n_resamples: int = 499) -> dict[str, Any]:
    assignments, orders = build_tables()
    users = prepare_user_metrics(assignments, orders)
    control = users.loc[users["variant"] == "control"]
    treatment = users.loc[users["variant"] == "treatment"]
    metrics = {
        "revenue/user": (lambda sample: np.mean(sample[:, 0]), "currency/user", "bca"),
        "orders/user": (lambda sample: np.mean(sample[:, 1]), "orders/user", "bca"),
        # The 0.6 release study found materially poor finite-sample BCa coverage
        # for skewed/correlated ratios. Basic performed better in those tested
        # user-level cases; this is evidence for the example, not a guarantee.
        "revenue/order": (RatioOfSums(), "currency/order", "basic"),
    }
    results = {}
    for name, (metric, unit, method) in metrics.items():
        results[name] = bootstrap_two_sample(
            control[["revenue", "orders"]],
            treatment[["revenue", "orders"]],
            metric,
            allow_2d=True,
            control_unit_ids=control["user_id"],
            treatment_unit_ids=treatment["user_id"],
            metric_name=name,
            effect_unit=unit,
            method=method,
            n_resamples=n_resamples,
            random_state=42,
        )
    buyer_summaries = []
    for name, arm in (("control", control), ("treatment", treatment)):
        buyers = arm.loc[arm["orders"] > 0]
        buyer_summaries.append(
            {
                "variant": name,
                "assigned_users": len(arm),
                "buyers": len(buyers),
                "mean_buyer_revenue_per_order": np.mean(buyers["revenue"] / buyers["orders"]),
            }
        )
    return {
        "assignments": assignments,
        "orders": orders,
        "users": users,
        "results": results,
        "summary": pd.DataFrame([{"metric": name, **r.to_dict()} for name, r in results.items()]),
        "buyer_descriptions": pd.DataFrame(buyer_summaries),
    }


if __name__ == "__main__":
    analysis = analyze_example()
    print(
        analysis["summary"][
            ["metric", "control_estimate", "treatment_estimate", "estimate", "ci_low", "ci_high"]
        ].to_string(index=False)
    )
    print("\nBuyer-only descriptions (post-treatment subsets, not an assigned-user causal effect):")
    print(analysis["buyer_descriptions"].to_string(index=False))
    print("\nSeparate pointwise intervals; no multiplicity control or automatic shipping decision.")
