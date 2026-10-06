"""Compare revenue per assigned user when users can place several orders.

Run from the checkout: PYTHONPATH=src python examples/practitioner_user_orders.py
Requires pandas. SciPy is already a core bootstrapx dependency.
"""

from __future__ import annotations

import numpy as np
from assigned_users_composite_metrics import build_tables, prepare_user_metrics
from scipy import stats

from bootstrapx import bootstrap_two_sample


def run(n_resamples: int = 999) -> dict[str, object]:
    assignments, orders = build_tables()
    users = prepare_user_metrics(assignments, orders)
    control = users.loc[users["variant"] == "control"]
    treatment = users.loc[users["variant"] == "treatment"]
    control_revenue = control["revenue"].to_numpy(dtype=float)
    treatment_revenue = treatment["revenue"].to_numpy(dtype=float)

    result = bootstrap_two_sample(
        control_revenue,
        treatment_revenue,
        np.mean,
        effect="difference",
        method="basic",
        control_unit_ids=control["user_id"],
        treatment_unit_ids=treatment["user_id"],
        metric_name="revenue per assigned user",
        effect_unit="currency per assigned user",
        n_resamples=n_resamples,
        random_state=42,
    )

    # The same estimand and independent-user design in SciPy. Both libraries
    # receive the *prepared user-level arrays*, never the raw order rows.
    reference = stats.bootstrap(
        (control_revenue, treatment_revenue),
        lambda c, t: float(np.mean(t) - np.mean(c)),
        vectorized=False,
        method="basic",
        n_resamples=n_resamples,
        random_state=42,
    )

    # This is a different estimand: mean revenue of observed orders. It omits
    # non-buyers and weights users by how many orders they placed.
    observed_orders = orders.merge(
        assignments[["user_id", "variant"]],
        on="user_id",
        how="left",
        validate="many_to_one",
    )
    order_means = observed_orders.groupby("variant")["revenue"].mean()
    order_weighted_difference = float(order_means["treatment"] - order_means["control"])

    return {
        "assignments": assignments,
        "orders": orders,
        "users": users,
        "bootstrapx": result,
        "scipy": reference,
        "order_weighted_difference": order_weighted_difference,
    }


if __name__ == "__main__":
    analysis = run()
    result = analysis["bootstrapx"]
    reference = analysis["scipy"]
    print(f"Assigned users: {len(analysis['users'])}; observed orders: {len(analysis['orders'])}")
    print(f"Revenue per assigned user, treatment - control: {result.estimate:+.3f}")
    print(
        f"bootstrapx 95% basic CI: "
        f"[{result.confidence_interval.low:+.3f}, {result.confidence_interval.high:+.3f}]"
    )
    print(
        f"SciPy 95% basic CI: "
        f"[{reference.confidence_interval.low:+.3f}, "
        f"{reference.confidence_interval.high:+.3f}]"
    )
    print(
        "Different question: mean revenue per observed order, treatment - control: "
        f"{analysis['order_weighted_difference']:+.3f}"
    )
