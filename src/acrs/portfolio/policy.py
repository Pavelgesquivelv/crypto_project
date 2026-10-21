from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

PORTFOLIO_POLICY_VERSION = (
    "5B1-portfolio-policy-v1"
)


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    initial_nav: float = 5000.0

    base_risk_fraction: float = 0.005
    aggregate_risk_fraction: float = 0.03

    max_positions: int = 6

    asset_notional_fraction: float = 0.20
    gross_notional_fraction: float = 0.95

    cluster_correlation_threshold: float = 0.75
    cluster_risk_fraction: float = 0.015
    cluster_notional_fraction: float = 0.50

    daily_pause_loss_fraction: float = 0.03

    hard_halt_drawdown_fraction: float = 0.20

    entry_fee_bps: float = 10.0
    exit_fee_bps: float = 10.0

    normal_exit_slippage_bps: float = 2.0
    stop_stress_exit_slippage_bps: float = 5.0


@dataclass(frozen=True)
class SizingResult:
    accepted: bool

    quantity: float
    notional: float
    stop_risk: float

    target_risk_budget: float

    risk_quantity_cap: float
    aggregate_risk_quantity_cap: float
    asset_quantity_cap: float
    gross_quantity_cap: float
    cash_quantity_cap: float
    liquidity_quantity_cap: float
    cluster_risk_quantity_cap: float
    cluster_notional_quantity_cap: float

    binding_constraint: str | None
    reject_reason: str | None


@dataclass(frozen=True)
class ClusterResult:
    usable: bool
    components: tuple[
        tuple[str, ...],
        ...,
    ]
    unusable_reason: str | None


def validate_policy_config(
    config: PortfolioPolicyConfig,
) -> None:
    if config.initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    fractions = {
        "base_risk_fraction": (
            config.base_risk_fraction
        ),
        "aggregate_risk_fraction": (
            config.aggregate_risk_fraction
        ),
        "asset_notional_fraction": (
            config.asset_notional_fraction
        ),
        "gross_notional_fraction": (
            config.gross_notional_fraction
        ),
        "cluster_risk_fraction": (
            config.cluster_risk_fraction
        ),
        "cluster_notional_fraction": (
            config.cluster_notional_fraction
        ),
        "daily_pause_loss_fraction": (
            config.daily_pause_loss_fraction
        ),
        "hard_halt_drawdown_fraction": (
            config.hard_halt_drawdown_fraction
        ),
    }

    for name, value in fractions.items():
        if not (
            0.0
            < value
            <= 1.0
        ):
            raise ValueError(
                f"{name} must be in (0, 1]."
            )

    if config.max_positions < 1:
        raise ValueError(
            "max_positions must be positive."
        )

    if not (
        -1.0
        <= config.cluster_correlation_threshold
        <= 1.0
    ):
        raise ValueError(
            "cluster_correlation_threshold "
            "must be in [-1, 1]."
        )

    costs = {
        "entry_fee_bps": (
            config.entry_fee_bps
        ),
        "exit_fee_bps": (
            config.exit_fee_bps
        ),
        "normal_exit_slippage_bps": (
            config.normal_exit_slippage_bps
        ),
        "stop_stress_exit_slippage_bps": (
            config
            .stop_stress_exit_slippage_bps
        ),
    }

    for name, value in costs.items():
        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative."
            )


def drawdown_risk_multiplier(
    drawdown_fraction: float,
) -> float:
    if drawdown_fraction < 0.0:
        raise ValueError(
            "drawdown_fraction cannot "
            "be negative."
        )

    if drawdown_fraction < 0.05:
        return 1.00

    if drawdown_fraction < 0.10:
        return 0.75

    if drawdown_fraction < 0.15:
        return 0.50

    if drawdown_fraction < 0.20:
        return 0.25

    return 0.0


def compute_drawdown_fraction(
    nav: float,
    peak_nav: float,
) -> float:
    if nav < 0.0:
        raise ValueError(
            "nav cannot be negative."
        )

    if peak_nav <= 0.0:
        raise ValueError(
            "peak_nav must be positive."
        )

    return max(
        0.0,
        1.0
        - (
            nav
            / peak_nav
        ),
    )


def compute_open_stop_risk(
    *,
    quantity: float,
    mark_price: float,
    active_stop_price: float,
) -> float:
    if quantity < 0.0:
        raise ValueError(
            "quantity cannot be negative."
        )

    if mark_price <= 0.0:
        raise ValueError(
            "mark_price must be positive."
        )

    if active_stop_price <= 0.0:
        raise ValueError(
            "active_stop_price must "
            "be positive."
        )

    return (
        quantity
        * max(
            mark_price
            - active_stop_price,
            0.0,
        )
    )


def size_candidate(
    *,
    nav: float,
    cash: float,
    drawdown_multiplier: float,
    current_gross_notional: float,
    current_aggregate_open_risk: float,
    current_cluster_notional: float,
    current_cluster_open_risk: float,
    entry_price: float,
    initial_stop_price: float,
    liquidity_notional_cap: float,
    config: (
        PortfolioPolicyConfig
        | None
    ) = None,
) -> SizingResult:
    config = (
        config
        or PortfolioPolicyConfig()
    )

    validate_policy_config(
        config
    )

    if nav <= 0.0:
        return SizingResult(
            accepted=False,
            quantity=0.0,
            notional=0.0,
            stop_risk=0.0,
            target_risk_budget=0.0,
            risk_quantity_cap=0.0,
            aggregate_risk_quantity_cap=0.0,
            asset_quantity_cap=0.0,
            gross_quantity_cap=0.0,
            cash_quantity_cap=0.0,
            liquidity_quantity_cap=0.0,
            cluster_risk_quantity_cap=0.0,
            cluster_notional_quantity_cap=0.0,
            binding_constraint=None,
            reject_reason="NONPOSITIVE_NAV",
        )

    if cash <= 0.0:
        return SizingResult(
            accepted=False,
            quantity=0.0,
            notional=0.0,
            stop_risk=0.0,
            target_risk_budget=0.0,
            risk_quantity_cap=0.0,
            aggregate_risk_quantity_cap=0.0,
            asset_quantity_cap=0.0,
            gross_quantity_cap=0.0,
            cash_quantity_cap=0.0,
            liquidity_quantity_cap=0.0,
            cluster_risk_quantity_cap=0.0,
            cluster_notional_quantity_cap=0.0,
            binding_constraint=None,
            reject_reason="NO_CASH",
        )

    if not (
        0.0
        <= drawdown_multiplier
        <= 1.0
    ):
        raise ValueError(
            "drawdown_multiplier must "
            "be in [0, 1]."
        )

    if entry_price <= 0.0:
        raise ValueError(
            "entry_price must be positive."
        )

    if (
        initial_stop_price <= 0.0
        or initial_stop_price
        >= entry_price
    ):
        raise ValueError(
            "initial stop must be "
            "strictly between zero and "
            "entry price."
        )

    if liquidity_notional_cap <= 0.0:
        return SizingResult(
            accepted=False,
            quantity=0.0,
            notional=0.0,
            stop_risk=0.0,
            target_risk_budget=0.0,
            risk_quantity_cap=0.0,
            aggregate_risk_quantity_cap=0.0,
            asset_quantity_cap=0.0,
            gross_quantity_cap=0.0,
            cash_quantity_cap=0.0,
            liquidity_quantity_cap=0.0,
            cluster_risk_quantity_cap=0.0,
            cluster_notional_quantity_cap=0.0,
            binding_constraint=None,
            reject_reason="LIQUIDITY_UNUSABLE",
        )

    stop_distance = (
        entry_price
        - initial_stop_price
    )

    target_risk_budget = (
        nav
        * config.base_risk_fraction
        * drawdown_multiplier
    )

    if target_risk_budget <= 0.0:
        return SizingResult(
            accepted=False,
            quantity=0.0,
            notional=0.0,
            stop_risk=0.0,
            target_risk_budget=(
                target_risk_budget
            ),
            risk_quantity_cap=0.0,
            aggregate_risk_quantity_cap=0.0,
            asset_quantity_cap=0.0,
            gross_quantity_cap=0.0,
            cash_quantity_cap=0.0,
            liquidity_quantity_cap=0.0,
            cluster_risk_quantity_cap=0.0,
            cluster_notional_quantity_cap=0.0,
            binding_constraint=None,
            reject_reason="ZERO_RISK_BUDGET",
        )

    risk_quantity_cap = (
        target_risk_budget
        / stop_distance
    )

    aggregate_risk_remaining = max(
        nav
        * config.aggregate_risk_fraction
        - current_aggregate_open_risk,
        0.0,
    )

    aggregate_risk_quantity_cap = (
        aggregate_risk_remaining
        / stop_distance
    )

    asset_quantity_cap = (
        nav
        * config.asset_notional_fraction
        / entry_price
    )

    gross_remaining = max(
        nav
        * config.gross_notional_fraction
        - current_gross_notional,
        0.0,
    )

    gross_quantity_cap = (
        gross_remaining
        / entry_price
    )

    entry_fee_fraction = (
        config.entry_fee_bps
        / 10_000.0
    )

    cash_quantity_cap = (
        cash
        / (
            entry_price
            * (
                1.0
                + entry_fee_fraction
            )
        )
    )

    liquidity_quantity_cap = (
        liquidity_notional_cap
        / entry_price
    )

    cluster_risk_remaining = max(
        nav
        * config.cluster_risk_fraction
        - current_cluster_open_risk,
        0.0,
    )

    cluster_risk_quantity_cap = (
        cluster_risk_remaining
        / stop_distance
    )

    cluster_notional_remaining = max(
        nav
        * config.cluster_notional_fraction
        - current_cluster_notional,
        0.0,
    )

    cluster_notional_quantity_cap = (
        cluster_notional_remaining
        / entry_price
    )

    caps = {
        "BASE_RISK": (
            risk_quantity_cap
        ),
        "AGGREGATE_RISK": (
            aggregate_risk_quantity_cap
        ),
        "ASSET_NOTIONAL": (
            asset_quantity_cap
        ),
        "GROSS_NOTIONAL": (
            gross_quantity_cap
        ),
        "CASH": (
            cash_quantity_cap
        ),
        "LIQUIDITY": (
            liquidity_quantity_cap
        ),
        "CLUSTER_RISK": (
            cluster_risk_quantity_cap
        ),
        "CLUSTER_NOTIONAL": (
            cluster_notional_quantity_cap
        ),
    }

    finite_caps = {
        key: value
        for key, value in caps.items()
        if math.isfinite(value)
    }

    if (
        len(finite_caps)
        != len(caps)
    ):
        raise ValueError(
            "Non-finite sizing cap."
        )

    quantity = min(
        finite_caps.values()
    )

    binding_constraint = min(
        finite_caps,
        key=lambda name: finite_caps[name],
    )

    if quantity <= 0.0:
        return SizingResult(
            accepted=False,
            quantity=0.0,
            notional=0.0,
            stop_risk=0.0,
            target_risk_budget=(
                target_risk_budget
            ),
            risk_quantity_cap=(
                risk_quantity_cap
            ),
            aggregate_risk_quantity_cap=(
                aggregate_risk_quantity_cap
            ),
            asset_quantity_cap=(
                asset_quantity_cap
            ),
            gross_quantity_cap=(
                gross_quantity_cap
            ),
            cash_quantity_cap=(
                cash_quantity_cap
            ),
            liquidity_quantity_cap=(
                liquidity_quantity_cap
            ),
            cluster_risk_quantity_cap=(
                cluster_risk_quantity_cap
            ),
            cluster_notional_quantity_cap=(
                cluster_notional_quantity_cap
            ),
            binding_constraint=(
                binding_constraint
            ),
            reject_reason=(
                f"{binding_constraint}_EXHAUSTED"
            ),
        )

    notional = (
        quantity
        * entry_price
    )

    stop_risk = (
        quantity
        * stop_distance
    )

    return SizingResult(
        accepted=True,
        quantity=quantity,
        notional=notional,
        stop_risk=stop_risk,
        target_risk_budget=(
            target_risk_budget
        ),
        risk_quantity_cap=(
            risk_quantity_cap
        ),
        aggregate_risk_quantity_cap=(
            aggregate_risk_quantity_cap
        ),
        asset_quantity_cap=(
            asset_quantity_cap
        ),
        gross_quantity_cap=(
            gross_quantity_cap
        ),
        cash_quantity_cap=(
            cash_quantity_cap
        ),
        liquidity_quantity_cap=(
            liquidity_quantity_cap
        ),
        cluster_risk_quantity_cap=(
            cluster_risk_quantity_cap
        ),
        cluster_notional_quantity_cap=(
            cluster_notional_quantity_cap
        ),
        binding_constraint=(
            binding_constraint
        ),
        reject_reason=None,
    )


def build_correlation_components(
    symbols: Iterable[str],
    pairwise: pl.DataFrame,
    *,
    threshold: float = 0.75,
) -> ClusterResult:
    unique_symbols = sorted(
        set(symbols)
    )

    if not (
        -1.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be in [-1, 1]."
        )

    if not unique_symbols:
        return ClusterResult(
            usable=True,
            components=(),
            unusable_reason=None,
        )

    if len(unique_symbols) == 1:
        return ClusterResult(
            usable=True,
            components=(
                (
                    unique_symbols[0],
                ),
            ),
            unusable_reason=None,
        )

    required = {
        "left_symbol",
        "right_symbol",
        "correlation",
        "pair_usable",
    }

    missing = (
        required
        - set(pairwise.columns)
    )

    if missing:
        raise ValueError(
            "Pairwise frame missing columns: "
            f"{sorted(missing)}"
        )

    lookup: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}

    for source_row in pairwise.iter_rows(
        named=True
    ):
        left = str(
            source_row[
                "left_symbol"
            ]
        )

        right = str(
            source_row[
                "right_symbol"
            ]
        )

        if left <= right:
            key: tuple[
                str,
                str,
            ] = (
                left,
                right,
            )
        else:
            key = (
                right,
                left,
            )

        if key in lookup:
            raise ValueError(
                "Duplicate correlation pair "
                f"{key}."
            )

        lookup[
            key
        ] = source_row

    adjacency: dict[
        str,
        set[str],
    ] = {
        symbol: set()
        for symbol in unique_symbols
    }

    for left_index in range(
        len(unique_symbols) - 1
    ):
        for right_index in range(
            left_index + 1,
            len(unique_symbols),
        ):
            left = unique_symbols[
                left_index
            ]

            right = unique_symbols[
                right_index
            ]

            pair_row = lookup.get(
                (
                    left,
                    right,
                )
            )

            if pair_row is None:
                return ClusterResult(
                    usable=False,
                    components=(),
                    unusable_reason=(
                        "MISSING_PAIR"
                    ),
                )

            if not bool(
                pair_row[
                    "pair_usable"
                ]
            ):
                return ClusterResult(
                    usable=False,
                    components=(),
                    unusable_reason=(
                        "UNUSABLE_PAIR"
                    ),
                )

            correlation_value = (
                pair_row[
                    "correlation"
                ]
            )

            if correlation_value is None:
                return ClusterResult(
                    usable=False,
                    components=(),
                    unusable_reason=(
                        "NULL_CORRELATION"
                    ),
                )

            if not isinstance(
                correlation_value,
                (
                    int,
                    float,
                ),
            ):
                raise ValueError(
                    "Correlation value must "
                    "be numeric."
                )

            correlation = float(
                correlation_value
            )

            if correlation >= threshold:
                adjacency[
                    left
                ].add(
                    right
                )

                adjacency[
                    right
                ].add(
                    left
                )

    visited: set[str] = set()

    components: list[
        tuple[str, ...]
    ] = []

    for symbol in unique_symbols:
        if symbol in visited:
            continue

        stack: list[str] = [
            symbol
        ]

        component: list[
            str
        ] = []

        while stack:
            current = (
                stack.pop()
            )

            if current in visited:
                continue

            visited.add(
                current
            )

            component.append(
                current
            )

            for neighbor in sorted(
                adjacency[
                    current
                ],
                reverse=True,
            ):
                if (
                    neighbor
                    not in visited
                ):
                    stack.append(
                        neighbor
                    )

        components.append(
            tuple(
                sorted(
                    component
                )
            )
        )

    components.sort()

    return ClusterResult(
        usable=True,
        components=tuple(
            components
        ),
        unusable_reason=None,
    )