from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from acrs.portfolio.policy import (
    PortfolioPolicyConfig,
)

EXECUTION_ACCOUNTING_VERSION = (
    "5B2-execution-accounting-v1"
)


@dataclass(frozen=True)
class PositionMark:
    quantity: float
    mark_price: float


@dataclass(frozen=True)
class EntryCashResult:
    cash_before: float
    quantity: float
    fill_price: float
    notional: float
    fee: float
    cash_after: float


@dataclass(frozen=True)
class ExitCashResult:
    cash_before: float
    quantity: float
    reference_price: float
    slippage_bps: float
    fill_price: float
    gross_proceeds: float
    fee: float
    net_proceeds: float
    cash_after: float


def _require_finite(
    name: str,
    value: float,
) -> None:
    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )


def exit_slippage_bps(
    exit_reason: str,
    config: (
        PortfolioPolicyConfig
        | None
    ) = None,
) -> float:
    config = (
        config
        or PortfolioPolicyConfig()
    )

    if exit_reason in {
        "STOP",
        "GLOBAL_STRESS",
    }:
        return (
            config
            .stop_stress_exit_slippage_bps
        )

    return (
        config
        .normal_exit_slippage_bps
    )


def compute_exit_fill_price(
    reference_price: float,
    exit_reason: str,
    config: (
        PortfolioPolicyConfig
        | None
    ) = None,
) -> float:
    config = (
        config
        or PortfolioPolicyConfig()
    )

    _require_finite(
        "reference_price",
        reference_price,
    )

    if reference_price <= 0.0:
        raise ValueError(
            "reference_price must "
            "be positive."
        )

    slippage_bps = (
        exit_slippage_bps(
            exit_reason,
            config,
        )
    )

    fill_price = (
        reference_price
        * (
            1.0
            - slippage_bps
            / 10_000.0
        )
    )

    if fill_price <= 0.0:
        raise ValueError(
            "Computed exit fill price "
            "must be positive."
        )

    return fill_price


def apply_entry_cash(
    *,
    cash: float,
    quantity: float,
    fill_price: float,
    config: (
        PortfolioPolicyConfig
        | None
    ) = None,
) -> EntryCashResult:
    config = (
        config
        or PortfolioPolicyConfig()
    )

    for name, value in {
        "cash": cash,
        "quantity": quantity,
        "fill_price": fill_price,
    }.items():
        _require_finite(
            name,
            value,
        )

    if cash < 0.0:
        raise ValueError(
            "cash cannot be negative."
        )

    if quantity <= 0.0:
        raise ValueError(
            "quantity must be positive."
        )

    if fill_price <= 0.0:
        raise ValueError(
            "fill_price must be positive."
        )

    notional = (
        quantity
        * fill_price
    )

    fee = (
        notional
        * config.entry_fee_bps
        / 10_000.0
    )

    total_debit = (
        notional
        + fee
    )

    cash_after = (
        cash
        - total_debit
    )

    tolerance = 1e-10

    if cash_after < -tolerance:
        raise ValueError(
            "Insufficient cash for "
            "entry notional plus fee."
        )

    if cash_after < 0.0:
        cash_after = 0.0

    return EntryCashResult(
        cash_before=cash,
        quantity=quantity,
        fill_price=fill_price,
        notional=notional,
        fee=fee,
        cash_after=cash_after,
    )


def apply_exit_cash(
    *,
    cash: float,
    quantity: float,
    reference_price: float,
    exit_reason: str,
    config: (
        PortfolioPolicyConfig
        | None
    ) = None,
) -> ExitCashResult:
    config = (
        config
        or PortfolioPolicyConfig()
    )

    for name, value in {
        "cash": cash,
        "quantity": quantity,
        "reference_price": (
            reference_price
        ),
    }.items():
        _require_finite(
            name,
            value,
        )

    if cash < 0.0:
        raise ValueError(
            "cash cannot be negative."
        )

    if quantity <= 0.0:
        raise ValueError(
            "quantity must be positive."
        )

    if reference_price <= 0.0:
        raise ValueError(
            "reference_price must "
            "be positive."
        )

    slippage_bps = (
        exit_slippage_bps(
            exit_reason,
            config,
        )
    )

    fill_price = (
        compute_exit_fill_price(
            reference_price,
            exit_reason,
            config,
        )
    )

    gross_proceeds = (
        quantity
        * fill_price
    )

    fee = (
        gross_proceeds
        * config.exit_fee_bps
        / 10_000.0
    )

    net_proceeds = (
        gross_proceeds
        - fee
    )

    cash_after = (
        cash
        + net_proceeds
    )

    return ExitCashResult(
        cash_before=cash,
        quantity=quantity,
        reference_price=(
            reference_price
        ),
        slippage_bps=(
            slippage_bps
        ),
        fill_price=fill_price,
        gross_proceeds=(
            gross_proceeds
        ),
        fee=fee,
        net_proceeds=(
            net_proceeds
        ),
        cash_after=cash_after,
    )


def compute_mark_to_market_nav(
    *,
    cash: float,
    positions: Iterable[
        PositionMark
    ],
) -> float:
    _require_finite(
        "cash",
        cash,
    )

    if cash < 0.0:
        raise ValueError(
            "cash cannot be negative."
        )

    market_value = 0.0

    for position in positions:
        _require_finite(
            "quantity",
            position.quantity,
        )

        _require_finite(
            "mark_price",
            position.mark_price,
        )

        if position.quantity < 0.0:
            raise ValueError(
                "Position quantity "
                "cannot be negative."
            )

        if position.mark_price <= 0.0:
            raise ValueError(
                "Position mark price "
                "must be positive."
            )

        market_value += (
            position.quantity
            * position.mark_price
        )

    return (
        cash
        + market_value
    )


def compute_realized_trade_pnl(
    *,
    quantity: float,
    entry_fill_price: float,
    exit_fill_price: float,
    entry_fee: float,
    exit_fee: float,
) -> float:
    for name, value in {
        "quantity": quantity,
        "entry_fill_price": (
            entry_fill_price
        ),
        "exit_fill_price": (
            exit_fill_price
        ),
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
    }.items():
        _require_finite(
            name,
            value,
        )

    if quantity <= 0.0:
        raise ValueError(
            "quantity must be positive."
        )

    if (
        entry_fill_price <= 0.0
        or exit_fill_price <= 0.0
    ):
        raise ValueError(
            "Fill prices must be positive."
        )

    if (
        entry_fee < 0.0
        or exit_fee < 0.0
    ):
        raise ValueError(
            "Fees cannot be negative."
        )

    gross_pnl = (
        quantity
        * (
            exit_fill_price
            - entry_fill_price
        )
    )

    return (
        gross_pnl
        - entry_fee
        - exit_fee
    )
