from __future__ import annotations

import pytest

from acrs.portfolio.execution import (
    PositionMark,
    apply_entry_cash,
    apply_exit_cash,
    compute_exit_fill_price,
    compute_mark_to_market_nav,
    compute_realized_trade_pnl,
    exit_slippage_bps,
)


def test_normal_exit_uses_two_bps() -> None:
    assert (
        exit_slippage_bps(
            "RANGE_MEAN_REVERSION"
        )
        == 2.0
    )

    assert (
        compute_exit_fill_price(
            100.0,
            "RANGE_MEAN_REVERSION",
        )
        == pytest.approx(
            99.98
        )
    )


def test_stop_exit_uses_five_bps() -> None:
    assert (
        exit_slippage_bps(
            "STOP"
        )
        == 5.0
    )

    assert (
        compute_exit_fill_price(
            100.0,
            "STOP",
        )
        == pytest.approx(
            99.95
        )
    )


def test_global_stress_uses_five_bps() -> None:
    assert (
        exit_slippage_bps(
            "GLOBAL_STRESS"
        )
        == 5.0
    )


def test_entry_cash_includes_fee() -> None:
    result = apply_entry_cash(
        cash=5000.0,
        quantity=10.0,
        fill_price=100.0,
    )

    assert (
        result.notional
        == pytest.approx(
            1000.0
        )
    )

    assert (
        result.fee
        == pytest.approx(
            1.0
        )
    )

    assert (
        result.cash_after
        == pytest.approx(
            3999.0
        )
    )


def test_exit_cash_includes_slippage_and_fee() -> None:
    result = apply_exit_cash(
        cash=1000.0,
        quantity=10.0,
        reference_price=110.0,
        exit_reason=(
            "RANGE_MEAN_REVERSION"
        ),
    )

    assert (
        result.fill_price
        == pytest.approx(
            109.978
        )
    )

    assert (
        result.gross_proceeds
        == pytest.approx(
            1099.78
        )
    )

    assert (
        result.fee
        == pytest.approx(
            1.09978
        )
    )

    assert (
        result.net_proceeds
        == pytest.approx(
            1098.68022
        )
    )

    assert (
        result.cash_after
        == pytest.approx(
            2098.68022
        )
    )


def test_realized_pnl_includes_both_fees() -> None:
    quantity = 10.0
    entry_price = 100.0
    entry_fee = 1.0

    exit_result = (
        apply_exit_cash(
            cash=0.0,
            quantity=quantity,
            reference_price=110.0,
            exit_reason=(
                "RANGE_MEAN_REVERSION"
            ),
        )
    )

    pnl = (
        compute_realized_trade_pnl(
            quantity=quantity,
            entry_fill_price=(
                entry_price
            ),
            exit_fill_price=(
                exit_result.fill_price
            ),
            entry_fee=entry_fee,
            exit_fee=(
                exit_result.fee
            ),
        )
    )

    assert pnl == pytest.approx(
        97.68022
    )


def test_mark_to_market_nav() -> None:
    nav = (
        compute_mark_to_market_nav(
            cash=1000.0,
            positions=[
                PositionMark(
                    quantity=2.0,
                    mark_price=100.0,
                ),
                PositionMark(
                    quantity=3.0,
                    mark_price=50.0,
                ),
            ],
        )
    )

    assert nav == pytest.approx(
        1350.0
    )
