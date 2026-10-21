from __future__ import annotations

import polars as pl
import pytest

from acrs.portfolio.policy import (
    PortfolioPolicyConfig,
    build_correlation_components,
    compute_drawdown_fraction,
    compute_open_stop_risk,
    drawdown_risk_multiplier,
    size_candidate,
)


def test_initial_nav_base_risk_is_25() -> None:
    result = size_candidate(
        nav=5000.0,
        cash=5000.0,
        drawdown_multiplier=1.0,
        current_gross_notional=0.0,
        current_aggregate_open_risk=0.0,
        current_cluster_notional=0.0,
        current_cluster_open_risk=0.0,
        entry_price=100.0,
        initial_stop_price=95.0,
        liquidity_notional_cap=10_000.0,
    )

    assert result.accepted

    assert (
        result.target_risk_budget
        == pytest.approx(25.0)
    )

    assert (
        result.risk_quantity_cap
        == pytest.approx(5.0)
    )


def test_asset_cap_limits_initial_position() -> None:
    result = size_candidate(
        nav=5000.0,
        cash=5000.0,
        drawdown_multiplier=1.0,
        current_gross_notional=0.0,
        current_aggregate_open_risk=0.0,
        current_cluster_notional=0.0,
        current_cluster_open_risk=0.0,
        entry_price=100.0,
        initial_stop_price=99.0,
        liquidity_notional_cap=100_000.0,
    )

    assert result.accepted

    assert (
        result.notional
        == pytest.approx(1000.0)
    )

    assert (
        result.binding_constraint
        == "ASSET_NOTIONAL"
    )


def test_liquidity_cap_can_size_trade_down() -> None:
    result = size_candidate(
        nav=5000.0,
        cash=5000.0,
        drawdown_multiplier=1.0,
        current_gross_notional=0.0,
        current_aggregate_open_risk=0.0,
        current_cluster_notional=0.0,
        current_cluster_open_risk=0.0,
        entry_price=100.0,
        initial_stop_price=95.0,
        liquidity_notional_cap=250.0,
    )

    assert result.accepted

    assert (
        result.notional
        == pytest.approx(250.0)
    )

    assert (
        result.binding_constraint
        == "LIQUIDITY"
    )


def test_cluster_risk_cap_is_enforced() -> None:
    result = size_candidate(
        nav=5000.0,
        cash=5000.0,
        drawdown_multiplier=1.0,
        current_gross_notional=0.0,
        current_aggregate_open_risk=0.0,
        current_cluster_notional=0.0,
        current_cluster_open_risk=70.0,
        entry_price=100.0,
        initial_stop_price=95.0,
        liquidity_notional_cap=10_000.0,
    )

    assert result.accepted

    assert (
        result.stop_risk
        == pytest.approx(5.0)
    )

    assert (
        result.binding_constraint
        == "CLUSTER_RISK"
    )


@pytest.mark.parametrize(
    ("drawdown", "expected"),
    [
        (0.00, 1.00),
        (0.0499, 1.00),
        (0.05, 0.75),
        (0.0999, 0.75),
        (0.10, 0.50),
        (0.1499, 0.50),
        (0.15, 0.25),
        (0.1999, 0.25),
        (0.20, 0.00),
        (0.30, 0.00),
    ],
)
def test_drawdown_multiplier_boundaries(
    drawdown: float,
    expected: float,
) -> None:
    assert (
        drawdown_risk_multiplier(
            drawdown
        )
        == expected
    )


def test_drawdown_fraction() -> None:
    assert (
        compute_drawdown_fraction(
            nav=4500.0,
            peak_nav=5000.0,
        )
        == pytest.approx(0.10)
    )


def test_open_stop_risk_uses_current_mark() -> None:
    risk = compute_open_stop_risk(
        quantity=2.0,
        mark_price=110.0,
        active_stop_price=100.0,
    )

    assert risk == pytest.approx(
        20.0
    )


def test_stop_above_mark_has_zero_open_risk() -> None:
    risk = compute_open_stop_risk(
        quantity=2.0,
        mark_price=100.0,
        active_stop_price=105.0,
    )

    assert risk == 0.0


def test_transitive_correlation_cluster() -> None:
    pairs = pl.DataFrame(
        {
            "left_symbol": [
                "AAA",
                "AAA",
                "BBB",
            ],
            "right_symbol": [
                "BBB",
                "CCC",
                "CCC",
            ],
            "correlation": [
                0.80,
                0.60,
                0.82,
            ],
            "pair_usable": [
                True,
                True,
                True,
            ],
        }
    )

    result = (
        build_correlation_components(
            [
                "AAA",
                "BBB",
                "CCC",
            ],
            pairs,
        )
    )

    assert result.usable

    assert result.components == (
        (
            "AAA",
            "BBB",
            "CCC",
        ),
    )


def test_unusable_correlation_blocks_cluster() -> None:
    pairs = pl.DataFrame(
        {
            "left_symbol": [
                "AAA",
            ],
            "right_symbol": [
                "BBB",
            ],
            "correlation": [
                None,
            ],
            "pair_usable": [
                False,
            ],
        }
    )

    result = (
        build_correlation_components(
            [
                "AAA",
                "BBB",
            ],
            pairs,
        )
    )

    assert not result.usable

    assert (
        result.unusable_reason
        == "UNUSABLE_PAIR"
    )


def test_cash_cap_includes_entry_fee() -> None:
    config = PortfolioPolicyConfig(
        asset_notional_fraction=1.0,
        gross_notional_fraction=1.0,
        cluster_notional_fraction=1.0,
        aggregate_risk_fraction=1.0,
        cluster_risk_fraction=1.0,
    )

    result = size_candidate(
        nav=5000.0,
        cash=100.10,
        drawdown_multiplier=1.0,
        current_gross_notional=0.0,
        current_aggregate_open_risk=0.0,
        current_cluster_notional=0.0,
        current_cluster_open_risk=0.0,
        entry_price=100.0,
        initial_stop_price=99.99,
        liquidity_notional_cap=100_000.0,
        config=config,
    )

    assert result.accepted

    assert (
        result.notional
        == pytest.approx(100.0)
    )

    assert (
        result.binding_constraint
        == "CASH"
    )
