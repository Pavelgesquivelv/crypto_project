from acrs.config import PROJECT_ROOT, load_settings, load_universe_config


def test_settings_load() -> None:
    settings = load_settings()

    assert settings.project.name == "ACRS"
    assert settings.project.specification_version == "1.0"
    assert settings.project.timezone == "UTC"

    assert settings.timeframes.canonical == "1h"
    assert settings.timeframes.signal == "4h"
    assert settings.timeframes.macro == "1d"


def test_research_dates_are_consistent() -> None:
    settings = load_settings()

    assert settings.research.economic_start < settings.research.development_end
    assert settings.research.development_end < settings.research.lockbox_start
    assert settings.research.lockbox_start < settings.research.lockbox_end


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()


def test_universe_config() -> None:
    universe = load_universe_config()

    assert universe["exchange"] == "BINANCE"
    assert universe["market"] == "SPOT"
    assert universe["quote_asset"] == "USDT"
    assert universe["selection"]["top_n"] == 20
    assert universe["eligibility"]["minimum_history_days"] == 180
    assert universe["eligibility"]["minimum_completeness"] == 0.99
