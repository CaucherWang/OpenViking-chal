import pytest

from openviking_cli.utils.config.open_viking_config import OpenVikingConfig


def test_rule_cleanup_defaults_to_thirty_day_interval():
    config = OpenVikingConfig.from_dict({})

    assert config.rules.archived_retention_days == 30
    assert config.rules.archived_cleanup_interval_seconds == 30 * 24 * 60 * 60


def test_openviking_config_accepts_dedicated_rule_models():
    config = OpenVikingConfig.from_dict(
        {
            "rules": {
                "compiler_context_window": 32768,
                "archived_retention_days": 45,
                "archived_cleanup_interval_seconds": 3600,
                "archived_cleanup_enabled": False,
                "scenario_irrelevance_threshold": 0.95,
                "jev_rule_matcher": {
                    "provider": "jev",
                    "api_key": "jev-key",
                    "model": "jev-latest",
                    "mode": "noul",
                },
                "compiler": {
                    "provider": "litellm",
                    "model": "ollama/rule-compiler",
                },
            }
        }
    )

    assert config.rules.jev_rule_matcher is not None
    assert config.rules.jev_rule_matcher.model == "jev-latest"
    assert config.rules.compiler_context_window == 32768
    assert config.rules.archived_retention_days == 45
    assert config.rules.archived_cleanup_interval_seconds == 3600
    assert config.rules.archived_cleanup_enabled is False
    assert config.rules.scenario_irrelevance_threshold == 0.95
    compiler = config.rules.get_compiler()
    assert compiler is not None
    assert compiler.model == "ollama/rule-compiler"


def test_empty_rule_compiler_block_falls_back_to_default_vlm():
    config = OpenVikingConfig.from_dict(
        {
            "rules": {
                "compiler": {},
            }
        }
    )

    assert config.rules.get_compiler() is None


@pytest.mark.parametrize(
    "jev_config,expected",
    [
        (
            {"provider": "cohere", "api_key": "key"},
            "must use provider 'jev'",
        ),
        (
            {"provider": "jev", "api_key": "key", "mode": "choice"},
            "must use mode 'noul'",
        ),
    ],
)
def test_jev_rule_matcher_requires_jev_noul(jev_config, expected):
    with pytest.raises(ValueError, match=expected):
        OpenVikingConfig.from_dict({"rules": {"jev_rule_matcher": jev_config}})


def test_rule_compiler_context_window_rejects_too_small_value():
    with pytest.raises(ValueError, match="must be 0 or at least 2048"):
        OpenVikingConfig.from_dict({"rules": {"compiler_context_window": 1024}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("archived_retention_days", 0),
        ("archived_cleanup_interval_seconds", 59),
    ],
)
def test_rule_cleanup_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        OpenVikingConfig.from_dict({"rules": {field: value}})
