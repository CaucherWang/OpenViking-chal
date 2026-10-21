from unittest.mock import MagicMock, patch

import pytest

from openviking.service.rule_jev import (
    RuleJevClient,
    RuleJevInput,
    RuleJevScore,
    create_rule_jev_judge,
)
from openviking_cli.utils.config.rerank_config import RerankConfig


@pytest.mark.parametrize("applies_when", ["none", "(none)", "[]"])
def test_rule_jev_input_rejects_placeholder_applies_when(applies_when):
    with pytest.raises(ValueError, match="requires applies_when"):
        RuleJevInput(
            name="Protect production configuration",
            content="Obtain approval before changing production configuration.",
            applies_when=applies_when,
            exception="",
        )


@pytest.mark.parametrize("applies_when", ["", " \n"])
def test_rule_jev_input_accepts_empty_applies_when(applies_when):
    rule = RuleJevInput(
        name="Unconditional constraint",
        content="Always preserve confidentiality.",
        applies_when=applies_when,
        exception="",
    )
    assert rule.applies_when == ""


@patch("openviking.models.rerank.jev_rerank.httpx.Client")
def test_rule_jev_judge_isolates_custom_question_and_headers(mock_client_class):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "answers": {
            "relevance_0": {
                "type": "noul",
                "noul": 0.93,
            }
        }
    }
    mock_client_class.return_value.post.return_value = response
    config = RerankConfig(
        provider="jev",
        api_key="key",
        mode="noul",
        extra_headers={"ark-beta-systemone-api": "true"},
    )

    client = create_rule_jev_judge(
        config,
        question="Is the candidate completely unrelated?",
        true_criterion="There is no plausible connection.",
        false_criterion="The candidate may be relevant.",
    )

    assert client is not None
    assert client.rerank_batch("task", ["scenario"]) == [0.93]
    question = mock_client_class.return_value.post.call_args.kwargs["json"]["questions"][
        "relevance_0"
    ]
    assert question["instructions"]["question"] == "Is the candidate completely unrelated?"
    assert question["criteria"] == {
        "true": "There is no plausible connection.",
        "false": "The candidate may be relevant.",
    }
    mock_client_class.return_value.headers.update.assert_called_once_with(
        {"ark-beta-systemone-api": "true"}
    )
    client.close()


@patch("openviking.service.rule_jev.httpx.Client")
def test_rule_jev_sends_rule_state_and_clause_criteria(mock_client_class):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "answers": {
            "applies_when_0": {"type": "noul", "noul": 0.91},
            "exception_0": {"type": "noul", "noul": 0.12},
        }
    }
    mock_client_class.return_value.post.return_value = response
    config = RerankConfig(provider="jev", api_key="key", mode="noul")
    rules = [
        RuleJevInput(
            name="Protect production configuration",
            content="Obtain approval before changing production configuration.",
            applies_when="The task changes production configuration.",
            exception="The task is limited to read-only inspection.",
        )
    ]

    client = RuleJevClient.from_config(config)

    assert client is not None
    assert client.judge_rules("Deploy production configuration.", rules) == [
        RuleJevScore(applies_when=0.91, exception=0.12)
    ]
    request = mock_client_class.return_value.post.call_args.kwargs["json"]
    assert request["state"] == {
        "task_context": "Deploy production configuration.",
        "rules": [
            {
                "name": "Protect production configuration",
                "content": "Obtain approval before changing production configuration.",
            }
        ],
    }
    assert "query" not in request["state"]
    assert "candidate_documents" not in request["state"]
    assert request["questions"]["applies_when_0"]["instructions"] == {
        "rule_index": 0,
        "question": (
            "Does rules[rule_index] apply to task_context? Judge only applicability, "
            "not whether the Rule is satisfied or violated."
        ),
    }
    assert request["questions"]["applies_when_0"]["criteria"] == {
        "true": "The task changes production configuration."
    }
    assert request["questions"]["exception_0"]["instructions"] == {
        "rule_index": 0,
        "question": "Does an exception to rules[rule_index] apply to task_context?",
    }
    assert request["questions"]["exception_0"]["criteria"] == {
        "true": "The task is limited to read-only inspection."
    }
    assert mock_client_class.return_value.post.call_count == 1
    client.close()


@patch("openviking.service.rule_jev.httpx.Client")
def test_rule_jev_rejects_missing_expected_answer(mock_client_class):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"answers": {}}
    mock_client_class.return_value.post.return_value = response
    client = RuleJevClient.from_config(RerankConfig(provider="jev", api_key="key", mode="noul"))

    assert client is not None
    assert (
        client.judge_rules(
            "Deploy production configuration.",
            [
                RuleJevInput(
                    name="Protect production configuration",
                    content="Obtain approval before changing production configuration.",
                    applies_when="The task changes production configuration.",
                    exception="",
                )
            ],
        )
        is None
    )
    client.close()


@patch("openviking.service.rule_jev.httpx.Client")
def test_rule_jev_omits_empty_exception_question(mock_client_class):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"answers": {"applies_when_0": {"type": "noul", "noul": 0.84}}}
    mock_client_class.return_value.post.return_value = response
    client = RuleJevClient.from_config(RerankConfig(provider="jev", api_key="key", mode="noul"))

    assert client is not None
    assert client.judge_rules(
        "Deploy production configuration.",
        [
            RuleJevInput(
                name="Protect production configuration",
                content="Obtain approval before changing production configuration.",
                applies_when="The task changes production configuration.",
                exception="",
            )
        ],
    ) == [RuleJevScore(applies_when=0.84, exception=0.0)]
    questions = mock_client_class.return_value.post.call_args.kwargs["json"]["questions"]
    assert set(questions) == {"applies_when_0"}
    client.close()


@patch("openviking.service.rule_jev.httpx.Client")
def test_rule_jev_empty_applies_when_judges_exception_only(mock_client_class):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"answers": {"exception_0": {"type": "noul", "noul": 0.9}}}
    mock_client_class.return_value.post.return_value = response
    client = RuleJevClient.from_config(RerankConfig(provider="jev", api_key="key", mode="noul"))

    assert client is not None
    rules = [
        RuleJevInput("Global rule", "Require approval.", "", "Emergency recovery."),
        RuleJevInput("Other global rule", "Preserve confidentiality.", "", ""),
    ]
    assert client.judge_rules("Emergency recovery.", rules) == [
        RuleJevScore(applies_when=1.0, exception=0.9),
        RuleJevScore(applies_when=1.0, exception=0.0),
    ]
    questions = mock_client_class.return_value.post.call_args.kwargs["json"]["questions"]
    assert set(questions) == {"exception_0"}
    client.close()


@patch("openviking.service.rule_jev.httpx.Client")
def test_rule_jev_unconditional_rules_do_not_request_judgment(mock_client_class):
    client = RuleJevClient.from_config(RerankConfig(provider="jev", api_key="key", mode="noul"))
    assert client is not None
    assert client.judge_rules(
        "Unrelated topic.", [RuleJevInput("Global rule", "Require approval.", "", "")]
    ) == [RuleJevScore(applies_when=1.0, exception=0.0)]
    mock_client_class.return_value.post.assert_not_called()
    client.close()
