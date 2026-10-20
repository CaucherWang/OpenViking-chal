from unittest.mock import MagicMock, patch

from openviking.service.rule_jev import create_rule_jev_judge
from openviking_cli.utils.config.rerank_config import RerankConfig


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
    assert client.rerank_batch("task", ["case"]) == [0.93]
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
