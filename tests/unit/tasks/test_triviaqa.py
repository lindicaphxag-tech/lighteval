from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.triviaqa import triviaqa, triviaqa_prompt


def _score(prediction: str) -> float:
    doc = triviaqa_prompt(
        {
            "question": "Who won Super Bowl 50?",
            "answer": {"aliases": ["Denver Broncos", "Broncos"]},
        },
        task_name="triviaqa",
    )
    result = triviaqa.metrics[0].compute_sample(
        doc=doc,
        model_response=ModelResponse(text=[prediction]),
    )
    return result["em"]


def test_triviaqa_prediction_normalization_matches_gold_normalization():
    assert _score("Denver Broncos!") == 1.0
    assert _score("BRONCOS.") == 1.0
    assert _score("Carolina Panthers") == 0.0
