"""
name:
Triviaqa

dataset:
mandarjoshi/trivia_qa

abstract:
TriviaqQA is a reading comprehension dataset containing over 650K
question-answer-evidence triples. TriviaqQA includes 95K question-answer pairs
authored by trivia enthusiasts and independently gathered evidence documents,
six per question on average, that provide high quality distant supervision for
answering the questions.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1705.03551
"""

import numpy as np

from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.normalizations import harness_triviaqa_normalizer
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


def triviaqa_prompt(line, task_name: str = None):
    def _remove_prefixes(aliases):
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)
        return ret

    list_of_candidates = [harness_triviaqa_normalizer(alias) for alias in _remove_prefixes(line["answer"]["aliases"])]

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        gold_index=0,
        choices=[list_of_candidates],
    )


triviaqa_exact_match = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=ExactMatches(
        strip_strings=True,
        normalize_pred=harness_triviaqa_normalizer,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


triviaqa = LightevalTaskConfig(
    name="triviaqa",
    prompt_function=triviaqa_prompt,
    hf_repo="mandarjoshi/trivia_qa",
    hf_subset="rc.nocontext",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[triviaqa_exact_match],
    stop_sequence=["\n", ".", ","],
    version=1,
)

TASKS_TABLE = [
    triviaqa,
]
