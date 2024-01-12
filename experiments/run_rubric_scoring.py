import wandb
import time

from tqdm import tqdm
import pandas as pd

from langchain_core.output_parsers import StrOutputParser
from experiment_utils import load_llm, get_gpt_reference, prometheus_rubric

from box import Box
from typing import List

# import scoring methodologies
from rouge import Rouge

OUTPUT_PARSER = StrOutputParser()
DESIRED_COLUMNS = [
    "query",
    "challenge",
    "reference",
    "gpt_reference",
    "model_reference",
]


llm = load_llm("gpt-4", temperature=0.6)
rubric = Box(prometheus_rubric())


def create_dataframe(run_ids):
    api = wandb.Api()

    # Creating a dataset of two longer runs that have summarization, debugging, and QA
    df = pd.DataFrame()
    for run in run_ids:
        run = api.run(f"/opentensor-dev/synapse_agent_experiments/runs/{run}")
        df = pd.concat([df, pd.DataFrame(list(run.scan_history()))])

    return df


def compute_rouge(string1, string2, key: str, return_metric="f"):
    # Compute rouge scores between two strings and return one of the metrics for simple logging
    rouge = Rouge()
    rouge_scores = rouge.get_scores(string1, string2)[0]

    return {
        f"rouge_1_{return_metric}_score_{key}": rouge_scores["rouge-1"][
            return_metric
        ],
        f"rouge-2_{return_metric}_score_{key}": rouge_scores["rouge-2"][
            return_metric
        ],
        f"rouge-l_{return_metric}_score_{key}": rouge_scores["rouge-l"][
            return_metric
        ],
    }


def main(run_ids=["f6ul5wr8", "85sx63hh"], tasks: List = ["qa"]):
    wandb.init(
        project="agent_experiments", entity="sn1", tags=["rubric"] + tasks
    )

    df = create_dataframe(run_ids=run_ids)

    results = []
    # sample the dataframe 3 times on each topic
    for task in tasks:
        subset = df[df.task_name == task]
        sampled = subset.groupby("topic").apply(lambda x: x.sample(3))

        for _, row in tqdm(sampled.iterrows()):
            for desired_score in rubric.possible_scores:
                user_prompt = rubric.user_prompt_template.format(
                    query=row["query"],
                    reference=row["reference"],
                    desired_score=desired_score,
                )

                messages = [
                    {"role": "system", "content": rubric.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                t0 = time.time()
                generated_reference = get_gpt_reference(
                    message=messages, model=llm, output_parser=OUTPUT_PARSER
                )
                gpt_reference_time = time.time() - t0

                ## Computing scorings
                wiki_reference_to_generated_reference_scores = compute_rouge(
                    row.reference, generated_reference, key="wiki_gen"
                )

                gpt_reference_to_generated_reference_scores = compute_rouge(
                    row.gpt_reference, generated_reference, key="gpt_gen"
                )

                result = {
                    "task_name": task,
                    "user_prompt": user_prompt,
                    "system_prompt": rubric.system_prompt,
                    "original_reference": row.reference,
                    "desired_score": desired_score,
                    "generated_reference": generated_reference,
                    "gpt_reference_time": gpt_reference_time,
                    **wiki_reference_to_generated_reference_scores,
                    **gpt_reference_to_generated_reference_scores,
                }

                results.append(result)
                wandb.log(result)


main(tasks=["qa", "summarization"])
