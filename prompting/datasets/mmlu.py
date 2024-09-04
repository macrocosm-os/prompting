import random
import pandas as pd
from typing import ClassVar
from prompting.datasets.base import MMLUEntry, BaseDataset

MMLU_TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MMLUDataset(BaseDataset):
    name: ClassVar[str] = "mmlu"
    max_tries: int = 10
    data_queue: set[MMLUEntry] = set()

    def random(self) -> MMLUEntry:
        df = pd.read_parquet("hf://datasets/cais/mmlu/" + random.choice(MMLU_TASKS))
        [
            self.data_queue.add(
                MMLUEntry(
                    query=row["question"], subject=row["subject"], choices=row["choices"], answer=str(row["answer"])
                )
            )
            for _, row in df.iterrows()
        ]
        sample = df.sample(1).iloc[0]
        return MMLUEntry(
            query=sample["question"], subject=sample["subject"], choices=sample["choices"], answer=str(sample["answer"])
        )

    def get(self) -> MMLUEntry:
        if not self.data_queue:
            self.random()
        return self.data_queue.pop()
