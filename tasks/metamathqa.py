"""
MetaMathQA dataset by Meta.
https://huggingface.co/datasets/meta-math/MetaMathQA

395K math QA pairs with chain-of-thought reasoning.
Augments GSM8K and MATH via question rephrasing and backward reasoning.
"""

from datasets import load_dataset
from tasks.common import Task

class MetaMathQA(Task):
    """ MetaMathQA dataset. train is ~395K rows (4 rows filtered for empty query). """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "MetaMathQA only has a train split"
        ds = load_dataset("meta-math/MetaMathQA", split=split)
        # Filter out rows with empty/missing query or response
        self.ds = ds.filter(lambda row: bool(row["query"]) and bool(row["response"])).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": row["response"]},
        ]
        return {"messages": messages}
