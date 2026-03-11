"""
MetaMathQA dataset by Meta.
https://huggingface.co/datasets/meta-math/MetaMathQA

395K math QA pairs with chain-of-thought reasoning.
Augments GSM8K and MATH via question rephrasing and backward reasoning.
"""

from datasets import load_dataset
from tasks.common import Task

class MetaMathQA(Task):
    """ MetaMathQA dataset. train is ~395K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "MetaMathQA only has a train split"
        self.ds = load_dataset("meta-math/MetaMathQA", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        query = row["query"]
        response = row["response"]
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(response, str) and len(response) > 0
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}
