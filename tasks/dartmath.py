"""
DART-Math-Hard dataset by HKUST.
https://huggingface.co/datasets/hkust-nlp/dart-math-hard

585K hard math problems with chain-of-thought solutions,
generated via Difficulty-Aware Rejection Tuning (DART).
"""

from datasets import load_dataset
from tasks.common import Task

class DARTMath(Task):
    """ DART-Math-Hard dataset. train is ~585K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "DARTMath only has a train split"
        self.ds = load_dataset("hkust-nlp/dart-math-hard", split=split).shuffle(seed=42)
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
