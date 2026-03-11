"""
Orca-Math word problems dataset by Microsoft.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

200K math word problems with step-by-step solutions,
specifically designed for smaller language models.
"""

from datasets import load_dataset
from tasks.common import Task

class OrcaMath(Task):
    """ Orca-Math dataset. train is ~200K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "OrcaMath only has a train split"
        self.ds = load_dataset("microsoft/orca-math-word-problems-200k", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        assert isinstance(question, str) and len(question) > 0
        assert isinstance(answer, str) and len(answer) > 0
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {"messages": messages}
