"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


def extract_numeric(text):
    """Convert an answer string (e.g. '42', '-3.5', '1,200') to float, or None if unparseable."""
    if text is None:
        return None
    try:
        return float(text.replace(",", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Reward functions for Part 4
# Each takes (pred_num: float|None, ref_num: float|None, assistant_response: str)
# and returns a float reward in [0, 1].
# ---------------------------------------------------------------------------

def reward_binary(pred_num, ref_num, assistant_response):
    """Original binary reward: 1.0 if exact string match, else 0.0."""
    if pred_num is not None and ref_num is not None and pred_num == ref_num:
        return 1.0
    return 0.0


def reward_format(pred_num, ref_num, assistant_response):
    """Reward correct format: 1.0 if correct, 0.2 if #### present but wrong, 0.0 if no ####."""
    if pred_num is not None and ref_num is not None and pred_num == ref_num:
        return 1.0
    if "####" in assistant_response:
        return 0.2
    return 0.0


def reward_tolerance(pred_num, ref_num, assistant_response):
    """Reward numerical closeness: 1.0 exact, 0.5 within 10%, 0.0 otherwise."""
    pn = extract_numeric(str(pred_num)) if pred_num is not None else None
    rn = extract_numeric(str(ref_num)) if ref_num is not None else None
    if pn is not None and rn is not None:
        if pn == rn:
            return 1.0
        if rn != 0 and abs(pn - rn) / abs(rn) <= 0.10:
            return 0.5
        if rn == 0 and abs(pn) <= 1e-6:
            return 1.0
    return 0.0


def reward_steps(pred_num, ref_num, assistant_response):
    """Reward showing work: 1.0 if correct, else partial credit for step-like structure."""
    if pred_num is not None and ref_num is not None and pred_num == ref_num:
        return 1.0
    lines = assistant_response.strip().split("\n")
    non_empty = [l for l in lines if l.strip()]
    credit = 0.0
    if len(non_empty) >= 3:
        credit += 0.1
    if assistant_response.count("=") >= 2:
        credit += 0.1
    digit_lines = [l for l in non_empty if any(c.isdigit() for c in l)]
    if len(digit_lines) >= 2:
        credit += 0.1
    return credit


def reward_combined(pred_num, ref_num, assistant_response):
    """Weighted combination: 0.2*format + 0.3*tolerance + 0.5*steps."""
    return (
        0.2 * reward_format(pred_num, ref_num, assistant_response)
        + 0.3 * reward_tolerance(pred_num, ref_num, assistant_response)
        + 0.5 * reward_steps(pred_num, ref_num, assistant_response)
    )


REWARD_FNS = {
    "binary": reward_binary,
    "format": reward_format,
    "tolerance": reward_tolerance,
    "steps": reward_steps,
    "combined": reward_combined,
}


class GSM8K(Task):

    def __init__(self, subset, split, reward_fn="binary", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        assert reward_fn in REWARD_FNS, f"Unknown reward_fn '{reward_fn}', choose from {list(REWARD_FNS.keys())}"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        self.reward_fn = REWARD_FNS[reward_fn]
        self.reward_fn_name = reward_fn

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. Dispatches to the selected reward function.
        Extracts pred/ref answer strings and passes them along with the full response.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # Extract ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return self.reward_fn(pred_num, ref_num, assistant_response)
