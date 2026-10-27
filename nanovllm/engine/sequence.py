from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

# Running example used in the comments below:
#   a prompt of 300 tokens, max_tokens = 128, block_size = 256
#   seq = Sequence([t0, t1, ..., t299], SamplingParams(temperature=0.6, max_tokens=128))


class SequenceStatus(Enum):
    """Lifecycle state of a request: WAITING -> RUNNING -> FINISHED (RUNNING -> WAITING on preemption)."""
    WAITING = auto()    # queued, prefill not started (or preempted)
    RUNNING = auto()    # in the running batch, being prefilled or decoded
    FINISHED = auto()   # hit EOS or max_tokens; KV blocks freed


class Sequence:
    """One request: its tokens (prompt + generated), sampling settings and KV-cache block table."""

    # Class attributes: shared by all Sequence objects.
    block_size = 256    # tokens per KV-cache block
    counter = count()   # global id generator: next(counter) -> 0, 1, 2, ...

    # Note: the default SamplingParams() is created once, when the function is defined,
    # and shared by every call that omits it. Fine here because it is only read.
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # Unique, increasing id. e.g. first Sequence -> 0, second -> 1, ...
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # Copy so that appending generated tokens does not change the caller's list.
        # e.g. self.token_ids = [t0, ..., t299]
        self.token_ids = copy(token_ids)
        # Last token; decode steps only need this one as model input. e.g. t299
        self.last_token = token_ids[-1]
        # Total tokens so far (prompt + generated). e.g. 300; grows by 1 per decode step
        self.num_tokens = len(self.token_ids)
        # Prompt length; never changes. e.g. 300
        self.num_prompt_tokens = len(token_ids)
        # Prompt tokens whose KV is already in the cache (prefix-cache hit). e.g. 256 if block 0 was cached
        self.num_cached_tokens = 0
        # Tokens scheduled for this step (chunked prefill may do only part of the prompt). e.g. 44
        self.num_scheduled_tokens = 0
        # True until the whole prompt has been prefilled.
        self.is_prefill = True
        # Physical KV-cache block ids for this sequence. e.g. [3, 7]: tokens 0-255 in block 3, 256-299 in block 7
        self.block_table = []
        # Copy the sampling settings onto the sequence. e.g. 0.6, 128, False
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # len(seq) -> total number of tokens. e.g. 300
    def __len__(self):
        return self.num_tokens

    # seq[i] or seq[a:b] -> index into token_ids. e.g. seq[0] -> t0, seq[-1] -> t299
    def __getitem__(self, key):
        return self.token_ids[key]

    # @property: called like an attribute, without (). e.g. seq.is_finished -> False
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # Number of generated tokens. e.g. after 5 decode steps: 305 - 300 = 5
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # Prompt part of token_ids. e.g. [t0, ..., t299]
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # Generated part of token_ids. e.g. [g0, g1, g2, g3, g4] after 5 decode steps
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # Number of KV blocks needed = ceil(num_tokens / block_size).
    # e.g. 300 tokens -> (300 + 255) // 256 = 2;  256 tokens -> 1;  257 tokens -> 2
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # Tokens stored in the last block. e.g. 300 - (2 - 1) * 256 = 44;  256 tokens -> 256 (last block full)
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # Token ids of block i (used to hash blocks for prefix caching).
    # e.g. block(0) -> [t0, ..., t255] (256 tokens), block(1) -> [t256, ..., t299] (44 tokens)
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # Called once per decode step with the newly sampled token.
    # e.g. append_token(g0): token_ids gets g0 at the end, last_token = g0, num_tokens 300 -> 301
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # __getstate__ / __setstate__ control how pickle serializes this object.
    # The main process pickles the batch and sends it to the worker processes through shared memory
    # every step, so only what the workers need for the forward pass is sent:
    #   prefill: the full token_ids (the whole prompt is model input)
    #   decode:  only last_token (the one new input token), to keep the message small
    # Not sent: seq_id, status, temperature, max_tokens, ignore_eos, is_prefill
    # (sampling and finish checks happen in the main process only).
    # e.g. prefill: (300, 300, 0, 300, [3, 7], [t0, ..., t299])
    #      decode:  (305, 300, 0, 1,   [3, 7], g4)
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # Runs in the worker process when unpickling (__init__ is NOT called there).
    # e.g. prefill state -> token_ids = [t0, ..., t299], last_token = t299
    #      decode state  -> token_ids = [], last_token = g4 (num_tokens still says 305, so len(seq) stays correct)
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
