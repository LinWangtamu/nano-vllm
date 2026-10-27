import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)} # config_fields = {"model","max_num_batched_tokens","max_model_len"...}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} #config_kwargs = {"max_num_batched_tokens":16384, "max_num_seqs": 100} only include the argument that include in config field name
        config = Config(model, **config_kwargs)#Instantiate a Config object
        Sequence.block_size = config.kvcache_block_size
        self.ps = [] #store process
        self.events = []#store event
        ctx = mp.get_context("spawn")#
        for i in range(1, config.tensor_parallel_size):#subprocess from 1
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()#start multiprocess
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)# rank 0 's model runner: object run in main process
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        # Broadcast "exit" to all ModelRunners (rank 0 here, ranks 1..n-1 in worker processes);
        # each one cleans up and the workers leave their command loop. Does not wait for them.
        self.model_runner.call("exit")
        # Drop the reference to the rank-0 runner so it can be garbage-collected (frees its GPU memory).
        # This deletes the attribute, not the main process.
        del self.model_runner
        # Block until every worker process has actually finished its cleanup and exited.
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # Accept either raw text or token ids; encode text into token ids.
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
           # Wrap the token ids and sampling settings into a Sequence (one request; starts in WAITING state).
        seq = Sequence(prompt, sampling_params)
            # Append it to the scheduler's waiting queue; it is prefilled when a later step() schedules it.
        self.scheduler.add(seq)

    def step(self):
        # Pick the batch for this step. A step is either all-prefill or all-decode (no mixed batches).
        # e.g. prefill step: seqs = [seq0 (5 prompt tokens), seq1 (8), seq2 (6)], is_prefill = True
        #      decode step:  seqs = [seq0, seq1, seq2],                           is_prefill = False
        seqs, is_prefill = self.scheduler.schedule()
    
        # Token count for throughput stats: prefill -> number of prompt tokens scheduled (positive);
        # decode -> one token per sequence, stored as negative so the caller can tell the two apart.
        # e.g. prefill: 5 + 8 + 6 = 19;  decode: -3
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    
        # Run the forward pass on all ranks; returns one sampled token id per sequence.
        # e.g. token_ids = [791, 13, 4438]  (one new token for seq0, seq1, seq2)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
    
        # Append the new tokens, mark sequences that hit EOS / max_tokens as finished, free their KV blocks.
        # e.g. seq0 gets 791 appended; if seq1 has now reached max_tokens, it is marked finished and removed
        #      from the running queue.
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
    
        # Return (seq_id, generated token ids) for the sequences that finished in this step.
        # e.g. nothing finished -> outputs = []
        #      seq1 finished    -> outputs = [(1, [13, 220, 1820, ..., 128009])]  (all tokens seq1 generated)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # Progress bar over requests (advances by 1 each time a request finishes).
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
  # One SamplingParams for all prompts -> repeat it so each prompt has its own entry.
    # e.g. 3 prompts, sp -> [sp, sp, sp]  (the same object 3 times; fine, it is only read)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
    # Queue every request (all go to the waiting queue; nothing runs yet).
    # e.g. prompts = ["Hello", "What is AI?", "Write a poem"] -> seq_id 0, 1, 2
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # Loop while there are still unfinished requests (waiting or running); one step() per iteration.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
               # Throughput of this single step (not an average); the sign of num_tokens says which phase.
        # e.g. prefill step, 19 tokens in 0.02 s -> 950 tok/s;  decode step, -3 in 0.015 s -> 200 tok/s
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
              # Store the requests that finished in this step.
        # e.g. output = [(1, [13, 220, ..., 128009])] -> outputs = {1: [13, 220, ..., 128009]}
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
