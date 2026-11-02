from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # Maximum number of sequences selected in one scheduling step.
        # Example: max_num_seqs = 4 allows at most 4 sequences in this step.
        self.max_num_seqs = config.max_num_seqs

        # Token budget used by the prefill scheduling loop below.
        # Example: with a budget of 512, two uncached prompts of lengths
        # 200 and 300 can fit together: 200 + 300 = 500.
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # The token ID that indicates the end of a generated response.
        self.eos = config.eos

        # Number of token positions stored in each KV cache block.
        # Example: block_size = 256; 600 tokens require ceil(600 / 256) = 3 blocks.
        self.block_size = config.kvcache_block_size

        # Manage physical KV cache blocks: allocation, sharing, and release.
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )

        # Sequences waiting for prefill.
        # This includes new requests, partially prefilled requests,
        # and requests moved back here by preemption.
        self.waiting: deque[Sequence] = deque()

        # Sequences admitted to the running queue.
        # A sequence can enter this queue when its final prefill chunk
        # is scheduled, before that chunk actually executes.
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # The scheduler is finished only when BOTH queues are empty.
        # Example:
        # waiting = [], running = [A] -> False
        # waiting = [], running = []  -> True
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # Add a new request to the back of the waiting queue.
        # Example: [A, B] -> [A, B, C] after adding C.
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Return:
        #   1. The sequences selected for this step.
        #   2. Whether this is a prefill step: True=prefill, False=decode.
        #
        # This implementation prioritizes prefill and returns a batch
        # containing either prefill work or decode work.

        scheduled_seqs = []
        num_batched_tokens = 0

        # -------------------- PREFILL --------------------

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # Inspect the first waiting sequence without removing it.
            # This preserves queue order.
            seq = self.waiting[0]

            # Remaining prefill token budget for this step.
            # Example: budget=512, already selected=200 -> remaining=312.
            remaining = self.max_num_batched_tokens - num_batched_tokens

            if remaining == 0:
                break

            if not seq.block_table:
                # No KV blocks are currently assigned to this sequence.
                #
                # can_allocate() returns:
                #   -1: insufficient block capacity.
                #    0: allocation is possible, but no cached prefix is reusable.
                #   >0: number of reusable prefix blocks.
                num_cached_blocks = self.block_manager.can_allocate(seq)

                if num_cached_blocks == -1:
                    # Stop trying to add prefill work at this queue head.
                    # If nothing has been selected, execution below can
                    # fall through to the decode scheduling loop.
                    break

                # Only tokens outside the reusable prefix need computation.
                #
                # Example:
                # total tokens = 600
                # cached prefix = 2 blocks * 256 tokens = 512 tokens
                # tokens to compute = 600 - 512 = 88
                num_tokens = (
                    seq.num_tokens
                    - num_cached_blocks * self.block_size
                )

            else:
                # This sequence already has allocated blocks.
                # For example, an earlier step processed a prefill chunk.
                #
                # Example: 600 total tokens, 256 cached -> 344 still to compute.
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # A sequence can be split into a prefill chunk only if it is
            # the first sequence selected in this batch.
            #
            # Example 1:
            # remaining=256, num_tokens=600, scheduled_seqs=[]
            # -> Allow the first 256 tokens of this sequence.
            #
            # Example 2:
            # remaining=100, num_tokens=200, scheduled_seqs=[A]
            # -> Do not add a partial chunk of this second sequence.
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                # Assign KV cache blocks and attach any reusable prefix.
                # allocate() also sets seq.num_cached_tokens for that prefix.
                #
                # In this implementation, allocation covers the sequence's
                # current token length, even if prefill executes in chunks.
                self.block_manager.allocate(seq, num_cached_blocks)

            # Schedule either all remaining tokens or one budget-sized chunk.
            # Example: min(600, 256) = 256 tokens.
            seq.num_scheduled_tokens = min(num_tokens, remaining)

            num_batched_tokens += seq.num_scheduled_tokens

            # Check whether THIS scheduled chunk will finish prefill.
            #
            # Example:
            # cached=512, scheduled=88, total=600
            # -> The final chunk has been selected.
            if (
                seq.num_cached_tokens + seq.num_scheduled_tokens
                == seq.num_tokens
            ):
                seq.status = SequenceStatus.RUNNING

                # Move this sequence from waiting to running.
                self.waiting.popleft()
                self.running.append(seq)

            # A partially prefilled sequence stays in waiting,
            # but its current chunk is still included in this step.
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # Return immediately if any prefill work was selected.
            # Decode requests will wait for a later scheduling step.
            return scheduled_seqs, True

        # -------------------- DECODE --------------------

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # Take the first running sequence for this decode step.
            seq = self.running.popleft()

            # Check whether this sequence can store KV for its next input token.
            # A new physical block is needed only when crossing a block boundary.
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Release another sequence's block references to make room.
                    # Choose from the back of the running queue.
                    #
                    # Example:
                    # current seq=A, remaining running=[B, C]
                    # -> Preempt C first.
                    self.preempt(self.running.pop())

                else:
                    # No other running sequence can be preempted.
                    # Move the current sequence back to waiting.
                    self.preempt(seq)
                    break

            else:
                # Python's "while ... else":
                # This branch runs if the while loop ends WITHOUT a break.
                #
                # It also runs if can_append(seq) was True immediately,
                # so the while loop body never executed.

                # Ordinary decode processes one input token per sequence.
                # This is the previously sampled token whose KV is not yet cached.
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False

                # Allocate a new block if this token needs one.
                self.block_manager.may_append(seq)

                scheduled_seqs.append(seq)

        # At least one sequence must have been selected for execution.
        assert scheduled_seqs

        # Put selected sequences back at the FRONT, preserving their order.
        #
        # extendleft() reverses the order of its input, so reversed() compensates.
        #
        # Example:
        # running=[C], scheduled_seqs=[A, B]
        # -> running becomes [A, B, C].
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # Move a running sequence back to waiting for prefill/recomputation.
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True

        # Release this sequence's KV block references and clear its block table.
        # Its token IDs are preserved, including tokens already generated.
        #
        # Physical blocks may remain reusable through prefix caching;
        # releasing references does not necessarily erase their contents.
        self.block_manager.deallocate(seq)

        # Place the preempted sequence at the front of the waiting queue.
        # Example: waiting=[D, E] -> [C, D, E] after preempting C.
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        # Match each executed sequence with its sampled token ID.
        #
        # Example:
        # seqs=[A, B], token_ids=[123, 456]
        # -> (A, 123), then (B, 456).
        for seq, token_id in zip(seqs, token_ids):
            # Record hashes for blocks completed by this execution step.
            # These hashes support future prefix-cache lookup.
            #
            # This happens BEFORE advancing num_cached_tokens, so the block
            # manager can identify which blocks were completed in this step.
            self.block_manager.hash_blocks(seq)

            # Tokens processed in this step now have their KV cached.
            #
            # Example:
            # before: num_cached_tokens=256, num_scheduled_tokens=256
            # after:  num_cached_tokens=512
            seq.num_cached_tokens += seq.num_scheduled_tokens

            # Clear the per-step scheduling count after accounting for it.
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # An intermediate prefill chunk has finished.
                # The full input has not been processed yet, so ignore
                # the sampled token returned for this intermediate chunk.
                #
                # Example: cached=256, total=600 -> more prefill is needed.
                continue

            # Full prefill or one decode step has finished:
            # append the sampled token to the sequence.
            #
            # Example:
            # token_ids=[101, 102, 103], sampled token_id=999
            # -> token_ids becomes [101, 102, 103, 999].
            #
            # The newly sampled token has NOT been processed by the model yet.
            # Its KV will be computed when it becomes a later step's input.
            seq.append_token(token_id)

            # Finish if:
            #   1. EOS was generated and EOS is not being ignored, OR
            #   2. The requested number of output tokens has been reached.
            #
            # Example:
            # max_tokens=20, num_completion_tokens=20 -> finish.
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED

                # Release this completed sequence's KV block references.
                self.block_manager.deallocate(seq)

                # Remove it from running so it receives no further decode steps.
                self.running.remove(seq)
