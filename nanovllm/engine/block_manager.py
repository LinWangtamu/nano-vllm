from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# A logical block represents a block position within a sequence.
# A physical block represents an actual storage block in the global KV-cache pool.
class Block:

    def __init__(self, block_id):
        self.block_id = block_id #physical block number;
        self.ref_count = 0 #how many requests share this block
        self.hash = -1  # current block corresponding tokens hash value, -1 represent no valid hash value yet.
        self.token_ids = []  # token id list  [102,205..]

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash # block hash value
        self.token_ids = token_ids # token id list under this block

    def reset(self):
        # Prepare this block for allocation to a new sequence.
        self.ref_count = 1  # The new sequence holds one reference to this block.
        self.hash = -1     # Invalidate the hash from the previous contents.
        self.token_ids = []  # Clear the recorded token IDs; GPU KV data is not erased here.


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        # Number of token positions stored in each KV cache block.
        # Example: block_size=256 means one physical block can store KV for 256 tokens.
        self.block_size = block_size
        # Create metadata objects for all physical KV cache blocks.
        # Example: num_blocks=4 -> [Block(0), Block(1), Block(2), Block(3)]
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Map a prefix hash to the physical block that stores its KV cache.
        # key:   hash value
        # value: physical block ID
        # Example: {9527: 2} means the content identified by logical block hash 9527 is stored in physical block 2.
        self.hash_to_block_id: dict[int, int] = dict()
        # IDs of currently available physical blocks.
        # popleft() can quickly select the next free block.
        # Example: num_blocks=4 -> deque([0, 1, 2, 3])
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # IDs of physical blocks currently referenced by active sequences.
        # Example: {0, 2} means blocks 0 and 2 are currently in use.
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()  # Create an empty 64-bit xxHash calculator.
        if prefix != -1:    # Include the previous block's hash so that the current hash, represents both the preceding context and this block.
            h.update(prefix.to_bytes(8, "little"))# Convert the 64-bit integer hash into 8 bytes, using little-endian byte order.
        h.update(np.array(token_ids).tobytes())    # Convert the current block's token IDs into raw bytes and add them to the hash input.
        return h.intdigest()     # Return the final 64-bit hash as a Python integer.
"""
block_0 = [10, 20, 30, 40]
block_1 = [50, 60, 70, 80]

hash_0 = BlockManager.compute_hash(block_0)
hash_1 = BlockManager.compute_hash(block_1, prefix=hash_0)
hash_0 = hash(block_0)

hash_1 = hash(hash_0 + block_1)
"""
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]  # Retrieve the metadata object for this physical block.
        assert block.ref_count == 0# A free block must not be referenced by any active sequence.
        # The free block may still contain reusable prefix-cache metadata.If the global hash table still maps this hash to this block,
    # remove that mapping because the block will be overwritten.
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:     
            del self.hash_to_block_id[block.hash]
        block.reset()     # Prepare the block for its new owner:ref_count = 1, hash = -1, token_ids = [].
        self.used_block_ids.add(block_id)    # Record that this physical block is now in use.
        return block_id  # Return the allocated physical block ID.

    def _deallocate_block(self, block_id: int):
        # A block can be released only when no active sequence references it.
        assert self.blocks[block_id].ref_count == 0
    
        # Mark the physical block as no longer being used by an active sequence.
        self.used_block_ids.remove(block_id)
    
        # Add the block ID to the free queue so that it can be reused later.
        # The old hash, token IDs, and GPU KV data are intentionally preserved
        # for possible prefix-cache reuse until the block is reallocated.
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1 # Hash of the previous logical block.-1 means that the first block has no prefix.
        num_cached_blocks = 0    # Number of consecutive prefix blocks whose KV cache can be reused.
        num_new_blocks = seq.num_blocks   # Initially assume that every logical block needs to take one block from the free-block pool.
        for i in range(seq.num_blocks - 1): #Check all logical blocks except the last one.The last block is treated as a mutable tail and is not reused here.
            token_ids = seq.block(i)  # Get the token IDs in logical block i.
            h = self.compute_hash(token_ids, h) # Compute a chained hash using:previous prefix hash + current block's token IDs.
            block_id = self.hash_to_block_id.get(h, -1) #Look for a physical block storing this cached prefix.Return -1 if the hash is not found.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # Stop at the first cache miss.The token comparison also protects against hash collisions:two different token lists could theoretically have the same hash.
                break
            num_cached_blocks += 1# This logical block's KV cache can be reused.
            if block_id in self.used_block_ids: # This physical block is already active.The new sequence can share it by increasing ref_count,so no free block is consumed.
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:    # Allocation is impossible if the free pool is too small.
            return -1
        return num_cached_blocks  # Return the number of reusable consecutive prefix blocks.
"""
seq.num_blocks = 3
num_cached_blocks = 2

# logical block 0's cached KV is in physical block 7
# logical block 1's cached KV is in physical block 2
self.hash_to_block_id = {
    hash_0: 7,
    hash_1: 2,
}

self.used_block_ids = {7}
self.free_block_ids = deque([2, 3, 4])
"""
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # A new sequence must not already have any physical-block mappings.
        # Example: seq.block_table must be [] before allocation starts.
        assert not seq.block_table
    
        # Store the previous logical block's hash.
        # -1 means that logical block 0 has no preceding prefix.
        h = -1
    
        # Reuse the first num_cached_blocks logical blocks.
        # Example: num_cached_blocks=2 -> process logical blocks 0 and 1.
        for i in range(num_cached_blocks):
    
            # Get the token IDs of logical block i.
            # Example: i=0 -> [10, 20, 30, 40].
            token_ids = seq.block(i)
    
            # Recompute the chained hash for this logical prefix.
            # Example:
            # i=0 -> h = hash(block_0)
            # i=1 -> h = hash(hash(block_0) + block_1)
            h = self.compute_hash(token_ids, h)
    
            # Use the hash to find the physical block containing the cached KV.
            # Example: hash_0 -> block 7; hash_1 -> block 2.
            block_id = self.hash_to_block_id[h]
    
            # Retrieve the metadata object for that physical block.
            # Example: block_id=7 -> self.blocks[7].
            block = self.blocks[block_id]
    
            # Check whether the cached physical block is already active.
            # Example: block 7 is currently used by another sequence.
            if block_id in self.used_block_ids:
    
                # Share the active block with this sequence.
                # Example: ref_count changes from 1 to 2.
                block.ref_count += 1
    
            else:
                # The cached block is free but still contains reusable KV data.
    
                # Reactivate the cached block for this sequence.
                # Example: block 2's ref_count changes from 0 to 1.
                block.ref_count = 1
    
                # Remove the reactivated block from the free-block queue.
                # Example: deque([2, 3, 4]) -> deque([3, 4]).
                self.free_block_ids.remove(block_id)
    
                # Mark the reactivated physical block as active.
                # Example: used_block_ids {7} -> {2, 7}.
                self.used_block_ids.add(block_id)
    
            # Record the logical-to-physical block mapping for this sequence.
            # Example:
            # after i=0: seq.block_table = [7]
            # after i=1: seq.block_table = [7, 2]
            seq.block_table.append(block_id)
    
        # Allocate new physical blocks for all uncached logical blocks.
        # Example:
        # num_cached_blocks=2 and seq.num_blocks=3
        # -> allocate only logical block 2.
        for i in range(num_cached_blocks, seq.num_blocks):
    
            # Allocate one physical block and append its ID to the block table.
            # Example: allocate physical block 3
            # -> seq.block_table changes from [7, 2] to [7, 2, 3].
            seq.block_table.append(self._allocate_block())
    
        # Record how many leading tokens already have reusable KV cache.
        # Example: 2 cached blocks * 4 tokens per block = 8 cached tokens.
        seq.num_cached_tokens = num_cached_blocks * self.block_size
"""
seq.block_table
# [7, 2, 3]
logical block 0 → cached physical block 7
logical block 1 → cached physical block 2
logical block 2 → newly allocated physical block 3
self.used_block_ids
# {2, 3, 7}

self.free_block_ids
# deque([4])
"""
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:   # Check whether a new block is available when the latest token starts a new logical block.
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)    # Example: block_size=4, len(seq)=5 -> 5 % 4 == 1,so at least one free block is required.

    def may_append(self, seq: Sequence): # Allocate a physical block if the latest token starts a new logical block.
        if len(seq) % self.block_size == 1:    # Example: block_size=4, len(seq)=5 -> allocate a block for token index 4.
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size    # Index of the first block that may be completed in this step. Example: 4 cached tokens / block_size 4 = block 1.
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size #One-past-the-last complete block after this step. Example: (4 cached + 4 scheduled) / 4 = 2.
        if start == end: return    # No complete block was produced in this step.  # Continue the chained hash from the previous block.The first block has no prefix, so use -1.
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1#
        for i in range(start, end):   # Hash each newly completed logical block.
            block = self.blocks[seq.block_table[i]] # Find the corresponding physical block.
            token_ids = seq.block(i) # Get this logical block's token IDs.
            h = self.compute_hash(token_ids, h) # Compute the hash using the previous hash and current tokens.
            block.update(h, token_ids)# Store the hash and token IDs in the physical block metadata.
            self.hash_to_block_id[h] = block.block_id  # Register this block in the global prefix-cache lookup table.
