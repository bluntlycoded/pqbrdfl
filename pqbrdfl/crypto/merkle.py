"""Append-only Merkle audit log with leaf/node domain separation (domains AUD_LEAF and AUD_NODE).

Each aggregation round appends the digest of its signed round certificate, so
any agent can later prove which aggregate was accepted in which round.
"""

from __future__ import annotations

from typing import Callable

from .domains import D


class MerkleLog:
    def __init__(self, H: Callable[..., bytes]):
        self.H = H
        self.leaves: list[bytes] = []

    def _leaf(self, data: bytes) -> bytes:
        return self.H(D.AUD_LEAF, data, 32)

    def _node(self, left: bytes, right: bytes) -> bytes:
        return self.H(D.AUD_NODE, left + right, 32)

    def append(self, data: bytes) -> int:
        self.leaves.append(self._leaf(data))
        return len(self.leaves) - 1

    def _levels(self) -> list[list[bytes]]:
        levels = [list(self.leaves)]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            nxt = [self._node(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i]
                   for i in range(0, len(cur), 2)]
            levels.append(nxt)
        return levels

    def root(self) -> bytes:
        if not self.leaves:
            return self.H(D.AUD_NODE, b"", 32)
        return self._levels()[-1][0]

    def proof(self, index: int) -> list[tuple[bool, bytes]]:
        """Inclusion proof: list of (sibling_is_left, sibling_hash)."""
        path = []
        for level in self._levels()[:-1]:
            sib = index ^ 1
            if sib < len(level):
                path.append((sib < index, level[sib]))
            index //= 2
        return path

    def verify(self, data: bytes, index: int, proof, root: bytes) -> bool:
        h = self._leaf(data)
        for is_left, sib in proof:
            h = self._node(sib, h) if is_left else self._node(h, sib)
        return h == root
