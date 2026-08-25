"""Information-theoretic coding: Shannon limit, Huffman code, typical set."""

import heapq

import numpy as np

from stochpylib.information_theory._base import (
    _validate_probs, _safe_log2, _normalise,
)
from stochpylib.information_theory.entropy import Entropy

__all__ = [
    "ShannonLimit", "HuffmanCode", "TypicalSet", "AEP",
]


class ShannonLimit:
    """Shannon channel-coding theorem limit: maximum achievable rate = C.

    For a BSC with crossover p, capacity is 1 - H_b(p) bits per channel use.
    """

    def __init__(self, crossover_prob=None):
        self.p = float(crossover_prob) if crossover_prob is not None else None

    def fit(self):
        if self.p is not None:
            from stochpylib.queueing.birth_death import erlang_c_formula \
                as _unused
            h_b = -self.p * np.log2(max(self.p, 1e-300)) \
                - (1 - self.p) * np.log2(max(1 - self.p, 1e-300))
            self.capacity_ = max(1.0 - h_b, 0.0)
        else:
            raise ValueError("crossover_prob must be specified for BSC")
        self.result_ = self.capacity_
        return self

    @classmethod
    def compute(cls, crossover_prob=0.1):
        return cls(crossover_prob=crossover_prob).fit().result_


class HuffmanCode:
    """Builds an optimal prefix-free Huffman code from symbol probabilities::

        hc = HuffmanCode().fit(probs=[.4, .3, .2, .1])
        hc.code_table_     # {0: '00', 1: '01', ...}
        hc.average_length_ # weighted average code length
        hc.is_optimal_     # within [H, H+1] bound
    """

    def __init__(self):
        self.code_table_ = None
        self.average_length_ = None

    def fit(self, probs):
        probs_v, _ = _validate_probs(np.asarray(probs, dtype=float))
        n = len(probs_v)
        if n < 2:
            self.code_table_ = {0: "0"}
            self.average_length_ = 1.0
            self.is_optimal_ = True
            return self

        # build Huffman tree using heap of (prob, node_id, tree)
        heap = []
        node_id = [0]
        for i in range(n):
            node_id[0] += 1
            heapq.heappush(heap, (probs_v[i], node_id[0], ("leaf", i)))

        while len(heap) > 1:
            p1, _, t1 = heapq.heappop(heap)
            p2, _, t2 = heapq.heappop(heap)
            node_id[0] += 1
            heapq.heappush(heap, (p1 + p2, node_id[0], ("node", t1, t2)))

        # traverse to assign codes
        codes = {}

        def assign(tree, prefix=""):
            if tree[0] == "leaf":
                codes[tree[1]] = prefix or "0"
                return 1.0
            return max(assign(tree[1], prefix + "0"),
                       assign(tree[2], prefix + "1"))

        assign(heap[0][2])
        self.code_table_ = dict(sorted(codes.items()))
        lengths = np.array([len(self.code_table_[i]) for i in range(n)])
        self.average_length_ = float(np.sum(probs_v * lengths))
        h = -float(np.sum(probs_v[probs_v > 0] *
                          np.log2(probs_v[probs_v > 0])))
        self.entropy_ = h
        self.is_optimal_ = h <= self.average_length_ <= h + 1
        return self


class TypicalSet:
    """Typical-set membership test under the AEP definition.

    A sequence x^n is ε-typical if |−(1/n) log P(x^n) − H(X)| ≤ ε.
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def fit(self, probs):
        """``probs``: probability of each symbol in the alphabet."""
        self.probs_ = _normalise(probs)
        self.entropy_ = -float(np.sum(
            self.probs_[self.probs_ > 0] *
            np.log2(self.probs_[self.probs_ > 0])))
        return self

    def is_typical(self, sequence):
        """Check whether ``sequence`` (list of symbol indices) is typical."""
        seq = np.asarray(sequence, dtype=int)
        n = len(seq)
        if n == 0:
            return False
        log_prob = float(np.sum(_safe_log2(
            np.maximum(self.probs_[seq], 1e-300))))
        empirical_entropy = -log_prob / n
        return abs(empirical_entropy - self.entropy_) <= self.epsilon

    @classmethod
    def compute(cls, probs, epsilon=0.1):
        return cls(epsilon=epsilon).fit(probs)


class AEP:
    """Asymptotic Equipartition Property: typical set size and probability."""

    def __init__(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def fit(self, probs, block_length=100):
        """Compute typical-set bounds for sequences of length ``block_length``
        drawn from the distribution ``probs``."""
        self.probs_ = _normalise(probs)
        self.block_length_ = int(block_length)
        self.epsilon_ = self.epsilon
        self.entropy_ = Entropy.compute(self.probs_)
        # bounds on typical-set size (Cover & Thomas Thm 3.1.2):
        lo = (1 - self.epsilon) * self.block_length_ * self.entropy_
        hi = self.block_length_ * self.entropy_
        self.typical_set_size_lower_ = 2 ** lo
        self.typical_set_size_upper_ = 2 ** hi
        # probability that a random sequence is typical >= 1 - epsilon
        self.typical_set_probability_lower_ = 1.0 - self.epsilon
        return self