"""Exact-arithmetic combinatorics: factorial, permutations, combinations, and counting sequences.

All functions use Python's arbitrary-precision integers — no floating point.
"""

import math


def factorial(n):
    """n! — number of orderings of n distinct items.

    >>> factorial(5)
    120
    """
    return math.factorial(n)


def permutation(n, k):
    """nPk = n! / (n - k)! — orderings of k items chosen from n.

    >>> permutation(5, 2)
    20
    """
    return math.perm(n, k)


def combination(n, k):
    """nCk = n! / (k! (n - k)!) — subsets of size k chosen from n.

    >>> combination(5, 2)
    10
    """
    return math.comb(n, k)


def multinomial(n, *ks):
    """Multinomial coefficient n! / (k1! k2! ... km!), where sum(ks) == n.

    >>> multinomial(10, 2, 3, 5)
    2520
    """
    if sum(ks) != n:
        raise ValueError(f"group sizes {ks} must sum to n={n}")
    result = factorial(n)
    for k in ks:
        result //= factorial(k)
    return result


def stirling(n, k, kind="second"):
    """Stirling numbers of the first or second kind.

    ``kind="second"`` (default) counts partitions of n labeled items into k non-empty,
    unlabeled subsets. ``kind="first"`` (unsigned) counts permutations of n items with
    exactly k cycles.

    >>> stirling(4, 2)
    7
    >>> stirling(4, 2, kind="first")
    11
    """
    if kind == "second":
        if k > n:
            return 0
        if k == 0:
            return 1 if n == 0 else 0
        total = 0
        for j in range(k + 1):
            term = (-1) ** (k - j) * combination(k, j) * j**n
            total += term
        return total // factorial(k)
    if kind == "first":
        if k > n:
            return 0
        if n == 0 and k == 0:
            return 1
        table = [[0] * (k + 1) for _ in range(n + 1)]
        table[0][0] = 1
        for ni in range(1, n + 1):
            for ki in range(1, k + 1):
                table[ni][ki] = table[ni - 1][ki - 1] + (ni - 1) * table[ni - 1][ki]
        return table[n][k]
    raise ValueError("kind must be 'first' or 'second'")


def bell_number(n):
    """Bell number B(n) — total number of partitions of an n-element set.

    >>> bell_number(4)
    15
    """
    return sum(stirling(n, k) for k in range(n + 1))


def catalan_number(n):
    """Catalan number C(n) = (2n)! / ((n+1)! n!).

    >>> catalan_number(4)
    14
    """
    return combination(2 * n, n) // (n + 1)


def derangement(n):
    """Number of permutations of n items with no fixed points.

    Uses the exact recurrence D(n) = (n - 1) * (D(n - 1) + D(n - 2)) rather than the
    floating-point series formula, to keep results exact for large n.

    >>> derangement(4)
    9
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    d_prev2, d_prev1 = 1, 0
    for i in range(2, n + 1):
        d_prev2, d_prev1 = d_prev1, (i - 1) * (d_prev1 + d_prev2)
    return d_prev1
