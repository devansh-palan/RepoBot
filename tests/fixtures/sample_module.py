"""A sample module used as a chunker fixture.

Deliberately contains every shape the chunker has to handle: module-level
constants, a decorated function, a class with a docstring, methods, a trailing
class attribute, a nested closure, and an async function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_RATE = 0.05
MAX_RETRIES = 3


def simple(value: int) -> int:
    """A plain function with no decorator."""
    return value * 2


@dataclass(frozen=True)
class Account:
    """A bank account with a balance and an interest rate."""

    balance: float
    rate: float = DEFAULT_RATE

    def interest(self, years: int) -> float:
        """Compound interest over `years`, ignoring deposits."""
        return self.balance * (math.pow(1 + self.rate, years) - 1)

    def with_deposit(self, amount: float) -> "Account":
        if amount < 0:
            raise ValueError("deposit must be non-negative")
        return Account(self.balance + amount, self.rate)

    CURRENCY = "USD"


def make_counter(start: int = 0):
    """Returns a closure; the closure must stay inside this chunk."""
    count = start

    def increment(step: int = 1) -> int:
        nonlocal count
        count += step
        return count

    return increment


async def fetch_rate(client, symbol: str) -> float:
    response = await client.get(f"/rates/{symbol}")
    return float(response["rate"])


if __name__ == "__main__":
    print(simple(21))
