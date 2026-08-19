"""Per-language chunking coverage.

One realistic snippet per registry language, asserting the symbols the chunker
must find and the invariant that no content line is dropped. Snippets live
inline rather than as fixture files so a language's expectations sit next to
the code that produces them.
"""

from __future__ import annotations

import pytest

from src import config
from src.ingest import chunk_source

# (extension, source, symbols that must appear as function/class/type chunks)
CASES: list[tuple[str, str, set[str]]] = [
    (
        ".py",
        '''
import os

@staticmethod
def decorated(a):
    return a

class Account:
    def interest(self, years):
        return years
''',
        {"decorated", "Account", "Account.interest"},
    ),
    (
        ".js",
        '''
export const arrow = (a) => a + 1;

export function plain(x) { return x; }

export class Widget {
  render() { return null; }
}
''',
        {"arrow", "plain", "Widget", "Widget.render"},
    ),
    (
        ".ts",
        '''
export interface Rate { value: number; }

export type Id = string;

export enum Color { Red }

export const handler = async (req: string): Promise<void> => {};

export class Service {
  run(): void {}
}
''',
        {"Rate", "Id", "Color", "handler", "Service", "Service.run"},
    ),
    (
        ".tsx",
        '''
export const Button = (props: Props) => <button>{props.label}</button>;

export interface Props { label: string; }
''',
        {"Button", "Props"},
    ),
    (
        ".go",
        '''
package main

type Account struct {
	Balance float64
}

type Payer interface {
	Pay(amount float64) error
}

func (a *Account) Interest(years int) float64 {
	return a.Balance
}

func Simple(v int) int { return v * 2 }
''',
        # Go declares methods at top level, so Interest is qualified by its
        # receiver type rather than by nesting.
        {"Account", "Payer", "Account.Interest", "Simple"},
    ),
    (
        ".rs",
        '''
pub struct Account {
    balance: f64,
}

pub trait Payer {
    fn pay(&self, amount: f64);
}

impl Account {
    pub fn interest(&self, years: u32) -> f64 {
        self.balance
    }
}

fn simple(v: i32) -> i32 { v * 2 }
''',
        {"Account", "Payer", "Account.interest", "simple"},
    ),
    (
        ".java",
        '''
package com.example;

public interface Payer {
    void pay(double amount);
}

public class Account {
    private double balance;

    public double interest(int years) {
        return balance;
    }
}
''',
        {"Payer", "Account", "Account.interest"},
    ),
    (
        ".cs",
        '''
namespace Demo {
    public class Account {
        private double balance;

        public double Interest(int years) {
            return balance;
        }
    }
}
''',
        {"Demo", "Demo.Account", "Demo.Account.Interest"},
    ),
    (
        ".rb",
        '''
RATE = 0.05

class Account
  def interest(years)
    @balance * years
  end

  def self.create
    new
  end
end

def simple(v)
  v * 2
end
''',
        {"Account", "Account.interest", "Account.create", "simple"},
    ),
    (
        ".c",
        '''
#include <stdio.h>

struct Account {
    double balance;
};

enum Color { RED, GREEN };

double interest(struct Account *a, int years) {
    return a->balance * years;
}
''',
        {"Account", "Color", "interest"},
    ),
    (
        ".cpp",
        '''
class Account {
public:
    double interest(int years);
private:
    double balance_;
};

double Account::interest(int years) {
    return balance_ * years;
}
''',
        {"Account", "Account::interest"},
    ),
    (
        ".php",
        '''<?php
interface Payer {
    public function pay(float $amount): void;
}

class Account {
    private float $balance;

    public function interest(int $years): float {
        return $this->balance * $years;
    }
}

function simple(int $v): int { return $v * 2; }
''',
        {"Payer", "Account", "Account.interest", "simple"},
    ),
]

IDS = [ext.lstrip(".") for ext, _, _ in CASES]


def _chunks(ext: str, source: str):
    spec = config.LANGUAGE_REGISTRY[ext]
    return chunk_source(source.encode("utf-8"), f"sample{ext}", spec)


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_expected_symbols_are_found(ext: str, source: str, expected: set[str]) -> None:
    found = {c.symbol for c in _chunks(ext, source) if c.kind != "module"}
    missing = expected - found
    assert not missing, f"{ext}: missing {sorted(missing)}; found {sorted(found)}"


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_no_content_line_is_dropped(ext: str, source: str, expected: set[str]) -> None:
    chunks = _chunks(ext, source)
    lines = source.split("\n")
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    dropped = [
        number
        for number, text in enumerate(lines, start=1)
        if text.strip() and number not in covered
    ]
    assert not dropped, f"{ext}: dropped lines {dropped}"


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_no_anonymous_symbols(ext: str, source: str, expected: set[str]) -> None:
    """Every named construct in these snippets should resolve to a real name."""
    anonymous = [c.location for c in _chunks(ext, source) if c.symbol == "<anonymous>"]
    assert not anonymous, f"{ext}: unnamed chunks at {anonymous}"


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_no_structural_only_chunks(ext: str, source: str, expected: set[str]) -> None:
    """A chunk whose whole body is `}` or `end` is index noise, not content."""
    junk = [
        c.location
        for c in _chunks(ext, source)
        if all(
            not line.strip() or line.strip() in {"end", "})", "});"} or set(line.strip()) <= set("{}()[];,")
            for line in c.code.split("\n")
        )
    ]
    assert not junk, f"{ext}: structural-only chunks at {junk}"


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_symbols_are_not_duplicated(ext: str, source: str, expected: set[str]) -> None:
    """No construct should be chunked twice.

    Keyed on (kind, symbol) rather than name alone: Rust genuinely declares
    `struct Account` and `impl Account` separately, and both are real.
    """
    named = [(c.kind, c.symbol) for c in _chunks(ext, source) if c.kind != "module"]
    assert len(named) == len(set(named)), f"{ext}: duplicate chunks in {named}"


@pytest.mark.parametrize(("ext", "source", "expected"), CASES, ids=IDS)
def test_metadata_is_complete(ext: str, source: str, expected: set[str]) -> None:
    spec = config.LANGUAGE_REGISTRY[ext]
    for chunk in _chunks(ext, source):
        assert chunk.language == spec.name
        assert chunk.file_path == f"sample{ext}"
        assert chunk.code
        assert chunk.symbol
        assert 1 <= chunk.start_line <= chunk.end_line


def test_every_registry_language_has_a_case() -> None:
    """A new registry entry without a test case should fail loudly."""
    covered = {ext for ext, _, _ in CASES}
    languages_covered = {config.LANGUAGE_REGISTRY[e].name for e in covered}
    all_languages = {s.name for s in config.LANGUAGE_REGISTRY.values()}
    assert all_languages == languages_covered, (
        f"languages without a chunking test: {sorted(all_languages - languages_covered)}"
    )


def test_go_methods_are_qualified_by_receiver() -> None:
    """Two types with a same-named method must not collide."""
    source = '''
package main

type Account struct{ Balance float64 }

type Ledger struct{ Total float64 }

func (a *Account) Total() float64 { return a.Balance }

func (l Ledger) Total() float64 { return l.Total }
'''
    symbols = {c.symbol for c in _chunks(".go", source) if c.kind == "function"}
    assert symbols == {"Account.Total", "Ledger.Total"}


def test_every_grammar_loads() -> None:
    """Catches a registry entry whose wheel is missing or entry point is wrong."""
    from src.ingest.chunker import get_parser

    for spec in set(config.LANGUAGE_REGISTRY.values()):
        get_parser(spec.grammar_module, spec.grammar_entrypoint)
