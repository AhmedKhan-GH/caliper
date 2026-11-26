"""Test observer for dictionary validation order."""
from weakref import WeakKeyDictionary

import pytest

from caliper.interp._intervene import InterventionService


@pytest.fixture
def dictionary_checks(monkeypatch):
    counts = WeakKeyDictionary()
    matches = InterventionService._dictionary_matches

    def checked(executor):
        try:
            return matches(executor)
        finally:
            counts[executor] = counts.get(executor, 0) + 1

    monkeypatch.setattr(InterventionService, "_dictionary_matches", checked)
    return lambda executor: counts.get(executor, 0)
