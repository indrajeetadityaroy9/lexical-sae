"""Shared fixtures for tests."""

import pytest


@pytest.fixture
def small_texts():
    return ["This is great", "This is terrible", "Amazing movie", "Awful film"]


@pytest.fixture
def small_labels():
    return [1, 0, 1, 0]


@pytest.fixture
def sample_attributions():
    """Attributions sorted by weight descending (as produced by explain())."""
    return [
        [("great", 0.9), ("amazing", 0.7), ("movie", 0.3), ("this", 0.1)],
        [("terrible", 0.8), ("awful", 0.6), ("film", 0.2), ("is", 0.05)],
    ]
