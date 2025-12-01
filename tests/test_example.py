"""Example test file to ensure pytest is working."""


def test_example():
    """Example test case."""
    assert True


def test_addition():
    """Test basic arithmetic."""
    assert 1 + 1 == 2


def test_string_operations():
    """Test string operations."""
    text = "movement chain"
    assert text.upper() == "MOVEMENT CHAIN"
    assert "chain" in text
