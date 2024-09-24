import pytest
from unittest.mock import patch
import importlib


@pytest.fixture
def mock_version():
    """Fixture to mock "importlib.metadata.version" before importing the module."""
    with patch("importlib.metadata.version") as mock_ver:
        yield mock_ver


def test_version_to_int():
    """Test the _version_to_int function with standard version strings."""
    from prompting import _version_to_int
    assert _version_to_int("1.2.3") == 10203, "Standard version conversion failed."
    assert _version_to_int("0.0.1") == 1, "Zero major/minor with patch failed."
    assert _version_to_int("10.20.30") == 102030, "Double-digit version parts failed."
    assert _version_to_int("3.0") == 30000, "Missing patch number failed."
    assert _version_to_int("4") == 40000, "Only major version provided failed."


def test_version_to_int_invalid_strings():
    """Test the _version_to_int function with invalid version strings."""
    from prompting import _version_to_int
    with pytest.raises(ValueError):
        _version_to_int("invalid.version.string")

    # ValueError: invalid literal for int() with base 10: ''
    with pytest.raises(ValueError):
        _version_to_int("")


def test_version_and_spec_version(mock_version):
    """Test that __version__ and __spec_version__ are set correctly."""
    mock_version.return_value = "2.5.7"
    # Import the module after mocking
    import prompting
    # Reload to apply mocks
    importlib.reload(prompting)

    assert prompting.__version__ == "2.5.7", "__version__ is not set correctly."
    assert prompting.__spec_version__ == 20507, "__spec_version__ is not computed correctly."


def test_spec_version_with_different_versions(mock_version):
    """Test __spec_version__ with various mocked __version__ values."""
    test_cases = {
        "3.4.5": 30405,
        "0.1.0": 100,
        "10.0": 100000,
        "7": 70000,
    }

    for version_str, expected_spec in test_cases.items():
        mock_version.return_value = version_str
        import prompting
        importlib.reload(prompting)

        assert prompting.__version__ == version_str, f"__version__ should be {version_str}."
        assert prompting.__spec_version__ == expected_spec, (
            f"__spec_version__ should be {expected_spec} for version {version_str}."
        )