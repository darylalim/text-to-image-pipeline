def pytest_configure(config):
    """Suppress warnings from third-party dependency imports on non-CUDA machines."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:User provided device_type of 'cuda':UserWarning",
    )
