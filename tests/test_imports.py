def test_package_imports():
    import lacunae_analysis

    assert lacunae_analysis.__all__ == ["__version__"]
