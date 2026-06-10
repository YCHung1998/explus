def test_phash_module_imports_without_running():
    # Must not exit() or run inference at import time (fixes #2).
    import batch_infer_phash
    assert hasattr(batch_infer_phash, "main")
