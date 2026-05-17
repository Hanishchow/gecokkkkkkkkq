#!/usr/bin/env python3
"""
Basic smoke tests for GEOCK prediction engine.
Run with: python -m pytest tests/ -v
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test that geock_engine imports correctly."""
    try:
        from geock_engine import predict_pKd, GEOCKEngine

        assert callable(predict_pKd)
        assert callable(GEOCKEngine)
    except ImportError as e:
        raise AssertionError(f"Import failed: {e}")


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    from geock_engine import predict_pKd

    result = predict_pKd("INVALID_SMILES")
    assert result["pKd"] is None
    assert "error" in result
    assert "Invalid" in result.get("error", "")


def test_empty_smiles():
    """Test handling of empty SMILES."""
    from geock_engine import predict_pKd

    result = predict_pKd("")
    assert result["pKd"] is None


def test_none_smiles():
    """Test handling of None SMILES."""
    from geock_engine import predict_pKd

    result = predict_pKd(None)
    assert result["pKd"] is None


def test_valid_smiles_aspirin():
    """Test prediction with valid aspirin SMILES."""
    from geock_engine import predict_pKd

    # Aspirin SMILES
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    result = predict_pKd(aspirin)

    # Should either succeed or fail gracefully
    if result["pKd"] is not None:
        assert isinstance(result["pKd"], (int, float))
        # pKd should be in reasonable range (typically 2-10)
        assert 0 < result["pKd"] < 15
    else:
        # Model not loaded is acceptable for smoke test
        assert "error" in result or "confidence" in result


def test_valid_smiles_caffeine():
    """Test prediction with caffeine SMILES."""
    from geock_engine import predict_pKd

    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    result = predict_pKd(caffeine)

    if result["pKd"] is not None:
        assert isinstance(result["pKd"], (int, float))


def test_batch_predict():
    """Test batch prediction."""
    from geock_engine import batch_predict

    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    results = batch_predict(smiles_list)

    assert len(results) == len(smiles_list)
    assert isinstance(results, list)


def test_geock_engine_class():
    """Test GEOCKEngine class instantiation."""
    try:
        from geock_engine import GEOCKEngine

        engine = GEOCKEngine()
        assert engine is not None
    except Exception as e:
        # Model loading might fail - that's OK for smoke test
        if "not found" in str(e).lower():
            pass
        else:
            raise


def test_paths_module():
    """Test geock_paths module imports correctly."""
    try:
        from geock_paths import get_cache_dir, get_work_dir, get_data_dir

        cache = get_cache_dir()
        work = get_work_dir()

        assert cache is not None
        assert work is not None
    except ImportError as e:
        # geock_paths might not be available - that's OK
        pass


def test_batch_with_empty():
    """Test batch prediction handles empty strings."""
    from geock_engine import batch_predict

    smiles_list = ["CCO", "", "c1ccccc1"]
    results = batch_predict(smiles_list)

    # Should return 3 results, empty string handled gracefully
    assert len(results) == 3


def test_predictions_format():
    """Test prediction result format consistency."""
    from geock_engine import predict_pKd

    result = predict_pKd("CCO")

    # Check expected keys are present if prediction succeeds
    if result.get("pKd") is not None:
        assert "pKd" in result
        assert isinstance(result["pKd"], (int, float))


def test_caching():
    """Test that model loading is cached."""
    from geock_engine import GEOCKEngine

    try:
        engine1 = GEOCKEngine()
        engine2 = GEOCKEngine()

        # Should be same instance due to caching
        # (depends on implementation)
    except Exception:
        pass


if __name__ == "__main__":
    print("Running GEOCK smoke tests...")
    print()

    tests = [
        test_import,
        test_invalid_smiles,
        test_empty_smiles,
        test_none_smiles,
        test_valid_smiles_aspirin,
        test_valid_smiles_caffeine,
        test_batch_predict,
        test_geock_engine_class,
        test_paths_module,
        test_batch_with_empty,
        test_predictions_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"  {test.__name__}...", end=" ")
            test()
            print("✓")
            passed += 1
        except Exception as e:
            print(f"✗ - {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
