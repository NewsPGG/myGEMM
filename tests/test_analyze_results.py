import pytest
import numpy
import math
import analyze_results


def test_parse_log_valid_data(tmp_path):
    log_content = """
    ## --- 1024x1024x1024 ---
    some compile info or metal warnings
    myGEMM.cl: 0.015 s --> 145.32 GFLOPS
    ## --- 2048x2048x2048 ---
    myGEMM.cl: 0.120 s --> 320.50 GFLOPS
    """

    fake_log = tmp_path / "run_1.log"
    fake_log.write_text(log_content)

    results = analyze_results.parse_log(str(fake_log))

    assert "1024x1024x1024" in results
    assert "2048x2048x2048" in results
    assert results["1024x1024x1024"] == 145.32
    assert results["2048x2048x2048"] == 320.50


def test_parse_log_missing_size(tmp_path):
    log_content = """
    ## --- BROKEN_HEADER ---
    myGEMM.cl: 0.015 s --> 145.32 GFLOPS
    """

    fake_log = tmp_path / "run_broken.log"
    fake_log.write_text(log_content)

    results = analyze_results.parse_log(str(fake_log))
    assert results == {}


def test_ci_for_std_bounds():
    numpy.random.seed(42)
    data = numpy.random.normal(loc=500, scale=10, size=30)

    lower, upper = analyze_results.ci_for_std(data, alpha=0.05)
    actual_std = numpy.std(data, ddof=1)

    assert lower < actual_std < upper
    assert lower > 0


def test_compute_stats_nan_handling():
    small_data = [100.0, 102.0, 101.0, 99.0, 100.0]

    mean, std, std_ci, mean_ci, shap_p, dag_p = analyze_results.compute_stats(
        small_data
    )

    assert mean == pytest.approx(100.4)
    assert math.isnan(dag_p)
    assert not math.isnan(shap_p)
