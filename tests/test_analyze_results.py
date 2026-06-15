import pytest
import numpy
import math
import analyze_results


def test_parse_log_valid_data(tmp_path):
    log_content = """## --- 4096x4096x4096 ---
some metal compile warnings or standard stdout
myGEMM.cl: 0.015 s --> 160.00 GFLOPS
## --- 8192x8192x8192 ---
myGEMM.cl: 0.120 s --> 161.32 GFLOPS"""

    fake_log = tmp_path / "run_1.log"
    fake_log.write_text(log_content)

    results = analyze_results.parse_log(str(fake_log))

    assert "4096x4096x4096" in results
    assert "8192x8192x8192" in results
    assert results["4096x4096x4096"] == pytest.approx(160.00)
    assert results["8192x8192x8192"] == pytest.approx(161.32)


def test_parse_log_corrupted_data(tmp_path):
    log_content = """## --- INVALID_HEADER ---
myGEMM.cl: broken line without metrics or timing data"""

    fake_log = tmp_path / "run_broken.log"
    fake_log.write_text(log_content)

    results = analyze_results.parse_log(str(fake_log))
    assert results == {}


def test_parse_log_multiple_same_sizes(tmp_path):
    log_content = """## --- 1024x1024x1024 ---
myGEMM.cl: 0.010 s --> 100.00 GFLOPS
## --- 1024x1024x1024 ---
myGEMM.cl: 0.009 s --> 112.00 GFLOPS"""

    fake_log = tmp_path / "run_dup.log"
    fake_log.write_text(log_content)

    results = analyze_results.parse_log(str(fake_log))
    assert results["1024x1024x1024"] == pytest.approx(112.00)


def test_ci_for_std_bounds():
    data = [155.0, 162.0, 159.0, 161.0, 158.0, 160.0, 163.0, 157.0]

    lower, upper = analyze_results.ci_for_std(data, alpha=0.05)
    actual_std = numpy.std(data, ddof=1)

    assert lower < actual_std < upper
    assert lower > 0


def test_compute_stats_zero_variance():
    perfect_data = [500.0] * 10

    mean, std, std_ci, mean_ci, shap_p, dag_p = analyze_results.compute_stats(
        perfect_data
    )

    assert mean == 500.0
    assert std == 0.0
    assert std_ci == (0.0, 0.0)
    assert mean_ci == (500.0, 500.0)


def test_compute_stats_insufficient_data_handling():
    small_data = [510.0, 512.0, 508.0, 515.0, 511.0]

    mean, std, std_ci, mean_ci, shap_p, dag_p = analyze_results.compute_stats(
        small_data
    )

    assert mean == pytest.approx(511.2)
    assert math.isnan(dag_p)
    assert not math.isnan(shap_p)


def test_compute_stats_sample_size_influence():
    data_small = [200.0, 205.0, 195.0, 202.0, 198.0]
    data_large = data_small * 5

    _, _, _, mean_ci_small, _, _ = analyze_results.compute_stats(data_small)
    _, _, _, mean_ci_large, _, _ = analyze_results.compute_stats(data_large)

    len_small = mean_ci_small[1] - mean_ci_small[0]
    len_large = mean_ci_large[1] - mean_ci_large[0]

    assert len_large < len_small
