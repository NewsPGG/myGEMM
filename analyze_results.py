import re
import glob
import numpy
from scipy import stats
import sys
import os
import csv

def parse_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    results = {}
    current_size = None
    for line in lines:
        if line.startswith('## ---') and 'x' in line:
            m = re.search(r'--- (\d+x\d+x\d+) ---', line)
            if m:
                current_size = m.group(1)
        if 'myGEMM.cl' in line and current_size:
            m = re.search(r'([\d\.]+)\s+s\s+-->\s+([\d\.]+)\s+GFLOPS', line)
            if m:
                gflops = float(m.group(2))
                results[current_size] = gflops
                current_size = None
    return results

def ci_for_std(data, alpha=0.05):
    n = len(data)
    var = numpy.var(data, ddof=1)
    chi2_lower = stats.chi2.ppf(1 - alpha/2, n-1)
    chi2_upper = stats.chi2.ppf(alpha/2, n-1)
    lower = numpy.sqrt((n-1) * var / chi2_lower)
    upper = numpy.sqrt((n-1) * var / chi2_upper)
    return lower, upper

def compute_stats(gflops_list):
    n = len(gflops_list)
    mean = numpy.mean(gflops_list)
    std = numpy.std(gflops_list, ddof=1)
    std_lower, std_upper = ci_for_std(gflops_list)
    sem = std / numpy.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_crit * sem
    ci_upper = mean + t_crit * sem
    shap_p = stats.shapiro(gflops_list)[1] if 3 <= n <= 5000 else numpy.nan
    dag_p = stats.normaltest(gflops_list)[1] if n >= 8 else numpy.nan
    return mean, std, (std_lower, std_upper), (ci_lower, ci_upper), shap_p, dag_p

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    folder = sys.argv[1]
    log_files = glob.glob(os.path.join(folder, "run_*.log"))
    if not log_files:
        print(f"No log files found in {folder}")
        sys.exit(1)

    all_data = {}
    for log in log_files:
        res = parse_log(log)
        for size, g in res.items():
            all_data.setdefault(size, []).append(g)

    rows = []
    for size in sorted(all_data.keys(), key=lambda x: int(x.split('x')[0])):
        gflops = all_data[size]
        mean, std, (std_low, std_high), (ci_low, ci_high), shap_p, dag_p = compute_stats(gflops)
        std_str = f"{std:.0f} ({std_low:.0f}–{std_high:.0f})"
        ci_str = f"{ci_low:.0f}–{ci_high:.0f}"
        print(f"{size:<12} {mean:8.0f} {std_str:>20} {ci_str:>20} {shap_p:12.4f} {dag_p:14.4f}")
        rows.append([size, mean, std, std_low, std_high, ci_low, ci_high, shap_p, dag_p])

    if '--csv' in sys.argv:
        csv_idx = sys.argv.index('--csv') + 1
        if csv_idx < len(sys.argv):
            csv_file = sys.argv[csv_idx]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Size','Mean','Std','StdCI low','StdCI high','MeanCI low','MeanCI high','Shapiro p','Dagostino p'])
                writer.writerows(rows)

if __name__ == "__main__":
    main()