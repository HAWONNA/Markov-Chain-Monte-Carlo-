import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import datetime
from scipy.stats import beta
import os
import matplotlib.pyplot as plt

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
start = datetime.datetime(2005, 1, 1)
end   = datetime.datetime(2025, 1, 1)

# S&P500 가격 (Close만 → SP500으로 고정)
sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True)[["Close"]]
if isinstance(sp500.columns[0], tuple):
    sp500.columns = ["SP500"]
else:
    sp500 = sp500.rename(columns={"Close": "SP500"})

# 금리 데이터 (FRED)
rate_codes = ["DGS2", "DGS10", "DGS30"]
rate_names = ["2Y", "10Y", "30Y"]

rates_list = []
for code, name in zip(rate_codes, rate_names):
    r = pdr.FredReader(code, start=start, end=end).read()
    r = r.rename(columns={code: name})
    rates_list.append(r)

rates = pd.concat(rates_list, axis=1)

# 병합
data = pd.concat([sp500, rates], axis=1).dropna()

# ---------------------------
# 2. 수익률 및 금리 변화 계산
# ---------------------------
data["SPX_ret"] = np.log(data["SP500"]).diff()
for name in ["2Y", "10Y", "30Y"]:
    data[f"{name}_chg"] = data[name].diff()
data = data.dropna()

# ---------------------------
# 3. Empirical Bayes Prior 추정
# ---------------------------
def estimate_prior_from_data(success, total):
    if total == 0:
        return 1, 1
    p_hat = success / total
    var_hat = p_hat * (1 - p_hat) / max(total, 1)
    if var_hat == 0:
        return 1, 1
    common = p_hat * (1 - p_hat) / var_hat - 1
    alpha = max(p_hat * common, 1e-3)
    beta_ = max((1 - p_hat) * common, 1e-3)
    return alpha, beta_

def bayes_analysis(rate_chg, spx_ret, rate_case="up", spx_case="up"):
    if rate_case == "up":
        mask = rate_chg > 0
    else:
        mask = rate_chg < 0

    if spx_case == "up":
        success = ((mask) & (spx_ret > 0)).sum()
    else:
        success = ((mask) & (spx_ret < 0)).sum()

    total = mask.sum()
    s, n = success, total

    alpha, beta_ = estimate_prior_from_data(s, n)
    a_post = alpha + s
    b_post = beta_ + (n - s)

    samples = beta.rvs(a_post, b_post, size=200000)
    mean_est = samples.mean()
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

    return {
        "success": s, "total": n,
        "prior": (alpha, beta_),
        "posterior_mean": mean_est,
        "ci_low": ci_low, "ci_high": ci_high,
        "samples": samples
    }

# ---------------------------
# 4. 5년 단위 구간 설정
# ---------------------------
periods = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31")
]

# ---------------------------
# 5. 구간별 MCMC 결과 저장 + Plot 저장
# ---------------------------
records = []
base_outdir = "results_by_period"
os.makedirs(base_outdir, exist_ok=True)

for (start_date, end_date) in periods:
    period_label = f"{start_date[:4]}–{end_date[:4]}"
    outdir = os.path.join(base_outdir, period_label)
    os.makedirs(outdir, exist_ok=True)

    sub = data.loc[start_date:end_date]

    for rate in ["2Y", "10Y", "30Y"]:
        for rate_case in ["up", "down"]:
            for spx_case in ["up", "down"]:
                res = bayes_analysis(sub[f"{rate}_chg"], sub["SPX_ret"],
                                     rate_case=rate_case, spx_case=spx_case)
                # 결과 기록
                records.append({
                    "Period": period_label,
                    "Rate": rate,
                    "Rate_case": rate_case,
                    "SPX_case": spx_case,
                    "N": res["total"],
                    "Success": res["success"],
                    "Prior_alpha": res["prior"][0],
                    "Prior_beta": res["prior"][1],
                    "Posterior_mean": res["posterior_mean"],
                    "CI_low": res["ci_low"],
                    "CI_high": res["ci_high"]
                })

                # 플롯 저장
                plt.figure(figsize=(6,4))
                plt.hist(res["samples"][::100], bins=50, density=True, alpha=0.7, color="steelblue")
                plt.title(f"{period_label} | {rate} - Rate {rate_case} / SPX {spx_case}")
                plt.xlabel("θ")
                plt.ylabel("Density")
                plt.axvline(res["posterior_mean"], color="red", linestyle="--", label=f"Mean={res['posterior_mean']:.3f}")
                plt.axvline(res["ci_low"], color="green", linestyle="--", label=f"2.5%={res['ci_low']:.3f}")
                plt.axvline(res["ci_high"], color="green", linestyle="--", label=f"97.5%={res['ci_high']:.3f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{outdir}/{rate}_{rate_case}_SPX{spx_case}.png")
                plt.close()

# ---------------------------
# 6. 결과 저장
# ---------------------------
results_df = pd.DataFrame(records)
results_df.to_csv("bayes_results_by_period.csv", index=False)
print("✅ 결과가 bayes_results_by_period.csv 및 results_by_period/<기간> 폴더에 저장되었습니다.")
print(results_df.head())
