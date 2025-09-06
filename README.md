# MCMC Bayes: Yields vs. S&P 500 (2005–2024)

미국 국채 **2Y / 10Y / 30Y** 금리 변화와 **S&P 500** 지수의 **방향성 관계**를 5년 구간으로 나눠 **베이지안(Beta–Binomial)** 모형으로 추정합니다.  
**Posterior mean**과 **95% Credible Interval**을 표/그림으로 저장합니다.

> **요약**: 2005–2019에는 금리와 주가가 **동행**(금리↑→SPX↑ 60–75%)하는 경향이 강했고, 2020–2024에는 관계가 **붕괴**되어 50% 내외 혼조가 나타났습니다.

---

## 📁 프로젝트 구조

```
.
├── results_by_period/
│   ├── 2005–2009/      # 각 조합별 posterior 히스토그램 PNG
│   ├── 2010–2014/
│   ├── 2015–2019/
│   └── 2020–2024/
├── bayes_results_by_period.csv  # 5년×(만기×조합) 결과 테이블
├── main.py
├── requirements.txt
└── .gitignore
```

**권장 `.gitignore`**
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.ipynb_checkpoints/

# Virtual env
.venv/
venv/

# Secrets
.env

# OS
.DS_Store

# Generated plots (용량 커지면 제외 권장)
results_by_period/**/*.png
```

---

## 🔧 설치

```bash
# 1) 가상환경 (선택)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) 패키지 설치
pip install -r requirements.txt
# (pandas, numpy, matplotlib, yfinance, pandas_datareader, scipy 등)
```

> 참고: 최신 `yfinance`는 `auto_adjust=True` 기본값입니다.

---

## ▶️ 실행

```bash
python main.py
```

- 결과 CSV: `bayes_results_by_period.csv`  
- 각 구간 폴더에 **12개 조합(만기 3 × 금리↑/↓ 2 × SPX↑/↓ 2)** 의 Posterior 히스토그램 **PNG 저장**

---

## 🧠 방법론 (수식)

### 변수 정의

주가 로그수익률

$$
r_t^{\mathrm{SPX}}=\ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

금리 변화(만기 $k\in\{2Y,10Y,30Y\}$)

$$
\Delta y_t^{(k)}=y_t^{(k)}-y_{t-1}^{(k)}
$$

케이스 정의

$$
\text{금리 상승일: }\Delta y_t^{(k)} > 0,\qquad
\text{금리 하락일: }\Delta y_t^{(k)} < 0
$$

$$
\text{SPX 상승일: }r_t^{\mathrm{SPX}} > 0,\qquad
\text{SPX 하락일: }r_t^{\mathrm{SPX}} < 0
$$

---

### 베타–베르누이 모형

금리 **상승(또는 하락)** 조건에서 SPX **상승** 여부를 성공(1)/실패(0) 베르누이 시행으로 모델링:

$$
X_i\sim\mathrm{Bernoulli}(\theta),\quad i=1,\dots,N,\qquad
S=\sum_{i=1}^{N}X_i
$$

우도(Likelihood)

$$
L(\theta\mid x)=\theta^{S}(1-\theta)^{N-S}
$$

사전분포(Prior)

$$
\theta\sim\mathrm{Beta}(\alpha_0,\beta_0)
\quad\text{(기본: }\mathrm{Beta}(1,1)\text{)}
$$

사후분포(Posterior)

$$
\theta\mid\text{data}\sim\mathrm{Beta}(\alpha_0+S,\ \beta_0+N-S)
$$

사후 기댓값/분산

$$
\mathbb{E}[\theta\mid\text{data}]=\frac{\alpha_0+S}{\alpha_0+\beta_0+N}
$$

$$
\mathrm{Var}[\theta\mid\text{data}]
=\frac{(\alpha_0+S)(\beta_0+N-S)}{(\alpha_0+\beta_0+N)^2(\alpha_0+\beta_0+N+1)}
$$

---


---

## 5년 구간별 결과 (Posterior mean \[95% CI\])

### 2005–2009
| Rate | 금리↑→SPX↑ | 금리↑→SPX↓ | 금리↓→SPX↑ | 금리↓→SPX↓ |
|---|---|---|---|---|
| 2Y | **65.5% [62.8–68.2]** | 34.5% [31.8–37.2] | 43.3% [40.3–46.3] | **56.5% [53.5–59.5]** |
| 10Y | **62.5% [59.6–65.2]** | 37.5% [34.8–40.3] | 46.4% [43.5–49.2] | **53.6% [50.8–56.5]** |
| 30Y | **58.6% [55.7–61.4]** | 41.3% [38.4–44.2] | 48.9% [46.0–51.8] | **51.1% [48.2–54.0]** |

### 2010–2014
| Rate | 금리↑→SPX↑ | 금리↑→SPX↓ | 금리↓→SPX↑ | 금리↓→SPX↓ |
|---|---|---|---|---|
| 2Y | **65.8% [62.6–68.9]** | 34.2% [31.0–37.4] | 43.5% [40.3–46.8] | **56.5% [53.2–59.7]** |
| 10Y | **74.4% [71.8–77.0]** | 25.6% [23.1–28.2] | 38.0% [35.3–40.7] | **62.0% [59.3–64.7]** |
| 30Y | **75.0% [72.4–77.5]** | 25.0% [22.5–27.6] | 38.8% [36.1–41.6] | **61.2% [58.4–63.9]** |

### 2015–2019
| Rate | 금리↑→SPX↑ | 금리↑→SPX↓ | 금리↓→SPX↑ | 금리↓→SPX↓ |
|---|---|---|---|---|
| 2Y | **65.7% [62.8–68.5]** | 34.3% [31.5–37.2] | 39.2% [36.2–42.4] | **60.5% [57.4–63.6]** |
| 10Y | **67.2% [64.3–69.9]** | 32.8% [30.1–35.7] | 39.3% [36.5–42.1] | **60.7% [57.8–63.5]** |
| 30Y | **66.9% [64.0–69.7]** | 33.1% [30.3–35.9] | 42.2% [39.4–45.1] | **57.8% [54.9–60.6]** |

### 2020–2024
| Rate | 금리↑→SPX↑ | 금리↑→SPX↓ | 금리↓→SPX↑ | 금리↓→SPX↓ |
|---|---|---|---|---|
| 2Y | 52.5% [49.5–55.5] | 47.5% [44.5–50.5] | 52.5% [49.3–55.6] | 47.5% [44.3–50.7] |
| 10Y | 54.3% [51.4–57.1] | 45.7% [42.9–48.6] | 54.5% [51.6–57.4] | 45.5% [42.6–48.4] |
| 30Y | 55.1% [52.3–57.9] | 44.9% [42.1–47.7] | 53.3% [50.4–56.1] | 46.7% [43.8–49.6] |

---

## ⚙️ 설정 변경 팁

`main.py` 상단의 기간/구간/샘플 수를 조정하세요.
```python
start = datetime.datetime(2005, 1, 1)
end   = datetime.datetime(2025, 1, 1)

periods = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

SAMPLES = 200_000
```

---

## 데이터 출처
- **S&P 500**: Yahoo Finance (`^GSPC`, `yfinance`)  
- **금리**: FRED – DGS2, DGS10, DGS30 (`pandas_datareader`)

---

## 디스클레이머
연구/교육 목적의 코드입니다. 투자 자문이 아니며, 과거 성과는 미래를 보장하지 않습니다.

