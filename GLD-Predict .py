import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
# === 添加这一行：用于残差的正式白噪声检验 ===
from statsmodels.stats.diagnostic import acorr_ljungbox
try:
    from pandas_datareader import data as web
    HAS_FRED = True
except Exception:
    HAS_FRED = False
    print("pandas_datareader/FRED 不可用，RealRate 将被跳过。")

print("ENV VERSIONS -> numpy:", np.__version__)
try:
    import pandas as _pd_check
    print("pandas:", _pd_check.__version__)
except Exception:
    pass
try:
    import yfinance as _yf_check
    print("yfinance:", getattr(_yf_check, "__version__", "unknown"))
except Exception:
    pass
try:
    import matplotlib as _mpl_check
    print("matplotlib:", _mpl_check.__version__)
except Exception:
    pass

# Forecast horizon in trading days (default: 1 day ahead)
H = int(os.environ.get("HORIZON", "1"))
print(f"Forecast horizon H = {H} days")

# 尝试可选依赖 Prophet（没有就跳过）
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    print("Prophet 未安装或不可用，将只运行随机游走基准。要启用请先 pip install prophet")

# yfinance 下载参数（尽量减小被限流概率）

YF_KW = dict(progress=False, threads=False, auto_adjust=True, actions=False, repair=True)

def try_yf(ticker, start=None, end=None, interval="1d",
           max_retries=6, base_sleep=1.0, cache_dir="cache"):
    """yfinance 下载，带指数回退+随机抖动，并写入本地缓存。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_fp = os.path.join(cache_dir, f"yf_{ticker}_{interval}_{start}_{end}.pkl")

    # 命中缓存直接返回
    if os.path.exists(cache_fp):
        try:
            with open(cache_fp, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, "yf-cache"
        except Exception:
            pass

    last_err = None
    for i in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, **YF_KW)
            if not df.empty:
                df.dropna(how="all", inplace=True)
                if not df.empty:
                    with open(cache_fp, "wb") as f:
                        pickle.dump(df, f)
                    return df, "yf"
        except Exception as e:
            last_err = e


        # 二级兜底：改用 Ticker.history 再试一次（有时能绕过 numpy 转换异常）
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end, interval=interval, auto_adjust=True, actions=False, repair=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # yfinance.history 可能返回列为单层，也可能有多层，统一一下
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(0, axis=1)
                df.dropna(how="all", inplace=True)
                if not df.empty:
                    with open(cache_fp, "wb") as f:
                        pickle.dump(df, f)
                    return df, "yf-history"
        except Exception as e2:
            last_err = e2

        # 指数回退 + 抖动（缓解 429）
        sleep_s = (base_sleep * (2 ** i)) + random.uniform(0, 0.5)
        time.sleep(min(sleep_s, 20))  # 单次等待不超过 20s

    raise last_err if last_err else RuntimeError("yfinance 下载失败且无具体异常")


def download_with_fallback(candidates, start="2000-01-01", end=None, interval="1d"):
    """仅使用 yfinance（多次重试+本地缓存）。"""
    last_err = None
    # 第一轮：yfinance
    for t in candidates:
        try:
            df, src = try_yf(t, start=start, end=end, interval=interval)
            return t, df, src
        except Exception as e:
            print(f"yfinance: {t} 失败：{e}")
            last_err = e
    raise RuntimeError(f"所有数据源均失败，最后错误：{last_err}")


# 入口参数与下载
candidates = ["GLD"]
start_date = "2000-01-01"
interval = "1d"

ticker, df, src = download_with_fallback(candidates, start=start_date, end=None, interval=interval)
print(f"使用数据源：{src}，ticker：{ticker}，样本数：{len(df)}")

# 选择价格列：优先 Adj Close，没有就用 Close
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
price_series = df[price_col].dropna()  # 原始价格
log_series = np.log(price_series)      # 对价格取对数（用于 ARIMA / ARIMAX）

# =========================
# Figure 1: Full sample time series (Data section) - 原始价格
# =========================
plt.figure()
plt.plot(price_series.index, price_series.values, label=f'{ticker} price')
plt.title(f'{ticker} full sample price series')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# 基础检查与切分
n = len(price_series)
if n < 2:
    raise ValueError(f"{ticker} 可用数据过少：{n} 行。")

split = int(n * 0.8)

# 价格空间和 log 空间都做 80/20 切分
price_train, price_test = price_series.iloc[:split], price_series.iloc[split:]
log_train, log_test = log_series.iloc[:split], log_series.iloc[split:]
if price_test.empty:
    raise ValueError("测试集为空，请调整时间范围或切分比例。")

# =========================
# 宏观因子：DXY, TNX, VIX, RealRate (DFII10 from FRED)
# =========================
macro_series = {}

# Dollar index (DXY)
try:
    _, dxy_df, _ = download_with_fallback(
        ["DX-Y.NYB", "DXY"], start=start_date, end=None, interval=interval
    )
    dxy_price_col = "Adj Close" if "Adj Close" in dxy_df.columns else "Close"
    macro_series["DXY"] = dxy_df[dxy_price_col].reindex(price_series.index).ffill()
except Exception as e:
    print("DXY 下载失败:", e)

# 10Y nominal yield (TNX)
try:
    _, tnx_df, _ = download_with_fallback(
        ["^TNX"], start=start_date, end=None, interval=interval
    )
    tnx_price_col = "Adj Close" if "Adj Close" in tnx_df.columns else "Close"
    macro_series["TNX"] = tnx_df[tnx_price_col].reindex(price_series.index).ffill()
except Exception as e:
    print("TNX 下载失败:", e)

# VIX index
try:
    _, vix_df, _ = download_with_fallback(
        ["^VIX"], start=start_date, end=None, interval=interval
    )
    vix_price_col = "Adj Close" if "Adj Close" in vix_df.columns else "Close"
    macro_series["VIX"] = vix_df[vix_price_col].reindex(price_series.index).ffill()
except Exception as e:
    print("VIX 下载失败:", e)

# S&P 500 index (^GSPC)
try:
    _, gspc_df, _ = download_with_fallback(
        ["^GSPC"], start=start_date, end=None, interval=interval
    )
    gspc_price_col = "Adj Close" if "Adj Close" in gspc_df.columns else "Close"
    macro_series["GSPC"] = gspc_df[gspc_price_col].reindex(price_series.index).ffill()
except Exception as e:
    print("GSPC 下载失败:", e)

# Real rate: 10-year TIPS real yield (DFII10 from FRED)
if HAS_FRED:
    try:
        rr = web.DataReader("DFII10", "fred", start_date)
        rr_series = rr.iloc[:, 0].reindex(price_series.index).ffill()
        macro_series["RealRate"] = rr_series
    except Exception as e:
        print("RealRate (DFII10) 下载失败:", e)

# 组装外生变量矩阵 exog
exog_list = []
for name in ["DXY", "TNX", "VIX", "RealRate", "GSPC"]:
    if name in macro_series:
        s = macro_series[name].copy()
        s.name = name
        exog_list.append(s)

if exog_list:
    # 原始宏观因子矩阵（水平数据）
    exog = pd.concat(exog_list, axis=1)
    # 对宏观变量做一阶差分，并用 0 填充首行，保持与价格序列长度一致
    exog_diff = exog.diff().fillna(0.0)
    # 使用差分后的宏观变量作为 ARIMAX 的外生输入（与 ts 的 80/20 切分对齐）
    exog_train = exog_diff.iloc[:split]
    exog_test = exog_diff.iloc[split:]
else:
    exog = None
    exog_train = None
    exog_test = None


# =========================
# Figure 1b: Differenced Macro Drivers (Data section)
#   图 1b-1: 仅 RealRate 与 DXY（差分后，标准化）
#   图 1b-2: RealRate, DXY, VIX, TNX, GSPC 全部（差分后，标准化）
# =========================
if macro_series:
    # 将宏观因子合并为一个 DataFrame（水平数据）
    macro_df = pd.concat(macro_series.values(), axis=1)
    macro_df.columns = list(macro_series.keys())
    # 对宏观因子做一阶差分，并去掉首行 NaN
    macro_diff = macro_df.diff().dropna()
    # 按列标准化
    macro_diff_norm = macro_diff.apply(lambda s: (s - s.mean()) / s.std())

    # ---- 图 1b-1: 仅 RealRate 与 DXY ----
    cols_rd = [c for c in ["RealRate", "DXY"] if c in macro_diff_norm.columns]
    colormap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    vlines = [
        pd.Timestamp('2008-09-15'),
        pd.Timestamp('2013-05-22'),
        pd.Timestamp('2020-03-16')
    ]
    if cols_rd:
        plt.figure(figsize=(10, 5))
        for i, col in enumerate(cols_rd):
            plt.plot(macro_diff_norm.index, macro_diff_norm[col], label=col, color=colormap[i % len(colormap)])
        # Removed vlines loop
        plt.title('Differenced macro drivers (RealRate & DXY)')
        plt.xlabel('Date')
        plt.ylabel('Standardized value (z-score)')
        plt.legend(title="Drivers")
        plt.tight_layout()
        plt.show()

    # ---- 图 1b-2: RealRate, DXY, VIX, TNX, GSPC 全部 ----
    cols_all = [c for c in ["RealRate", "DXY", "VIX", "TNX", "GSPC"] if c in macro_diff_norm.columns]
    if cols_all:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(cols_all):
            plt.plot(macro_diff_norm.index, macro_diff_norm[col], label=col, color=colormap[i % len(colormap)])
        # Removed vlines loop
        plt.title('Differenced macro drivers (All macro drivers)')
        plt.xlabel('Date')
        plt.ylabel('Standardized value (z-score)')
        plt.legend(title="Drivers")
        plt.tight_layout()
        plt.show()

# 随机游走基准（价格空间）：P̂_t = P_{t-1}
anchor = price_train.iloc[[-1]]
series_for_pred = pd.concat([anchor, price_test])
rw_preds = series_for_pred.shift(1).loc[price_test.index]

# 评估（价格空间）
mae = (price_test - rw_preds).abs().mean()
rmse = np.sqrt(((price_test - rw_preds) ** 2).mean())

# ⚠️ 修复: 将 Series 转换为浮点数再格式化。
print(f"RW baseline MAE={mae.item():.6f}, RMSE={rmse.item():.6f}")

# ARIMA 模型（作为统计型备选模型）
arima_order = (1, 1, 1)  # 可以在报告中说明通过 AIC/BIC 或试验选择
# 在 log 空间估计 ARIMA，在价格空间评估
hist_arima = log_train.copy()
arima_log_preds_list = []

for t in log_test.index:
    # 每一步用当前可用样本重新估计 ARIMA，并做一步预测（log 空间）
    arima_model = sm.tsa.ARIMA(hist_arima, order=arima_order)
    arima_res = arima_model.fit()
    forecast_log = arima_res.forecast(steps=1).iloc[0]
    arima_log_preds_list.append(float(forecast_log))
    # 将真实的 log 价格加入样本，递归前进
    hist_arima.loc[t] = log_test.loc[t]

arima_log_preds = pd.Series(arima_log_preds_list, index=log_test.index)
arima_price_preds = np.exp(arima_log_preds)

diff_arima = price_test.values - arima_price_preds.values
mae_arima = np.mean(np.abs(diff_arima))
rmse_arima = np.sqrt(np.mean(diff_arima ** 2))
print(f"ARIMA{arima_order} MAE={mae_arima.item():.6f}, RMSE={rmse_arima.item():.6f}")

# ARIMAX 模型：加入宏观因子 DXY, TNX, VIX, RealRate
arimax_preds = None
arimax_price_preds = None
mae_arimax = np.nan
rmse_arimax = np.nan

if exog_train is not None:
    try:
        # 使用与 ARIMA 相同的“滚动一步预测”方案，但加入宏观外生变量
        hist_y = log_train.copy()
        hist_exog = exog_train.copy()
        arimax_log_preds_list = []

        for t in log_test.index:
            # 当前这一步的外生变量（1 行 DataFrame）
            exog_fore = exog_test.loc[[t]]

            arimax_model = sm.tsa.SARIMAX(
                hist_y,
                order=arima_order,
                exog=hist_exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            arimax_res = arimax_model.fit(disp=False)
            forecast_log = arimax_res.forecast(steps=1, exog=exog_fore).iloc[0]
            arimax_log_preds_list.append(float(forecast_log))

            # 将真实的 log 价格和对应的外生变量加入样本，递归向前滚动
            hist_y.loc[t] = log_test.loc[t]
            hist_exog.loc[t] = exog_test.loc[t]

        arimax_log_preds = pd.Series(arimax_log_preds_list, index=log_test.index)
        arimax_price_preds = np.exp(arimax_log_preds)
        arimax_preds = arimax_price_preds  # 供后面指标统计使用

        diff_arimax = price_test.values - arimax_price_preds.values
        mae_arimax = np.mean(np.abs(diff_arimax))
        rmse_arimax = np.sqrt(np.mean(diff_arimax ** 2))
        print(f"ARIMAX{arima_order} MAE={mae_arimax:.6f}, RMSE={rmse_arimax:.6f}")
    except Exception as e:
        print("ARIMAX 拟合失败:", e) 
# =======================================================
# DIAGNOSTICS: ARIMA and ARIMAX Residual Analysis
# =======================================================
print("\n===== Running Residual Diagnostics for ARIMA and ARIMAX =====")

def run_diagnostics(model_name, y_train, order, exog_train=None):
    """
    对给定的训练集和模型配置运行残差诊断并绘图。
    适用于 ARIMA 和 ARIMAX。
    """
    try:
        if exog_train is not None:
            # 使用 SARIMAX 拟合 ARIMAX
            model = sm.tsa.SARIMAX(
                y_train,
                order=order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            # 使用 ARIMA 拟合 Simple ARIMA
            model = sm.tsa.ARIMA(y_train, order=order)
        
        # 拟合最终模型到整个训练集
        res = model.fit(disp=False)
        resid = res.resid
        resid_sq = resid**2
        
        print(f"\n[Diagnostics] Successfully fitted final {model_name} model.")
# --- 3. Descriptive Statistics (Mean and Std Dev) ---
        resid_mean = resid.mean()
        resid_std = resid.std()
        
        print(f"\n{model_name} Residual Descriptive Statistics:")
        print(f"    Mean (should be near 0): {resid_mean:.6f}")
        print(f"    Standard Deviation: {resid_std:.6f}")
        
        # ... [ACF/PACF 绘图代码继续] ...
        # --- 1. ACF/PACF of residuals (White Noise Check) ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        # 
        sm.graphics.tsa.plot_acf(resid, lags=40, ax=axes[0], title=f'{model_name} Residual ACF (White Noise Check)')
        sm.graphics.tsa.plot_pacf(resid, lags=40, ax=axes[1], title=f'{model_name} Residual PACF')
       
        # <<< 关键调整：设置 Y 轴刻度，放大相关性 >>>
        axes[0].set_ylim([-0.05, 0.05]) 
        axes[1].set_ylim([-0.05, 0.05])

        plt.tight_layout()
        plt.savefig(f"{model_name}_Residuals_ACF_PACF.png")
        plt.show()

        # --- 2. ACF/PACF of squared residuals (Volatility Clustering / GARCH Check) ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        # 
        sm.graphics.tsa.plot_acf(resid_sq, lags=40, ax=axes[0], title=f'{model_name} Squared Residuals ACF (GARCH Check)')
        sm.graphics.tsa.plot_pacf(resid_sq, lags=40, ax=axes[1], title=f'{model_name} Squared Residuals PACF')
       
       # <<< 关键调整：设置 Y 轴刻度，放大平方残差相关性 >>>
        axes[0].set_ylim([-0.05, 0.05]) 
        axes[1].set_ylim([-0.05, 0.05])
       
        plt.tight_layout()
        plt.savefig(f"{model_name}_SquaredResiduals_ACF_PACF.png")
        plt.show()

        # --- 3. Ljung-Box Test (Formal White Noise Test) ---
        # 仅在残差序列长度足够时执行（确保不会因数据过少报错）
        if len(resid) > 10:
            lb_test = sm.stats.acorr_ljungbox(resid, lags=[10], return_df=True)
            print(f"{model_name} Ljung-Box Test (p-value for no autocorrelation):")
            print(lb_test)

    except Exception as e:
        print(f"[Error] {model_name} Residual Diagnostics failed: {e}")
        
# -------------------------------------------------------
# 运行诊断
# -------------------------------------------------------

# 1. 诊断 Simple ARIMA
run_diagnostics("ARIMA", log_train, arima_order)

# 2. 诊断 ARIMAX (需要确保 exog_train 已成功创建)
# 注意：这里使用修正后的逻辑，exog_train 应该是水平值 (levels)
# 确保你已经修正了前面代码中对 exog_train 的赋值，使其为水平值！
if exog_train is not None and not exog_train.empty:
    run_diagnostics("ARIMAX", log_train, arima_order, exog_train=exog_train)
else:
    print("ARIMAX Diagnostics skipped: Exogenous variables not available.")

print("=======================================================\n")
# 如可用，计算 Prophet 的滚动一步预测
prophet_preds = None
if HAS_PROPHET:
    # 将训练集转为 Prophet 所需格式
    def to_prophet_df(s: pd.Series):
        dfp = s.reset_index()
        dfp.columns = ["ds", "y"]
        return dfp
    
    hist = price_train.copy()
    prophet_preds_list = []
    for t in price_test.index:
        df_p = to_prophet_df(hist)
        # ⚠️ 每次循环都实例化一个新的 Prophet 模型
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_p)
        future = pd.DataFrame({"ds": [t]})
        yhat = model.predict(future)["yhat"].iloc[0]
        prophet_preds_list.append(yhat)
        # 加入真实价格，滚动前进
        hist.loc[t] = price_test.loc[t]

    prophet_preds = pd.Series(prophet_preds_list, index=price_test.index)
    diff_pr = price_test.values - prophet_preds.values
    mae_pr = np.mean(np.abs(diff_pr))
    rmse_pr = np.sqrt(np.mean(diff_pr ** 2))
    # ⚠️ 修复: 将 Series 转换为浮点数再格式化。
    print(f"Prophet MAE={mae_pr.item():.6f}, RMSE={rmse_pr.item():.6f}")

# =========================
# Figure 2: Combined forecasts vs Actual (Results section)
# =========================
plt.figure()
plt.plot(price_test.index, price_test.values, label='Actual')
plt.plot(price_test.index, rw_preds.values, label='RW (benchmark)')
plt.plot(price_test.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if exog_train is not None and not np.isnan(mae_arimax):
    plt.plot(price_test.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
if prophet_preds is not None:
    plt.plot(price_test.index, prophet_preds.values, label='Prophet')
plt.title(f'{ticker} {H}-step forecasts (test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Figure 2b: Model comparison without RW (Results section)
#   Focus on ARIMA vs ARIMAX vs Prophet
# =========================
plt.figure()
plt.plot(price_test.index, price_test.values, label='Actual')
plt.plot(price_test.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if exog_train is not None and not np.isnan(mae_arimax):
    plt.plot(price_test.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
if prophet_preds is not None:
    plt.plot(price_test.index, prophet_preds.values, label='Prophet')
plt.title(f'{ticker} {H}-step price forecast comparison: univariate vs multivariate models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Figure 3: ARIMA vs Actual (Appendix)
# =========================
plt.figure()
plt.plot(price_test.index, price_test.values, label='Actual')
plt.plot(price_test.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if exog_train is not None and not np.isnan(mae_arimax):
    plt.plot(price_test.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
plt.title(f'{ticker} ARIMA vs Actual (test set, {H}-step ahead)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()


if prophet_preds is not None:
    # =========================
    # Figure 4: Prophet vs Actual (Appendix)
    # =========================
    plt.figure()
    plt.plot(price_test.index, price_test.values, label='Actual')
    plt.plot(price_test.index, prophet_preds.values, label='Prophet')
    plt.title(f'{ticker} Prophet vs Actual (test set, {H}-step ahead)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# 汇总指标输出：把所有模型的 MAE / RMSE 放到一个表里
# =========================
rows = []

# 随机游走
rows.append((
    "RW",
    float(mae.item()),
    float(rmse.item())
))

# ARIMA
rows.append((
    "ARIMA",
    float(mae_arima),
    float(rmse_arima)
))

# ARIMAX（如果成功估计）
if not np.isnan(mae_arimax):
    rows.append((
        "ARIMAX",
        float(mae_arimax),
        float(rmse_arimax)
    ))

# Prophet（如果存在）
if prophet_preds is not None:
    rows.append((
        "Prophet",
        float(mae_pr),
        float(rmse_pr)
    ))

metrics_df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE"]).set_index("Model")

print("\n===== Forecast Error Summary (MAE / RMSE) =====")
print(metrics_df)
print("==============================================\n")

# =========================
# Sanity check: show first few actual vs predicted values
# =========================
print("\n===== Sanity check: first 5 observations (price space) =====")

sanity_cols = {"Actual": price_test.iloc[:5]}
# 注意：逐个模型检查是否存在，再加入对比
sanity_cols["RW"] = rw_preds.iloc[:5]

sanity_cols["ARIMA"] = pd.Series(arima_price_preds.iloc[:5], index=price_test.iloc[:5].index, name="ARIMA")

if arimax_price_preds is not None and not np.isnan(mae_arimax):
    sanity_cols["ARIMAX"] = pd.Series(
        arimax_price_preds.iloc[:5],
        index=price_test.iloc[:5].index,
        name="ARIMAX"
    )

if prophet_preds is not None:
    sanity_cols["Prophet"] = pd.Series(
        prophet_preds.iloc[:5],
        index=price_test.iloc[:5].index,
        name="Prophet"
    )

 # Ensure everything is 1-D Series (avoid shape (5,1) issues)
aligned_index = price_test.iloc[:5].index
for k, v in sanity_cols.items():
    # Convert to numpy array and flatten, then wrap back into a Series
    arr = np.asarray(v).reshape(-1)
    sanity_cols[k] = pd.Series(arr, index=aligned_index, name=k)
sanity_df = pd.DataFrame(sanity_cols)
print("\nRaw values (first 5):")
print(sanity_df)

# 打印各模型的前 5 个绝对误差，方便和 MAE/RMSE 对照
print("\nAbsolute errors (first 5):")
for col in sanity_df.columns:
    if col == "Actual":
        continue
    abs_err = (sanity_df["Actual"] - sanity_df[col]).abs()
    print(f"{col}: {abs_err.values}")
print("==============================================\n")

# =========================
# Figure 5: Error Comparison (RW vs ARIMA)
# =========================
if {"RW", "ARIMA"}.issubset(metrics_df.index):
    subset = metrics_df.loc[["RW", "ARIMA"]]

    plt.figure()
    plt.bar(subset.index, subset["MAE"])
    plt.title("MAE Comparison: RW vs ARIMA")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(subset.index, subset["RMSE"])
    plt.title("RMSE Comparison: RW vs ARIMA")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

# =========================
# Figure 6: Error Comparison (all available models)
# =========================
plt.figure()
plt.bar(metrics_df.index, metrics_df["MAE"])
plt.title("MAE Comparison across models")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(metrics_df.index, metrics_df["RMSE"])
plt.title("RMSE Comparison across models")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
