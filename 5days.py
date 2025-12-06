import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

# Forecast horizon in trading days (default: 5 days ahead)
H = int(os.environ.get("HORIZON", "5"))
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
    # 原始宏观因子矩阵（水平数据，ARIMAX 使用 level）
    exog = pd.concat(exog_list, axis=1)
    # 向前填充缺失值，确保与价格序列对齐
    exog = exog.ffill()
    # 与价格序列相同的 80/20 切分，用于 ARIMAX 外生变量
    exog_train = exog.iloc[:split]
    exog_test = exog.iloc[split:]
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

# =========================
# 构造用于 H-step ahead 评估的测试索引
# =========================
test_index = price_test.index

if len(test_index) <= H:
    raise ValueError(f"测试集长度 {len(test_index)} 过短，无法进行 {H}-step ahead 预测评估。")

# target_index: 每个点都是“预测 H 日之后”的目标日期
target_index = test_index[H-1:]
price_test_H = price_test.loc[target_index]
log_test_H = log_test.loc[target_index]

# =========================
# 随机游走基准（H-step ahead）：P̂_{t+H} = P_t
# =========================
rw_dict = {}
last_train_price = price_train.iloc[-1]

for i, tgt in enumerate(target_index):
    # i = 0 时，origin 在训练集末尾；之后 origin 依次为 test_index[i-1]
    if i == 0:
        last_obs = last_train_price
    else:
        last_obs = price_test.iloc[i-1]
    rw_dict[tgt] = float(last_obs)

rw_preds = pd.Series(rw_dict, index=target_index)

# 评估（价格空间）
rw_diff = price_test_H.values - rw_preds.values
mae = np.mean(np.abs(rw_diff))
rmse = np.sqrt(np.mean(rw_diff ** 2))

print(f"RW baseline ({H}-step) MAE={mae:.6f}, RMSE={rmse:.6f}")

# =========================
# ARIMA 模型（H-step ahead，在 log 空间估计，价格空间评估）
# =========================
arima_order = (1, 1, 1)  # 在报告中说明通过 AIC/BIC 或试验选择

arima_dict = {}
hist_arima = log_train.copy()

for i, tgt in enumerate(target_index):
    # 每个 origin 重新估计一次 ARIMA
    arima_model = sm.tsa.ARIMA(hist_arima, order=arima_order)
    arima_res = arima_model.fit()

    # 预测未来 H 步（log 空间），取路径的最后一个点作为 H-step 预测
    fcast_log_path = arima_res.forecast(steps=H)
    forecast_log = fcast_log_path.iloc[-1]
    arima_dict[tgt] = float(np.exp(forecast_log))

    # 用新观测更新样本：新增一个 test_index[i] 的真实 log 价格
    new_idx = test_index[i]
    hist_arima.loc[new_idx] = float(log_test.loc[new_idx])

arima_price_preds = pd.Series(arima_dict, index=target_index)

diff_arima = price_test_H.values - arima_price_preds.values
mae_arima = np.mean(np.abs(diff_arima))
rmse_arima = np.sqrt(np.mean(diff_arima ** 2))
print(f"ARIMA{arima_order} ({H}-step) MAE={mae_arima:.6f}, RMSE={rmse_arima:.6f}")

# =========================
# ARIMAX 模型：加入宏观因子（H-step ahead）
# =========================
arimax_price_preds = None
mae_arimax = np.nan
rmse_arimax = np.nan

if exog_train is not None:
    try:
        arimax_dict = {}
        hist_y = log_train.copy()
        hist_exog = exog_train.copy()

        for i, tgt in enumerate(target_index):
            # 未来 H 日的外生变量路径（与目标对齐）
            exog_future = exog_test.iloc[i:i+H]
            if len(exog_future) < H:
                break  # 剩余长度不够 H 步，跳出

            model = sm.tsa.SARIMAX(
                hist_y,
                order=arima_order,
                exog=hist_exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

            fcast_log_path = res.forecast(steps=H, exog=exog_future)
            forecast_log = fcast_log_path.iloc[-1]
            arimax_dict[tgt] = float(np.exp(forecast_log))

            # 更新样本：加入一个新的真实观测（log 价格 + 外生变量）
            new_idx = test_index[i]
            hist_y.loc[new_idx] = float(log_test.loc[new_idx])
            hist_exog.loc[new_idx] = exog_test.loc[new_idx]

        if arimax_dict:
            arimax_price_preds = pd.Series(arimax_dict, index=sorted(arimax_dict.keys()))
            # 确保与目标集合交集
            common_idx = price_test_H.index.intersection(arimax_price_preds.index)
            diff_arimax = price_test_H.loc[common_idx].values - arimax_price_preds.loc[common_idx].values
            mae_arimax = np.mean(np.abs(diff_arimax))
            rmse_arimax = np.sqrt(np.mean(diff_arimax ** 2))
            print(f"ARIMAX{arima_order} ({H}-step) MAE={mae_arimax:.6f}, RMSE={rmse_arimax:.6f}")
    except Exception as e:
        print("ARIMAX 拟合失败:", e)

# =========================
# Prophet 模型：H-step ahead 预测（价格空间）
# =========================
prophet_preds = None
mae_pr = np.nan
rmse_pr = np.nan

if HAS_PROPHET:
    def to_prophet_df(s: pd.Series):
        dfp = s.reset_index()
        dfp.columns = ["ds", "y"]
        return dfp

    prophet_dict = {}
    hist = price_train.copy()

    for i, tgt in enumerate(target_index):
        df_p = to_prophet_df(hist)
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_p)

        # 向前扩展 H 个“工作日”（freq='B'），取最后一个点作为 H-step 预测
        future = model.make_future_dataframe(periods=H, freq='B')
        fcst = model.predict(future)
        yhat_H = fcst["yhat"].iloc[-1]
        prophet_dict[tgt] = float(yhat_H)

        # 更新样本：加入一个新的真实价格观测
        new_idx = test_index[i]
        hist.loc[new_idx] = float(price_test.loc[new_idx])

    if prophet_dict:
        prophet_preds = pd.Series(prophet_dict, index=sorted(prophet_dict.keys()))
        common_idx_p = price_test_H.index.intersection(prophet_preds.index)
        diff_pr = price_test_H.loc[common_idx_p].values - prophet_preds.loc[common_idx_p].values
        mae_pr = np.mean(np.abs(diff_pr))
        rmse_pr = np.sqrt(np.mean(diff_pr ** 2))
        print(f"Prophet ({H}-step) MAE={mae_pr:.6f}, RMSE={rmse_pr:.6f}")

# =========================
# Figure 2: Combined forecasts vs Actual (Results section)
# （所有模型的 H-step 预测）
# =========================
plt.figure()
plt.plot(price_test_H.index, price_test_H.values, label='Actual')
plt.plot(rw_preds.index, rw_preds.values, label='RW (benchmark)')
plt.plot(arima_price_preds.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if arimax_price_preds is not None and not np.isnan(mae_arimax):
    plt.plot(arimax_price_preds.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
if prophet_preds is not None:
    plt.plot(prophet_preds.index, prophet_preds.values, label='Prophet')
plt.title(f'{ticker} {H}-step ahead forecasts (test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Figure 2b: Model comparison without RW (Results section)
# =========================
plt.figure()
plt.plot(price_test_H.index, price_test_H.values, label='Actual')
plt.plot(arima_price_preds.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if arimax_price_preds is not None and not np.isnan(mae_arimax):
    plt.plot(arimax_price_preds.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
if prophet_preds is not None:
    plt.plot(prophet_preds.index, prophet_preds.values, label='Prophet')
plt.title(f'{ticker} {H}-step ahead price forecast comparison: univariate vs multivariate models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Figure 3: ARIMA vs Actual (Appendix)
# =========================
plt.figure()
plt.plot(price_test_H.index, price_test_H.values, label='Actual')
plt.plot(arima_price_preds.index, arima_price_preds.values, label=f'ARIMA{arima_order}')
if arimax_price_preds is not None and not np.isnan(mae_arimax):
    plt.plot(arimax_price_preds.index, arimax_price_preds.values, label=f'ARIMAX{arima_order} (macro)')
plt.title(f'{ticker} ARIMA vs Actual (test set, {H}-step ahead)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Figure 4: Prophet vs Actual (Appendix, 若存在)
# =========================
if prophet_preds is not None:
    plt.figure()
    plt.plot(price_test_H.index, price_test_H.values, label='Actual')
    plt.plot(prophet_preds.index, prophet_preds.values, label='Prophet')
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
rows.append(("RW", float(mae), float(rmse)))

# ARIMA
rows.append(("ARIMA", float(mae_arima), float(rmse_arima)))

# ARIMAX（如果成功估计）
if arimax_price_preds is not None and not np.isnan(mae_arimax):
    rows.append(("ARIMAX", float(mae_arimax), float(rmse_arimax)))

# Prophet（如果存在）
if prophet_preds is not None and not np.isnan(mae_pr):
    rows.append(("Prophet", float(mae_pr), float(rmse_pr)))

metrics_df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE"]).set_index("Model")

print("\n===== Forecast Error Summary (MAE / RMSE) =====")
print(metrics_df)
print("==============================================\n")

# =========================
# Sanity check: show first few actual vs predicted values (H-step)
# =========================
print("\n===== Sanity check: first 5 observations (H-step, price space) =====")

first5_idx = price_test_H.index[:5]
sanity_cols = {"Actual": price_test_H.loc[first5_idx]}

sanity_cols["RW"] = rw_preds.loc[first5_idx]
sanity_cols["ARIMA"] = arima_price_preds.loc[first5_idx]

if arimax_price_preds is not None and not np.isnan(mae_arimax):
    common_idx_ax = first5_idx.intersection(arimax_price_preds.index)
    sanity_cols["ARIMAX"] = arimax_price_preds.loc[common_idx_ax]

if prophet_preds is not None:
    common_idx_p5 = first5_idx.intersection(prophet_preds.index)
    sanity_cols["Prophet"] = prophet_preds.loc[common_idx_p5]

sanity_df = pd.DataFrame(sanity_cols)
print("\nRaw values (first 5):")
print(sanity_df)

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
    plt.title(f"MAE Comparison: RW vs ARIMA ({H}-step)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(subset.index, subset["RMSE"])
    plt.title(f"RMSE Comparison: RW vs ARIMA ({H}-step)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

# =========================
# Figure 6: Error Comparison (all available models)
# =========================
plt.figure()
plt.bar(metrics_df.index, metrics_df["MAE"])
plt.title(f"MAE Comparison across models ({H}-step)")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(metrics_df.index, metrics_df["RMSE"])
plt.title(f"RMSE Comparison across models ({H}-step)")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
