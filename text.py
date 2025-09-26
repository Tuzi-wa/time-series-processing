import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

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
    for t in candidates:
        try:
            df, src = try_yf(t, start=start, end=end, interval=interval)
            return t, df, src
        except Exception as e:
            print(f"yfinance: {t} 失败：{e}")
            last_err = e
    raise RuntimeError(f"所有数据源均失败，最后错误：{last_err}")

# 项目要求: 使用GLD自成立以来的全样本
candidates = ["GLD"]
start_date = "2004-11-18" # GLD ETF的成立日期
interval = "1d"

ticker, df, src = download_with_fallback(candidates, start=start_date, end=None, interval=interval)
print(f"使用数据源：{src}，ticker：{ticker}，样本数：{len(df)}")

# 选择价格列：优先 Adj Close，没有就用 Close
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
ts = df[price_col].dropna()

# 基础检查与切分
n = len(ts)
if n < 2:
    raise ValueError(f"{ticker} 可用数据过少：{n} 行。")

split = int(n * 0.8)
train, test = ts.iloc[:split], ts.iloc[split:]
if test.empty:
    raise ValueError("测试集为空，请调整时间范围或切分比例。")

# 随机游走基准：y_hat_t = y_{t-1}
anchor = train.iloc[[-1]]
series_for_pred = pd.concat([anchor, test])
rw_preds = series_for_pred.shift(1).loc[test.index]

# 评估
mae = (test - rw_preds).abs().mean()
rmse = np.sqrt(((test - rw_preds) ** 2).mean())

print(f"RW baseline MAE={mae.item():.6f}, RMSE={rmse.item():.6f}")

# ARIMA/ARIMAX 的滚动一步预测
arima_preds = None
# 请注意: ARIMA模型的阶数(p,d,q)需要通过ACF/PACF图和信息准则(AIC/BIC)来确定
# 由于本项目代码不执行自动选择，这里使用一个常用阶数作为示例 (e.g., ARIMA(1,1,0))
arima_order = (1, 1, 0)
print(f"正在进行ARIMA{arima_order}的滚动一步预测...")

hist_arima = train.copy()
arima_preds_list = []
for t in test.index:
    # 重新训练模型以实现滚动预测
    model_arima = ARIMA(hist_arima, order=arima_order)
    model_fit = model_arima.fit()
    yhat = model_fit.forecast(steps=1).iloc[0]
    arima_preds_list.append(yhat)
    # 加入真实值，滚动前进
    hist_arima.loc[t] = test.loc[t]
    
    # ⚠️ 项目要求: 残差诊断和稳健标准误
    # 残差白噪声检验: 在报告中讨论
    # Ljung-Box Q检验: acorr_ljungbox(model_fit.resid, lags=[...])
    # Durbin-Watson检验: durbin_watson(model_fit.resid)
    # 残差图: model_fit.plot_diagnostics()

arima_preds = pd.Series(arima_preds_list, index=test.index)
mae_arima = np.abs(test.to_numpy() - arima_preds.to_numpy()).mean()
rmse_arima = np.sqrt(np.power((test.to_numpy() - arima_preds.to_numpy()), 2).mean())
print(f"ARIMA{arima_order} MAE={mae_arima.item():.6f}, RMSE={rmse_arima.item():.6f}")

# Prophet 的滚动一步预测
prophet_preds = None
if HAS_PROPHET:
    print("正在进行Prophet的滚动一步预测...")
    def to_prophet_df(s: pd.Series):
        dfp = s.reset_index()
        dfp.columns = ["ds", "y"]
        return dfp
    
    hist_prophet = train.copy()
    prophet_preds_list = []
    for t in test.index:
        df_p = to_prophet_df(hist_prophet)
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_p) 
        future = pd.DataFrame({"ds": [t]})
        yhat = model.predict(future)["yhat"].iloc[0]
        prophet_preds_list.append(yhat)
        hist_prophet.loc[t] = test.loc[t]

    prophet_preds = pd.Series(prophet_preds_list, index=test.index)
    mae_pr = np.abs(test.to_numpy() - prophet_preds.to_numpy()).mean()
    rmse_pr = np.sqrt(np.power((test.to_numpy() - prophet_preds.to_numpy()), 2).mean())
    print(f"Prophet MAE={mae_pr.item():.6f}, RMSE={rmse_pr.item():.6f}")

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, rw_preds.values, label='RW (benchmark)')
plt.plot(test.index, arima_preds.values, label='ARIMA')
if prophet_preds is not None:
    plt.plot(test.index, prophet_preds.values, label='Prophet')
plt.title(f'{ticker} one-step forecasts (test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# 汇总指标输出
metrics = {
    'RW_MAE': float(mae),
    'RW_RMSE': float(rmse),
    'ARIMA_MAE': float(mae_arima),
    'ARIMA_RMSE': float(rmse_arima)
}
if prophet_preds is not None:
    metrics.update({
        'Prophet_MAE': float(mae_pr),
        'Prophet_RMSE': float(rmse_pr)
    })
print("\n=== 所有模型指标汇总 ===")
print(metrics)