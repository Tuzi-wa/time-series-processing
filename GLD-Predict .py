import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

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
candidates = ["AAPL", "GLD", "GC=F"]
start_date = "2000-01-01"
interval = "1d"

ticker, df, src = download_with_fallback(candidates, start=start_date, end=None, interval=interval)
print(f"使用数据源：{src}，ticker：{ticker}，样本数：{len(df)}")

# 选择价格列：优先 Adj Close，没有就用 Close
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
ts = df[price_col].dropna()

# ⚠️ 改进: Prophet 可以处理非连续时间序列，不需要强制转换为日频率并填充。
# 移除这一步可以避免对非交易日价格进行不必要的假设。
# ts = ts.asfreq("D").ffill() 

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

# ⚠️ 修复: 将 Series 转换为浮点数再格式化。
print(f"RW baseline MAE={mae.item():.6f}, RMSE={rmse.item():.6f}")

# 如可用，计算 Prophet 的滚动一步预测
prophet_preds = None
if HAS_PROPHET:
    # 将训练集转为 Prophet 所需格式
    def to_prophet_df(s: pd.Series):
        dfp = s.reset_index()
        dfp.columns = ["ds", "y"]
        return dfp
    
    hist = train.copy()
    prophet_preds_list = []
    for t in test.index:
        df_p = to_prophet_df(hist)
        # ⚠️ 修复: 对于滚动预测，需要每次循环都实例化一个新的模型。
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(df_p) 
        future = pd.DataFrame({"ds": [t]})
        yhat = model.predict(future)["yhat"].iloc[0]
        prophet_preds_list.append(yhat)
        # 加入真实值，滚动前进
        hist.loc[t] = test.loc[t]

    prophet_preds = pd.Series(prophet_preds_list, index=test.index)
    mae_pr = (test - prophet_preds).abs().mean()
    rmse_pr = np.sqrt(((test - prophet_preds) ** 2).mean())
    # ⚠️ 修复: 将 Series 转换为浮点数再格式化。
    print(f"Prophet MAE={mae_pr.item():.6f}, RMSE={rmse_pr.item():.6f}")

# 绘图
plt.figure()
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, rw_preds.values, label='RW (benchmark)')
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
    'RW_RMSE': float(rmse)
}
if prophet_preds is not None:
    metrics.update({
        'Prophet_MAE': float(mae_pr),
        'Prophet_RMSE': float(rmse_pr)
    })
print(metrics)
