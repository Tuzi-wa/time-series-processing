import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

# 忽略警告
warnings.filterwarnings("ignore")
# Mac 后台绘图设置
import matplotlib
matplotlib.use('Agg') 

print("==================================================")
print("项目启动：GLD 黄金价格预测 (多因子宏观模型版)")
print("==================================================")

# 1. 开启进度条，让你看到它在动！
YF_KW = dict(progress=True, threads=False, auto_adjust=True, actions=False, repair=True)

# ---------------------------------------------------------
# 1. 数据准备：下载 GLD + 宏观三剑客 (美元, 利率, 恐慌指数)
# ---------------------------------------------------------
# GLD: 黄金 ETF
# DX-Y.NYB: 美元指数
# ^TNX: 10年期美债收益率 (代表利率/政策机会成本)
# ^VIX: 恐慌指数 (代表市场风险/政策不确定性)
tickers = ["GLD", "DX-Y.NYB", "^TNX", "^VIX"]
print(f"\n[Step 1] 正在下载多因子数据: {tickers} ...")

try:
    # 批量下载
    raw_data = yf.download(tickers, start="2000-01-01", group_by='ticker', **YF_KW)
except Exception as e:
    print(f"下载致命错误: {e}")
    exit()

# 数据清洗与提取
df = pd.DataFrame()

# 辅助函数：安全提取收盘价
def get_close(data, ticker):
    # 尝试多种列名结构
    if (ticker, 'Adj Close') in data.columns:
        return data[(ticker, 'Adj Close')]
    elif (ticker, 'Close') in data.columns:
        return data[(ticker, 'Close')]
    elif 'Adj Close' in data.columns and ticker in data['Adj Close']:
        return data['Adj Close'][ticker]
    elif 'Close' in data.columns and ticker in data['Close']:
        return data['Close'][ticker]
    else:
        # 如果实在找不到，尝试单独补下载
        print(f"尝试补下载: {ticker}")
        tmp = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
        return tmp['Close'] if 'Close' in tmp else tmp.iloc[:, 0]

df['GLD'] = get_close(raw_data, 'GLD')
df['DXY'] = get_close(raw_data, 'DX-Y.NYB')
df['TNX'] = get_close(raw_data, '^TNX') # 利率
df['VIX'] = get_close(raw_data, '^VIX') # 恐慌

# 关键步骤：去除空值 (对齐数据)
# 注意：债市(^TNX)和股市(GLD)休市时间不同，必须取交集
original_len = len(df)
df.dropna(inplace=True)
print(f"数据对齐完成。原始数据: {original_len} -> 对齐后: {len(df)}")
print(f"包含变量: 黄金(GLD), 美元(DXY), 利率(TNX), 恐慌(VIX)")

# 划分训练/测试集
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ---------------------------------------------------------
# PART 1: 基础模型 (Random Walk & ARIMA)
# ---------------------------------------------------------
print("\n[PART 1] 运行基础模型 (Benchmark)...")

# 1. Random Walk
rw_preds = pd.concat([train['GLD'].iloc[[-1]], test['GLD']]).shift(1).iloc[1:]
rmse_rw = np.sqrt(((test['GLD'] - rw_preds) ** 2).mean())
print(f"   >> Random Walk RMSE: {rmse_rw:.4f}")

# 2. ARIMA (Univariate) - 为了速度，我们只跑测试集的前 50 天作为演示
# 如果你想跑全量，把下面这行改成 run_limit = None
run_limit = 50 
test_subset = test.iloc[:run_limit] if run_limit else test

print(f"   >> 正在运行 ARIMA (单变量) ...")
arima_preds = []
history = train['GLD'].copy()

for t in test_subset.index:
    model = sm.tsa.ARIMA(history, order=(1,1,1))
    res = model.fit()
    pred = res.forecast(steps=1).iloc[0]
    arima_preds.append(pred)
    history.loc[t] = test.loc[t, 'GLD']

rmse_arima = np.sqrt(((test_subset['GLD'] - pd.Series(arima_preds, index=test_subset.index)) ** 2).mean())
print(f"   >> ARIMA RMSE: {rmse_arima:.4f}")

# ---------------------------------------------------------
# PART 2: 进阶模型 ARIMAX (多因子：美元 + 利率 + 恐慌)
# ---------------------------------------------------------
print("\n[PART 2] 运行 ARIMAX 多因子模型 (引入 DXY, TNX, VIX)...")
print("注意：因为引入了多个变量，计算会比之前慢，请耐心等待刷屏...")

arimax_preds = []
# 历史 Y (黄金)
hist_y = train['GLD'].copy()
# 历史 X (宏观变量: 美元, 利率, 恐慌)
hist_x = train[['DXY', 'TNX', 'VIX']].copy()

counter = 0
total = len(test_subset)

for t in test_subset.index:
    counter += 1
    # 让程序“说话”，告诉你它没死机
    print(f"[{counter}/{total}] 正在预测日期: {t.date()} ...")
    
    # 1. 获取当天的宏观数据 (Exogenous variables)
    # 我们假设已知当天的宏观环境，来预测当天的金价
    current_exog = test.loc[[t], ['DXY', 'TNX', 'VIX']]
    
    # 2. 训练模型 (加入 exog)
    # 这里的 exog=hist_x 告诉模型：去学习过去金价和这些宏观变量的关系
    model = sm.tsa.ARIMA(endog=hist_y, exog=hist_x, order=(1,1,1))
    
    try:
        res = model.fit()
        # 3. 预测
        pred = res.forecast(steps=1, exog=current_exog).iloc[0]
    except Exception as e:
        print(f"   计算出错: {e}，使用昨日价格兜底")
        pred = hist_y.iloc[-1]
    
    arimax_preds.append(pred)
    
    # 4. 更新历史数据库
    hist_y.loc[t] = test.loc[t, 'GLD']
    hist_x.loc[t] = test.loc[t, ['DXY', 'TNX', 'VIX']]

# 计算 RMSE
arimax_series = pd.Series(arimax_preds, index=test_subset.index)
rmse_arimax = np.sqrt(((test_subset['GLD'] - arimax_series) ** 2).mean())

# ---------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------
print("\n" + "="*40)
print("FINAL RESULTS SUMMARY (RMSE)")
print("="*40)
# 重新计算 RW 在该子集上的 RMSE
rmse_rw_sub = np.sqrt(((test_subset['GLD'] - rw_preds.loc[test_subset.index]) ** 2).mean())

print(f"1. Benchmark (Random Walk) : {rmse_rw_sub:.4f}")
print(f"2. ARIMA (Only Price)      : {rmse_arima:.4f}")
print(f"3. ARIMAX (Macro Factors)  : {rmse_arimax:.4f}")
print("-" * 40)

if rmse_arimax < rmse_rw_sub:
    print("结论: 宏观模型 (ARIMAX) 胜出！\n引入利率(TNX)和恐慌指数(VIX) 成功提升了预测精度。")
else:
    print("结论: 随机游走 (RW) 依然强劲。\n这说明在超短期预测中，宏观基本面很难立即战胜市场噪音。")

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(test_subset.index, test_subset['GLD'], label='Actual GLD', color='black', linewidth=1.5)
plt.plot(test_subset.index, arimax_series, label='ARIMAX (Macro)', color='red', alpha=0.8)
plt.plot(test_subset.index, pd.Series(arima_preds, index=test_subset.index), label='ARIMA (Simple)', color='green', linestyle='--', alpha=0.6)

plt.title('Gold Price Forecast: Univariate vs Macro-Factor Model (DXY + TNX + VIX)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

img_name = "Final_Macro_Model_Result.png"
plt.savefig(img_name)
print(f"\n图片已保存为: {img_name}")