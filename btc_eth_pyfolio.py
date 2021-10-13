import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyfolio as pf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

tickers = "BTC-USD ETH-USD"   #adjust tickers
data = yf.download(tickers=tickers, period="5y")['Adj Close']   #adjust period time

roi = data.pct_change()
roi

np.random.seed(42)
num_ports = 10000
all_weights = np.zeros((num_ports, len(data.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    weights = np.array(np.random.random(2))   #number of assets
    weights = weights/np.sum(weights)

    all_weights[x,:] = weights
    
    ret_arr[x] = np.sum( (roi.mean() * weights * 252))
    
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(roi.cov()*252, weights)))
    
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(roi.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(roi.cov()*252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    return np.sum(weights)-1

cons = ({'type':'eq', 'fun':check_sum})
bounds = ((0,1),(0,1))
init_guess = [0.5, 0.5]   #initial guess (balanced portfolio)
opt_results = minimize(neg_sharpe, init_guess,method='SLSQP', bounds=bounds, constraints=cons)
print(opt_results)

get_ret_vol_sr(opt_results.x)

frontier_y = np.linspace(0.8,1.25,200)   #max height graph
def minimize_volatility(weights):
  return get_ret_vol_sr(weights)[1]

frontier_x = []

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50)
plt.savefig('cover.png')
plt.show()

adj_roi = (1 + roi).cumprod()
adj_roi.iloc[0] = 1

benchmark = [0.63007041, 0.36992959] * adj_roi.iloc[:, :]   #benchmark weighings
wallet = [5, 2.5] * adj_roi.iloc[:, :]   #adjust portfolio weights

benchmark["balance"] = benchmark.sum(axis=1)
benchmark["Benchmark"] = benchmark["balance"].pct_change()

wallet["balance"] = wallet.sum(axis=1)
wallet["roi"] = wallet["balance"].pct_change()
wallet

pf.create_full_tear_sheet(wallet["roi"], benchmark_rets=benchmark["Benchmark"])   #wallet x benchmark

fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_rolling_beta(wallet["roi"], factor_returns=benchmark["Benchmark"], ax=ax1)
plt.ylim((0.97, 1.03));   #coordinates

fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_rolling_volatility(wallet["roi"], factor_returns=benchmark["Benchmark"], ax=ax1)
plt.ylim((0.2, 1.2));   #coordinates

fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_returns(wallet["roi"], ax=ax1)
plt.ylim((-0.45, 0.45));   #coordinates