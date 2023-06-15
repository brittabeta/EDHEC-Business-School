########Financial Calculations#############
##Analyze Assets and Construct Portfolios##

import pandas as pd
import numpy as np
import scipy.stats 
from scipy.optimize import minimize

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def get_riskfree_rate():
    """
    Daily Treasury Bill Rates
    for use in sharpe_ratio to calculate risk-adjusted perfomance
    """
    I10YTCMR = pd.read_csv('https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2023/all?type=daily_treasury_yield_curve&field_tdr_date_value=2023&page&_format=csv',
                         index_col='Date', parse_dates=True)
    proxy = I10YTCMR['10 Yr'][0]
    print(proxy)
    return proxy

def sharpe_ratio(r, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    Changes arg. riskfree_rate to find based on treasury data
    """
    # convert the annual riskfree rate to per period
    riskfree_rate = get_riskfree_rate()/100
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def drawdown(return_series: pd.Series): # pd.Series optional # input = series, output = drawdown
    """
    Takes a time series of asset returns
    Computes and returns a dataframe that contains:
    wealth index
    previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({'WealthIndex': wealth_index,
                         'PreviousPeaks': previous_peaks,
                         'Drawdown': drawdowns})

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and
    Bottom Deciles by MarketCap
    """
    # load Fama-French, parse dates, set time series to index, return nan if value -99.99
    path = r'C:\Users\breta\OneDrive\Desktop\Workspace\EDHEC\Portfolios_Formed_on_ME_monthly_EW.csv'
    me_m = pd.read_csv(path, header=0, index_col=0, parse_dates=True, na_values=-99.99) 
    # pull desired columns, rename them
    rets = me_m[['Lo 10', 'Hi 10']] 
    rets.columns = ['SmallCap', 'LargeCap'] # rename columns
    # obtain decimal instead of percent
    rets = rets/100
    # parse dates does not work, change time series index to datetime with YYYY-MM format
    # period M month to change from YYYY-MM-01 to YYYY-MM
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    # load Fama-French, parse dates, set time series to index, return nan if value -99.99
    path = r'C:\Users\breta\OneDrive\Desktop\Workspace\EDHEC\edhec-hedgefundindices.csv'
    hfi = pd.read_csv(path, header=0, index_col=0, parse_dates=True) 
    # obtain decimal instead of percent
    hfi = hfi/100
    # parse dates DOES work, change time series index to datetime with YYYY-MM format
    # period M month to change from YYYY-MM-'end of month' to YYYY-MM 
    ## need hfi.index.to_period()
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    path = r'C:\Users\breta\OneDrive\Desktop\Workspace\EDHEC\ind30_m_vw_rets.csv'
    # set date column as index, header in row 0, convert from %
    ind = pd.read_csv(path, header=0, index_col=0)/100
    # date is as if int, parse does not work, convert to datetime YYYY-MM-01 --> to period M --> YYYY-MM
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    # clean whitespace, prevent indexing issues ['Food ']
    ind.columns = ind.columns.str.strip()
    return ind

def skewness(r): 
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use population standard deviation, set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r): 
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a series
    Versus scipy: does not return excess
    (scipy.stats.kurtosis: actual kurtosis - expected kurtosis of 3)
    """
    demeaned_r = r - r.mean()
    # use population standard deviation, set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine
    if a Series is normal or not
    Test is applied at the 1% level of significance by default
    Returns True if the hypothesis of normality is accepted,
    False otherwise
    """
    # level = 0.01, if no 2nd argument provided,
    # then defaults to 1%
    # unpack the tuple, equivalent to result[1] = p_value
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation0(r):
    """
    Appropriate if: mean close to 0 (as daily returns)
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    is_negative = r < 0 # bool mask filter
    return r[is_negative].std(ddof=0)

def semideviation3(r):
    """
    Returns the semideviation aka negative semi deviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    # demean the returns
    excess = r-r.mean()          
    # only the returns below the mean                              
    excess_negative = excess[excess<0]        
    # square the demeaned returns below the mean
    excess_negative_square = excess_negative**2
    # number of returns under the mean
    n_negative = (excess<0).sum() 
    # semideviation                            
    return (excess_negative_square.sum()/n_negative)**0.5     

def var_historic(r, level = 5):
    """
    Returns the historic Value at Risk as series
    at a specified level default percent 5% (5% level), 
    returns falling below that 'worst %', 
    Interpretation: level% chance that in any given _dataset_timeperiod_
    you will loose return [(value)*100] %
    Accepts dataframe or series
    """
    # if r is dataframe true
    if isinstance(r, pd.DataFrame):
        # then apply function to every column
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        # add - to return abs of values
        return -np.percentile(r, level)
    else: 
        raise TypeError('Expected r to be series or dataframe')
    
def var_gaussian(r, level=5):
    """
    Returns the Parametric Gaussian VaR of a Series 
    or DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))

def var_cornish_fisher(r, level=5, modified=True):
    """
    Returns the Cornish-Fisher Modified VaR of a Series 
    or DataFrame
    Modified = False returns Parametric Gaussian VaR
    """
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level/100)
    # compute skewness and kurtosis
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        # adjust z based on s and k
        z = (z +
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
             )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of
    Series or DataFrame
    """
    if isinstance(r, pd.Series):
        # find all returns less than historic var
        ## - because var_historic adjusted to be + value
        ## returns is_beyond mask
        is_beyond = r <= -var_historic(r, level=level)
        # use make is_beyond, mean gives conditional mean
        # - to report as + value
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def portfolio_return(weights, returns): 
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-assets, efficient frontier
    args: with 2 asset list l
    n_points n points/portfolios to generate
    er expected returns er[l]
    cov covariance matrix covariance cov.loc[l,l]
    style default is dot dash for markers on line
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style) 

def minimize_vol(target_return, er, cov):
    """
    Target return -> Optimal weight vector that acheive target return
    given a set of returns (er), and a covariance matrix (cov)
    """
    # number of assets = number of portfolio options
    n = er.shape[0]
    # initial guess for optimizer, 50:50
    init_guess = np.repeat(1/n, n)
    # constraints, bounds for optimizer, a sequence of bounds for every weight
    ## desire balanced weights, not >1 equivalent leverage, not neg. going short 
    ### -> min 0 max 1 for every n asset (n copys of tuple in tuple)
    bounds = ((0.0, 1),)*n
    ## ensure return satisfies target return, function fun should = 0 'eq'
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    ## weights sum to 1
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    # optimizer
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    """
    generates a list of weights to run the optimizer on to minimize the volatility
    """
    # set of target returns
    target_rs = np.linspace(er.min(), er.max(), n_points)
    # series of weights from series of returns from optimizer
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style=".-"):
    """
    Plots the multi-asset efficient frontier
    """
    # weights: finding the portfolio that minimizes volatility for a certain target return
    weights = optimal_weights(n_points, er, cov) # quadratic optimizer
    # rets, vols: same for n and 2 portfolios
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)
