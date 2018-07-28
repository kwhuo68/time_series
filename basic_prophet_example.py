import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import itertools
import statsmodels.api as sm

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
from time import time
from fbprophet import Prophet 


#Get time series, resample by a particular frequency 
def parseSeries(filepath, time_frame):
	"""
	Obtain dataframe with time series with counts via a particular frequency 

	:param str filepath: path of JSON file to parse time series for
	:param int time_frame: time frame to resample by in minutes
	"""

	print("this is filepath: " + str(filepath))
	df = pd.read_json(path_or_buf = filepath)
	df.columns = ['ts']
	df['ts'] = pd.to_datetime(df['ts'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')))
	df['count'] = df['ts'].dt.strftime('%Y-%m-%d')

	# Example - Get 15-min aggregations 
	data = df.set_index(['ts'])
	resampled = data.resample('%smin' %(time_frame), how = 'count')

	return(resampled)

# Get and save plot 
def saveFigure(fig, title, xlabel, ylabel, filename):
	"""
	Helper function for producing uniform graphics

	:param str filepath: path of JSON file to parse time series for
	:param int time_frame: time frame to resample by in minutes
	"""

	fig.set_figheight(8)
	fig.set_figwidth(16)

	if title is None and xlabel is None and ylabel is None:
		plt.savefig(filename)
		plt.close()
		return 
	
	plt.title(title, fontsize = 16)
	plt.xlabel(xlabel, fontsize = 12)
	plt.ylabel(ylabel, fontsize = 12)
	plt.savefig(filename)
	plt.close()


# Decompoisition of series via additive model
def decomposeSeasonal(df):
	"""
	Simple additive model decomposition  

	:param DataFrame df: dataframe of ts to decompose
	"""
	decomp = seasonal_decompose(df, model = 'additive', freq = 4 * 24)
	trend = decomp.trend
	seasonal = decomp.seasonal
	residual = decomp.resid 

	#test = np.random.normal(2,10, 1000)

	#sm.qqplot(test, line='45')

	# print("residual:")

	count_mean = residual['count'].mean()
	count_std = residual['count'].std()

	residual['count'] = (residual['count'] - count_mean)/count_std #normalize

	residual = residual.dropna(axis = 0)

	acf_fig = plot_acf(residual, lags = 400)

	#pacf_fig = plot_pacf(residual, lags = 400)

	plt.show()


#Checking stationarity 
def checkStationarity(df):
	"""
	Running and printing output of Augmented Dickey-Fuller test for stationarity check 

	:param DataFrame df: dataframe of ts to test stationarity of
	"""
	test = adfuller(df['count'], autolag = 'AIC')
	print('ADF Statistic: %f' % test[0])
	print('p-value: %f' % test[1])
	print('Critical Values:')
	for key, value in test[4].items():
		print('\t%s: %.3f' % (key, value))


#Looking at correlations
def examineCorrelations(df, lag_count):
	"""
	Exmaining ACF, PACF of given ts with particular lag 

	:param DataFrame df: dataframe of ts to decompose
	:param int lag_count: number of lags
	"""
	acf_fig = plot_acf(df, lags = lag_count)
	saveFigure(acf_fig, title = 'ACF Plot of Observed Series with %s lags' %(str(lag_count)), xlabel = 'Timestamp', ylabel = 'Autocorrelation', filename = 'acf.png')

	pacf_fig = plot_pacf(df, lags = lag_count)
	saveFigure(pacf_fig, title = 'PACF Plot of Observed Series  with %s lags' %(str(lag_count)), xlabel = 'Timestamp', ylabel = 'Partial Autocorrelation', filename = 'pacf.png')

	plt.show()


#Using Fourier transform to look into potential multiple seasonality 
def fourierAnalyze(df):
	"""
	Apply FFT to time series and produce plot of period vs. square of amplitude of signal 

	:param DataFrame df: dataframe of ts to run FFT on
	"""
	Y = np.fft.fft(df['count'])

	# Half of sample size
	N = int(len(Y)/2+1)

	f = 1.0/(15.0*60.0) #every 15 mins
	print('f = %.4f (Frequency)' % f)

	X = np.linspace(0, f/2, N, endpoint = True)

	# Plot, using period if an hour 
	X_p = 1.0 / X
	X_p_h = X_p / (60.0 * 60.0)

	fig = plt.figure(figsize = (16, 8))
	plt.plot(X_p_h, 2.0 * np.abs(Y[:N]) / N)
	plt.xlim(0, 240)
	plt.ylim(0, 10)
	plt.xticks([6, 12, 18, 24, 30, 48, 72, 84, 96, 100, 168, 200])

	plt.show()

#Periodogram 
def analyzePeriod(df):
	"""
	Apply periodogram to time series and get top values sorted by spectral density 

	:param DataFrame df: dataframe of ts to run FFT on
	"""
	f, Pxx = signal.periodogram(df['count'], fs = 1.0/(15.0*60.0)) #every 15 mins)

	res = pd.DataFrame(dict(f = f, pxx = Pxx)).reset_index()

	res['pd_hrs'] = (1.0 / res['f']) / (60.0 * 60.0)
	res['pd_15_mins'] = (1.0 / res['f'])  / (15.0 * 60.0)
	res['pd_mins'] = (1.0 / res['f']) /(60.0)

	print("Largest values: ")
	print(res.nlargest(10, columns=['pxx']))


# Model generating via Seasonal ARIMA - grid search. Unsued because too slow
def generateModel(train_df):
	p = range(0, 3)
	d = range(0, 3)
	q = range(0, 3)

	min_aic = float("inf")
	min_param = None
	min_param_seasonal = None

	pdq = list(itertools.product(p, d, q))

	#96 was manually selected: (60/15) * 24, for daily seasonality 
	seasonal_pdq = [(x[0], x[1], x[2], 96) for x in list(itertools.product(p, d, q))]

	start_time = time()

	for param in pdq:
		for param_seasonal in seasonal_pdq:
			try:
				if(sum(param) + sum(param_seasonal) > 6): #Enforce maximum sum on parameters
					continue
				model = sm.tsa.statespace.SARIMAX(train_df, order = param, seasonal_order = param_seasonal)
				res = model.fit()
				print('ARIMA {}, {}, has AIC: {}'.format(param, param_seasonal, res.aic))
				if(res.aic < min_aic):
					min_aic = res.aic
					min_param = param
					min_param_seasonal = param_seasonal
			except:
				continue

	end_time = time()

	print("start time: %s, end time: %s, difference: %s" %(str(start_time), str(end_time), str(end_time - start_time)))

	# print("min aic: ")
	# print(min_aic)
	# print("min param: ")
	# print(min_param)
	# print("min_param_seasonal: ")
	# print(min_param_seasonal)

	return min_aic, min_param, min_param_seasonal

#RMSE
def rmse(y, y_pred):
	"""
	Compute root mean squared error for actual and forecasted values

	:param DataFrame y: actual values
	:param DataFrame y_pred: forecasted values
	"""
	return np.sqrt(np.mean((y - y_pred)**2))

#MAPE
def mape(y, y_pred):
	"""
	Compute mean average percentage error for actual and forecasted values

	:param DataFrame y: actual values
	:param DataFrame y_pred: forecasted values
	"""
	return np.mean(abs((y - y_pred)/y)) * 100

#Checking confidence intervals
def evaluateConfidenceIntervals(y):
	"""
	Compute how often confidence intervals 

	:param DataFrame y: contains yhat, y_lower, y_upper, and y
	"""
	row_count = y.shape[0]
	y['contained'] = y.apply(lambda row: 1 if row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper'] else 0, axis = 1)
	total =  y['contained'].sum()
	print("Total number of forecasts: %d" %(row_count))
	print("Total number of times confidence intervals contain actual values: %d" % total)
	return 1.0 * row_count / total


#Model using Prophet
def forecastProphet(df, training = True, in_sample_periods_ahead = 4, out_sample_periods_ahead = 4, evaluate_conf_intervals = False, evaluate_residuals = False):
	"""
	Build model and forecast using Prophet

	:param DataFrame df: actual values
	:param boolean training: training or not training
	:param int in_sample_periods_ahead: periods ahead for in-sample fit
	:param int out_sample_periods_ahead: periods ahead for out-of-sample fit
	:param boolean evaluate_conf_intervals: evaluate or not evaluate confidence intervals overlapping with actual values
	:param boolean evaluate_residuals: evaluate or not evaluate residuals
	"""

	#Split into train and test
	train_df = df.iloc[:-in_sample_periods_ahead]
	test_df = df.iloc[-in_sample_periods_ahead:]

	#Prophet requires a dataframe with two columns: 'ds' and 'y'
	train_df = train_df.reset_index()
	train_df.columns = ['ds', 'y']

	test_df = test_df.reset_index()
	test_df.columns = ['ds', 'y']

	#pd.set_option('display.max_columns', None)
	model = Prophet(changepoint_prior_scale = 0.01).fit(train_df)
	in_sample_future = model.make_future_dataframe(periods = in_sample_periods_ahead, freq = '15min')
	in_sample_forecast = model.predict(in_sample_future)

	original_forecast = in_sample_forecast

	if training:
		print("In sample forecast with %d periods ahead" % in_sample_periods_ahead) 
		in_sample_forecast = in_sample_forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
		pred = in_sample_forecast.tail(in_sample_periods_ahead).reset_index()
		print("Predictions: ")
		print(pred)
		print("RMSE: %.4f" % rmse(pred['yhat'], test_df['y']))
		print("MAPE: %.4f" %mape(pred['yhat'], test_df['y']))

		if evaluate_conf_intervals:
			y_conf = pred
			y_conf['y'] = test_df['y']
			evaluateConfidenceIntervals(y_conf)

		if evaluate_residuals:
			resid = pred
			resid['y'] = test_df['y']
			resid['residual'] = resid['yhat'] - pred['y']
			resid = resid[['ds', 'residual']]
			fig = resid.set_index('ds').plot()
			saveFigure(fig.figure, title = 'Residuals for Prophet model forecasting last 1000 time steps', xlabel = 'Timestamp', ylabel = 'Residual value', filename = 'prophet_residuals.png')

	else:
		print("Out of sample forecast with %d periods out of sample: " % out_sample_periods_ahead) #Figure 11
		out_sample_future = model.make_future_dataframe(periods = in_sample_periods_ahead + out_sample_periods_ahead, freq = '15min')
		out_sample_forecast = model.predict(out_sample_future)
		out_sample_forecast = out_sample_forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
		pred = out_sample_forecast.tail(out_sample_periods_ahead).reset_index()
		print("Predictions: ")
		print(pred)


#Main function
def evaluate():
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	data_filepath = "" #path to data file here
	quarter_hour_df = parseSeries(data_filepath, 15) 
	
	# Looking into seasonality and trend
	# decomposeSeasonal(quarter_hour_df)
	# checkStationarity(quarter_hour_df) 
	# examineCorrelations(quarter_hour_df, 100) 
	# fourierAnalyze(quarter_hour_df) 
	# analyzePeriod(quarter_hour_df) 

	# Split into test and train
	# train_df, test_df = splitTrainTest(quarter_hour_df, test_num = 4)
	# forecastProphet(quarter_hour_df, training = True, in_sample_periods_ahead = 4) 
	# forecastProphet(quarter_hour_df, training = False, in_sample_periods_ahead = 4, out_sample_periods_ahead = 4) 
	#forecastProphet(quarter_hour_df, training = True, in_sample_periods_ahead = 1000, evaluate_conf_intervals = False, evaluate_residuals = True) 
 
	#generateModel(train_df) #unused

if __name__ == "__main__":
	evaluate()



