"""
This module is responsible for implementing all the econometric models specified in the article 
for the statistical study of data on cds spreads. 

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AutoRegModelsDf:
    def __init__(self, df, exog_data, max_lag):
        """
        Initialize the AutoRegModels class with data and configuration.

        Args:
        df : DataFrame containing the time series data for multiple series.
        exog_data : Series or array containing the exogenous variable.
        max_lag : int, the maximum lag order to be considered in the AutoReg model.
        """
        self.df = df
        self.exog_data = exog_data
        self.max_lag = max_lag
        self.results = pd.DataFrame()  # DataFrame to store the results

    def fit_models(self):
        """
        Fit an AutoReg model to each column in the DataFrame with dynamic lag specification.
        """
        # Add constant to the exogenous data
        exog = sm.add_constant(self.exog_data)
        
        # Iterate over each column in the DataFrame
        results_list = []
        for title in self.df.columns:
            data = self.df[title]
            model = sm.tsa.AutoReg(endog=data, lags=self.max_lag, exog=exog, old_names=False)
            results = model.fit()
            
            # Prepare a dictionary to store results
            results_dict = {
                'title': title,
                **{f'coef_lag{i}': coef for i, coef in enumerate(results.params, start=1)},
                **{f'pvalue_lag{i}': pval for i, pval in enumerate(results.pvalues, start=1)},
                **{f'tvalue_lag{i}': tval for i, tval in enumerate(results.tvalues, start=1)}
            }
            results_list.append(results_dict)

        # Create the final DataFrame from the list of dictionaries
        self.results = pd.DataFrame(results_list)
        self.results.set_index('title', inplace=True)

    def get_results(self):
        """
        Returns the DataFrame containing the coefficients, p-values, and t-values for each series.
        """
        return self.results
    



class ARXModel:
    def __init__(self, df_main, df_market, lags):
        """
        Initialize the ARXModel class with dataframes and lag order for the main variable.
        The market variable is always included with exactly one lag.
        
        :param df_main: A pandas DataFrame containing the main time series data.
        :param df_market: A pandas DataFrame containing the market variable data.
        :param lags: An integer indicating the number of lags for the main variable in the AR model.
        """
        self.df_main = df_main
        self.df_market = df_market
        self.lags = lags
        self.model = None
        self.results = None
    
    def fit(self):
        """
        Fit the ARX model using the provided main series and market data.
        Only one market lag is used regardless of the number of main lags.
        """
        # Prepare data by merging and aligning indices
        self.df_main['MarketVarLag1'] = self.df_market['MarketVar'].shift(1)
        combined_df = pd.concat([self.df_main, self.df_market.shift(1)], axis=1).dropna()

        # Define the endogenous and exogenous variables
        endog = combined_df['MainVar']
        exog = combined_df['MarketVarLag1']  # Only one lag for the market variable
        
        # Fit the model
        self.model = sm.tsa.AutoReg(endog, lags=self.lags, exog=exog, old_names=False)
        self.results = self.model.fit()

    def get_residuals(self):
        """
        Extract residuals from the fitted model.
        
        :return: A numpy array of residuals.
        """
        if self.results is not None:
            return self.results.resid
        else:
            raise ValueError("Model has not been fitted yet.")
    
    def plot_residuals(self):
        """
        Plot the residuals of the fitted model.
        """
        if self.results is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.get_residuals())
            plt.title('Residuals from ARX Model')
            plt.xlabel('Time')
            plt.ylabel('Residuals')
            plt.show()
        else:
            raise ValueError("Model has not been fitted yet.")
        

    def get_model_summary(self):
        """
        Extract model parameters, p-values, etc., and return them as a DataFrame.
        """
        if self.results is not None:
            # Extracting the parameters, p-values, standard errors, and confidence intervals
            params = self.results.params
            pvalues = self.results.pvalues
            stderr = self.results.bse
            conf_int = self.results.conf_int()

            # Creating a DataFrame
            df_summary = pd.DataFrame({
                'Coefficient': params,
                'P-value': pvalues,
                'Std Err': stderr,
                'CI Lower': conf_int[:, 0],
                'CI Upper': conf_int[:, 1]
            })

            # Adding variable names as the index
            df_summary.index = ['Lag_' + str(i + 1) if i < self.lags else 'MarketVar_Lag1' for i in range(len(df_summary))]

            return df_summary
        else:
            raise ValueError("Model has not been fitted yet.")



class SimpleAR:
    def __init__(self, dataframe, lags):
        """
        Initialize the SimpleAR class with a dataframe and the number of lags for the AR model.
        
        :param dataframe: A pandas DataFrame containing the time series data.
        :param lags: An integer indicating the number of lags in the AR model.
        """
        self.dataframe = dataframe
        self.lags = lags
        self.model = None
        self.results = None
    
    def fit(self):
        """
        Fit the AR model using the provided time series data.
        """
        # Make sure the dataframe has the correct column name for simplicity
        if 'value' not in self.dataframe.columns:
            raise ValueError("DataFrame must contain 'value' column with time series data")
        
        # Fit the model
        self.model = sm.tsa.AutoReg(self.dataframe['value'], lags=self.lags, old_names=False)
        self.results = self.model.fit()

    def get_residuals(self):
        """
        Extract residuals from the fitted model.
        
        :return: A numpy array of residuals.
        """
        if self.results is not None:
            return self.results.resid
        else:
            raise ValueError("Model has not been fitted yet.")
    
    def plot_residuals(self):
        """
        Plot the residuals of the fitted model.
        """
        if self.results is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.get_residuals(), label='Residuals')
            plt.title('Residuals from AR Model')
            plt.xlabel('Time')
            plt.ylabel('Residuals')
            plt.legend()
            plt.show()
        else:
            raise ValueError("Model has not been fitted yet.")

    def get_model_summary(self):
        """
        Extract model parameters, p-values, etc., and return them as a DataFrame.
        """
        if self.results is not None:
            # Extracting the parameters, p-values, standard errors, and confidence intervals
            params = self.results.params
            pvalues = self.results.pvalues
            stderr = self.results.bse
            conf_int = self.results.conf_int()

            # Creating a DataFrame
            df_summary = pd.DataFrame({
                'Coefficient': params,
                'P-value': pvalues,
                'Std Err': stderr,
                'CI Lower': conf_int[:, 0],
                'CI Upper': conf_int[:, 1]
            })

            return df_summary
        else:
            raise ValueError("Model has not been fitted yet.")
        
    


class GJRGARCH:
    def __init__(self, df_returns, df_market_residuals, lags):
        """
        Initialize the GJR-GARCH model with returns, market residuals, and the number of lags.
        
        :param df_returns: A pandas DataFrame containing the return series data.
        :param df_market_residuals: A pandas DataFrame containing the market residuals data.
        :param lags: An integer indicating the number of lags in the GARCH model.
        """
        self.df_returns = df_returns
        self.df_market_residuals = df_market_residuals
        self.lags = lags
        self.parameters = None
        self.residuals = None
    
    def fit(self):
        """
        Fit the GJR-GARCH model using the provided return series and market residuals data.
        """
        def gjr_garch_likelihood(params, returns, market_residuals):
            omega, alpha, gamma, delta, theta, kappa = params
            n = len(returns)
            sigma2 = np.full(n, 0.1)  # Initial variance estimate
            
            for t in range(1, n):
                residual = returns[t-1]
                market_residual = market_residuals[t-1]
                sigma2[t] = (omega +
                             alpha * sigma2[t-1] +
                             gamma * residual**2 +
                             delta * residual**2 * (residual > 0) +
                             theta * market_residual**2 +
                             kappa * market_residual**2 * (market_residual > 0))
            log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
            return -np.sum(log_likelihood)
        
        returns = self.df_returns['returns'].values
        market_residuals = self.df_market_residuals['market_residuals'].values
        initial_params = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        bounds = [(0, None), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        result = minimize(gjr_garch_likelihood, initial_params, args=(returns, market_residuals), bounds=bounds, method='L-BFGS-B')
        
        self.parameters = result.x
        self.residuals = returns - np.sqrt(self.conditional_volatility(result.x, returns, market_residuals))  # Update to calculate residuals

    def conditional_volatility(self, params, returns, market_residuals):
        omega, alpha, gamma, delta, theta, kappa = params
        n = len(returns)
        sigma2 = np.full(n, 0.1)
        
        for t in range(1, n):
            residual = returns[t-1]
            market_residual = market_residuals[t-1]
            sigma2[t] = (omega +
                         alpha * sigma2[t-1] +
                         gamma * residual**2 +
                         delta * residual**2 * (residual > 0) +
                         theta * market_residual**2 +
                         kappa * market_residual**2 * (market_residual > 0))
        return sigma2

    def get_residuals(self):
        return self.residuals

    def get_model_summary(self):
        return pd.DataFrame({'Parameter': ['omega', 'alpha', 'gamma', 'delta', 'theta', 'kappa'],
                             'Value': self.parameters})





