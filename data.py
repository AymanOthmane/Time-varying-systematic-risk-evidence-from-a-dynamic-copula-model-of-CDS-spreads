import pandas as pd
import numpy as np

class Univers:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def read_data(self):
        if self.filepath.endswith('.csv'):
            self.data = pd.read_csv(self.filepath)
        elif self.filepath.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.filepath)
        else:
            raise ValueError("Unsupported file format. Please use a CSV or Excel file.")
        
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
        else:
            raise ValueError("Data must contain a 'Date' column.")

    def build_univers_log_returns(self):
        if self.data is None:
            self.read_data()
        log_returns = np.log(self.data / self.data.shift(1))
        return log_returns

    def build_univers_simple_returns(self):
        if self.data is None:
            self.read_data()
        simple_returns = self.data.pct_change()
        return simple_returns

    def build_market_column(self):
        """
        Calculates the equally-weighted mean of all assets for each period.
        
        Returns:
        DataFrame with dates as index and a 'market' column representing the mean value.
        """
        if self.data is None:
            self.read_data()
        market_df = pd.DataFrame()
        market_df['market'] = self.data.mean(axis=1)
        return market_df

    def market_log_returns(self):
        """
        Calculates the log returns of the market column.
        
        Returns:
        DataFrame with dates as index and a 'market_log_return' column representing the log returns.
        """
        market_df = self.build_market_column()
        market_log_returns_df = pd.DataFrame(index=market_df.index)
        market_log_returns_df['market_log_return'] = np.log(market_df['market'] / market_df['market'].shift(1))
        return market_log_returns_df


