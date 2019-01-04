import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics


'''input this information for each class instantiation:
station="Morse"'''

class Station(object):

    '''preprocesses data from an individual cta station'''

    def __init__(self, name, ride_df, map_df):
        self.name = name
        self.ride_df = ride_df
        self.map_df = map_df
        self.df = self.ride_df[self.ride_df['stationname']==self.name].reset_index(drop=True)
        self.summary = {'station':self.name}

        self.set_station_location()
        self.assign_dates()
        self.assign_seasons()
        self.get_5yr_data()
        self.create_prophet_df()

    def set_station_location(self):
        '''get longitude & latitude from map df'''
        try:
            loc_string = self.map_df[self.map_df['STATION_NAME']==self.name].reset_index().loc[0,'Location']
            lat,long = loc_string.strip('()').split(',')
        except:
            lat,long = 0,0
        if self.name == 'Washington/State':
            lat = 41.8831
            long = -87.6287
        elif self.name == 'Addison-North Main':
            lat = 41.9472
            long = -87.6536
        elif self.name == 'Monroe/State':
            lat = 41.8807
            long = -87.6277
        elif self.name == 'Kedzie-Midway':
            lat = 41.8043
            long = -87.7044
        elif self.name == 'Damen-Brown':
            lat = 41.966286
            long = -87.678639
        elif self.name == 'California/Milwaukee':
            lat = 41.921939
            long = -87.69689

        self.summary['latitude'] = float(lat)
        self.summary['longitude'] = float(long)


    def assign_dates(self):
        '''add datetime format column; drop duplicate dates (keep max rides);
        fill in any missing date values with NANs, drop extra columns'''

        df = self.df
        df['datetime'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df = df.sort_values(by=['datetime','rides'])
        df = df.drop_duplicates(subset='datetime', keep='last')
        df.set_index('datetime', drop=True, inplace=True)
        df = df.resample('D').asfreq()
        df.drop(columns=['station_id', 'date'], inplace=True)
        self.resampled_df = df

    def assign_seasons(self):
        '''extract season & rename day types, year'''

        df = self.resampled_df
        seasons = ['Winter','Winter','Spring','Spring','Spring','Summer',
                   'Summer','Summer','Fall','Fall','Fall','Winter']
        months = range(1,13)
        mtos = dict(zip(months,seasons))
        df['season'] = df.index.month.map(mtos)
        df['year'] = df.index.year
        daytypes = ['U', 'W', 'A']
        self.day_labels = ['Sun/Hol', 'Weekday', 'Sat']
        self.season_labels = list(set(seasons))

        daytypedict = dict(zip(daytypes, self.day_labels))
        df['daytype']=df['daytype'].map(daytypedict)

        num_na = df.isna().values.sum()
        if num_na:
            df.dropna(inplace=True)

        self.summary['num_na'] = num_na

        self.preprocessed = df

        self.summary['daily_mean'] = df['rides'].mean()
        self.summary['daily_std'] = df['rides'].std()

        for day in df.daytype.unique():
            self.summary[f'{day}_mean'] = df.groupby(['daytype'])['rides'].mean()[day]
            self.summary[f'{day}_std'] = df.groupby(['daytype'])['rides'].std()[day]

    def get_5yr_data(self):
        '''get ridership change rate over the past 5 [full] years'''

        df = self.preprocessed
        # get latest years mean/std
        for year in [2017, 2018]:
            self.summary[f'{year}_mean'] = df[df['year']==year]['rides'].mean()
            self.summary[f'{year}_std'] = df[df['year']==year]['rides'].std()

        df_yrdiff=pd.DataFrame()
        years_list_5 = np.arange(2017, 2012, -1)
        df_yrdiff = df[df['year'].isin(years_list_5)]
        df_yrdiff = df_yrdiff.groupby(['year'])['rides'].mean().reset_index()

        self.summary['num_yrs_from_past_5'] = len(df_yrdiff) #number of years included from past 5 years
        self.summary['5_yr_num_diff'] = df_yrdiff['rides'].diff().mean()
        self.summary['5_yr_pct_diff'] = df_yrdiff['rides'].pct_change().mean()


    def make_layered_hist(self, varname='daytype'):
        if varname == 'daytype':
            varlist = self.day_labels
        elif varname == 'season':
            varlist = self.season_labels

        fig, ax = plt.subplots()
        for var in varlist:
            condarray = self.preprocessed[(self.preprocessed[varname]==var)]
            dist=sns.distplot(condarray['rides'], ax=ax, label=var)
        ax.legend()
        ax.set_xlabel('# of Rides Daily', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.set_xlim([0,25000])
        plt.title(self.name +' Daily Ridership', fontsize=20)
        plt.tight_layout()
        dist=dist.get_figure()
        fname = 'hist' +'_' + self.name +'_'+varname+ '.png'
        fname = fname.replace('/','_')
        print(fname)
        dist.savefig('figs/'+fname)
        plt.clf()
        return fig, ax

    def create_prophet_df(self):
        '''format df for fbprophet forcasting'''

        df = self.resampled_df.reset_index()
        self.prophet_df = df[['datetime','rides']].rename(columns={'datetime':'ds', 'rides':'y'})

    def run_prophet(self):
        df = self.prophet_df
        df['cap']=40000
        df['floor']=0
        years_in_future_5=365*5+1
        final_real_date = df.iloc[-1].ds
        if final_real_date == pd.to_datetime('06-30-2018'):
            m = Prophet(growth='logistic')
            m.fit(df)
            future = m.make_future_dataframe(periods=years_in_future_5)
            future['cap']=40000
            future['floor']=0
            forecast = m.predict(future)
            forecast_data = forecast[forecast['ds']>final_real_date][['ds', 'yhat']].reset_index(
            drop=True)
            forecast_data['ds']=pd.to_datetime(forecast_data['ds'])
            forecast_data['year']=forecast_data.ds.dt.year
            self.forecast_all = forecast
            self.forecast_future = forecast_data
            self.model=m
            for row, ldf in forecast_data.groupby('year'):
                mean = ldf['yhat'].mean()
                std = ldf['yhat'].std()

                self.summary[f'{row}_predicted_mean'] = mean
                self.summary[f'{row}_predicted_std'] = std


    def run_prophet_diagnostics(self):
        self.cv = cross_validation(self.model, initial='365 days', period='180 days', horizon = '180 days')
        self.performance = performance_metrics(self.cv)
