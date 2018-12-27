import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        elif self.name =='Addison-North Main':
            lat = 41.9472
            long = -87.6536

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
