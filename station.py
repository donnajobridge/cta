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
        self.set_station_location()
        self.assign_dates()
        self.assign_seasons()
        self.create_prophet_df()

    def set_station_location(self):
        '''get longitude & latitude from map df'''

        loc_string = self.map_df[self.map_df['STATION_NAME']==self.name].reset_index().loc[0,'Location']
        lat,long = loc_string.strip('()').split(',')
        self.latitude = float(lat)
        self.longitude = float(long)

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
        # extract season & rename day types,
        df = self.resampled_df
        seasons = ['Winter','Winter','Spring','Spring','Spring','Summer',
                   'Summer','Summer','Fall','Fall','Fall','Winter']
        months = range(1,13)
        mtos = dict(zip(months,seasons))
        df['season'] = df.index.month.map(mtos)
        daytypes = df.daytype.unique().tolist()
        daytypedict = dict(zip(daytypes, ['Sun/Hol', 'Weekday', 'Sat']))
        df['daytype']=df['daytype'].map(daytypedict)
        self.preprocessed = df


    def create_prophet_df(self):
        df = self.resampled_df.reset_index()
        self.prophet_df = df[['datetime','rides']].rename(columns={'datetime':'ds', 'rides':'y'})
