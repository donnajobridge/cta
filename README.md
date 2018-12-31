## Analysis of CTA train data

#cta_eda.ipynb
This notebook conducts some initial exploratory analysis of the train data.


#run_station_class.ipynb
Calls the Station class to preprocess the train data for each station. A summary dataframe is created, which includes key metrics. Additionally, fbprophet forecast model is applied to data from each station to obtain estimates of ridership growth in the next 5 years (second half of 2018 to 2023).


#station_summary_analysis.ipynb
Reads in the station summary dataframe and conducts some analysis. First I plot the stations with the highest daily ridership. Next I examine the stanard deviation of Washington/Wabash and explore the distribution of riders across days (weekends and weekdays) and seasons. A scatter plot shows that stations with large differences in Saturday vs. Weekday ridership tend to have the largest standard deviations.

#select_pie_location.ipynb
Finds the best CTA station location to open up a new pie shop. An interactive scatterplot shows stations plotted with mean daily ridership in 2018 on the x axis and projected ridership growth from 2018 to 2023 on the y axis. Stations with midrange mean daily ridership in 2018 were targeted. Out of these midrange stations, I chose the stations with the highest projected growth in 2023. These stations are highlighted in green.

I next applied the fbprophet forecast model to the selected stations. A plot for each station is made showing historical data and projected ridership through 2023.


#cta_maps.ipynb
Creates a Chicago map with recommended pie shop locations and their corresponding neighborhoods highlighted.
