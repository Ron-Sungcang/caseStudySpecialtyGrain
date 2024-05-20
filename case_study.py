#Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib as plt
import datetime as dt

grain_df = pd.read_excel('merged_specialty_grain_revenue.xlsx')
print(grain_df.head())

weather_df = pd.read_csv('en_climate_monthly_SK_estevan.csv')
print(weather_df.head())

grain_df.info()
weather_df.info()