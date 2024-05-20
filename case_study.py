# Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib as plt
import datetime as dt

# connect to database
con = sql.connect("grains.db")

# create a cursor
cur = con.cursor()

#pd.read_excel('merged_specialty_grain_revenue.xlsx').to_sql("specialty_grains",con,if_exists='append',index=False)
#pd.read_csv('en_climate_monthly_SK_estevan.csv').to_sql("historical_weather",con,if_exists='append',index=False)


grains_df = pd.read_sql(f'SELECT * FROM {"specialty_grains"}',con)
# Only pull rows that are between 1993 and 2015
weather_df = pd.read_sql(f'SELECT * FROM {"historical_weather"} WHERE Year BETWEEN 1993 AND 2015',con)
# checking how many tables in db
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())
print(weather_df["Year"])
print(weather_df.info())
print(grains_df.info())

# Processing the grains data set

# 
grain_columns = ['Oats 2CW ($ per tonne)', 'CW Feed ($ per tonne)', 'Flax 1CAN ($ per tonne)', 'Canola 1CAN ($ per tonne)',
                 'Chickpeas 1CW - Kabuli 9mm ($ per cwt)', 'Chickpeas 1CW - Desi ($ per cwt)', 'Field Peas 1CAN - Yellow ($ per bu)',
                 'Field Peas 1CAN - Green ($ per bu)', 'Field Peas 1CAN - Feed ($ per bu)', 'Canada Canary Seed ($ per cwt)',
                 'Lentils Small Red ($ per cwt)', 'Lentils Large Green ($ per cwt)', 'Lentils Small Green ($ per cwt)', 'Lentils Medium Green ($ per cwt)',
                 'Lentils French Green ($ per cwt)', 'Mustard 1CAN - Yellow ($ per cwt)', 'Mustard 1CAN - Brown ($ per cwt)', 'Mustard 1CAN - Oriental ($ per cwt)']

y = pd.to_datetime(grains_df['Date']).dt.year.rename('year')
print(y)
grains_yearly_df = grains_df.groupby(y)[grain_columns].mean().reset_index()
print(grains_yearly_df.head())


