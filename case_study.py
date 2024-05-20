# Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib as plt
import datetime as dt

# connect to database
con = sql.connect("grains.db")

# create a cursor
cur = con.cursor()

pd.read_excel('merged_specialty_grain_revenue.xlsx').to_sql("specialty_grains",con,if_exists='append',index=False)
pd.read_csv('en_climate_monthly_SK_estevan.csv').to_sql("historical_weather",con,if_exists='append',index=False)


grains_df = pd.read_sql(f'SELECT * FROM {"specialty_grains"}',con)
weather_df = pd.read_sql(f'SELECT * FROM {"historical_weather"}',con)
# checking how many tables in db
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

