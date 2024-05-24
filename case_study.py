# Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib as plt
import datetime as dt

# connect to database
con = sql.connect("grains.db")

# create a cursor
cur = con.cursor()

# Load spreadsheets into sql
# pd.read_excel('merged_specialty_grain_revenue.xlsx').to_sql("specialty_grains",con,if_exists='append',index=False)
# pd.read_csv('en_climate_monthly_SK_estevan.csv').to_sql("historical_weather",con,if_exists='append',index=False)
# pd.read_csv('cpi.csv').to_sql("cpi", con, if_exists='append',index=False)


grains_df = pd.read_sql(f'SELECT * FROM {"specialty_grains"}', con)
cpi_df = pd.read_sql(f'SELECT * FROM {"cpi"}', con)
# Only pull rows that are between 1993 and 2015
weather_df = pd.read_sql(f'SELECT * FROM {"historical_weather"} WHERE Year BETWEEN 1993 AND 2015', con)
# checking how many tables in db
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())
print("Weather Dataframe\n", weather_df.info())
print("Specialty Grains Dataframe", grains_df.info())
print("CPI dataframe", cpi_df.info())

# Processing the grains data set

grain_columns = ['Oats 2CW ($ per tonne)', 'CW Feed ($ per tonne)', 'Flax 1CAN ($ per tonne)',
                 'Canola 1CAN ($ per tonne)',
                 'Chickpeas 1CW - Kabuli 9mm ($ per cwt)', 'Chickpeas 1CW - Desi ($ per cwt)',
                 'Field Peas 1CAN - Yellow ($ per bu)',
                 'Field Peas 1CAN - Green ($ per bu)', 'Field Peas 1CAN - Feed ($ per bu)',
                 'Canada Canary Seed ($ per cwt)',
                 'Lentils Small Red ($ per cwt)', 'Lentils Large Green ($ per cwt)', 'Lentils Small Green ($ per cwt)',
                 'Lentils Medium Green ($ per cwt)',
                 'Lentils French Green ($ per cwt)', 'Mustard 1CAN - Yellow ($ per cwt)',
                 'Mustard 1CAN - Brown ($ per cwt)', 'Mustard 1CAN - Oriental ($ per cwt)']

# Checking for null values in grains dataset
grains_missing_values_count = grains_df.isnull().sum()
print(grains_missing_values_count[:])
grains_missing_percentage = (grains_missing_values_count[grain_columns]/len(grains_df)) * 100
print(grains_missing_percentage[:])

# Dropping the columns with 26% missing data
grains_df = grains_df.drop(columns=['Chickpeas 1CW - Kabuli 9mm ($ per cwt)', 'Chickpeas 1CW - Desi ($ per cwt)', 'Lentils Small Red ($ per cwt)'])

grain_columns.remove('Chickpeas 1CW - Kabuli 9mm ($ per cwt)')
grain_columns.remove('Chickpeas 1CW - Desi ($ per cwt)')
grain_columns.remove('Lentils Small Red ($ per cwt)')

# checking for rows where they are all null
cols_to_check = grains_df.columns.drop('Date')
condition = grains_df[cols_to_check].isnull().all(axis=1)
rows_null=grains_df[condition]
print("ROWS NULL \n",rows_null)

# Dropping rows that are all null
grains_df = grains_df.drop(416)
grains_df = grains_df.drop(521)

# Change grains to yearly
grain_years = pd.to_datetime(grains_df['Date']).dt.year.rename('year')
grains_yearly_df = grains_df.groupby(grain_years)[grain_columns].mean().reset_index()
pd.options.display.max_columns = None
print(grains_yearly_df.head())

# Get cpi values per year
cpi_years = pd.to_datetime(cpi_df['REF_DATE']).dt.year.rename('year')
cpi_yearly_values_df = cpi_df.groupby(cpi_years)["VALUE"].mean().reset_index()
print(cpi_yearly_values_df)

# Checking the amount of null values per year to drop extreme rows
yearly_missing_values = grains_df.groupby(grain_years).apply(lambda x: x.isnull().sum().sum())
print(yearly_missing_values)

# Calculate the total number of observations per year (rows * columns)
total_observations_per_year = grains_df.groupby(grain_years).apply(lambda x: x.shape[0] * x.shape[1])

yearly_missing_percentage = (yearly_missing_values/total_observations_per_year) * 100
print(yearly_missing_percentage)

# Dropping the year 1997 from cpi and grains df
grains_yearly_df = grains_yearly_df.drop(4)
cpi_yearly_values_df = cpi_yearly_values_df.drop(4)
print(grains_yearly_df)
print(cpi_yearly_values_df)

# Calculate for real prices
real_grain_columns = ['Real Oats 2CW Price (2015 $ per tonne)', 'Real CW Feed Price (2015 $ per tonne)',
                      'Real Flax 1CAN Price (2015 $ per tonne)', 'Real Canola 1CAN Price (2015 $ per tonne)',
                      'Real Field Peas 1CAN - Yellow Price (2015 $ per bu)',
                      'Real Field Peas 1CAN - Green Price (2015 $ per bu)', 'Real Field Peas 1CAN - Feed Price (2015 $ per bu)',
                      'Real Canada Canary Seed Price (2015 $ per cwt)', 'Real Lentils Large Green Price (2015 $ per cwt)',
                      'Real Lentils Small Green Price (2015 $ per cwt)', 'Real Lentils Medium Green Price (2015 $ per cwt)',
                      'Real Lentils French Green Price (2015 $ per cwt)', 'Real Mustard 1CAN - Yellow Price (2015 $ per cwt)',
                      'Real Mustard 1CAN - Brown (2015 $ per cwt) Price', 'Real Mustard 1CAN - Oriental (2015 $ per cwt) Price']

# Selecting a base year for real prices calculations
base_year = 2015
base_cpi = cpi_yearly_values_df[cpi_yearly_values_df['year'] == base_year]['VALUE'].values[0]
merged_df = pd.merge(grains_yearly_df,cpi_yearly_values_df,on='year')
print("Merged df\n",merged_df)

# Inserting real price columns for each grain
for real_col,grain_col in zip(real_grain_columns,grain_columns):
    merged_df[real_col] = merged_df[grain_col] * (base_cpi/merged_df['VALUE'])

print(merged_df)

# inserting new df to db
merged_df.to_sql("grain_real_prices",con,if_exists='append',index=False)

