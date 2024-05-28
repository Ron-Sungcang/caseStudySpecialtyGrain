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
# Check the weather table
cur.execute("PRAGMA table_info(historical_weather)")
col_info = cur.fetchall()
col_names = [info[1] for info in col_info]
print(col_names)
# Only pull rows that are between 1993 and 2015 and with important columns
weather_df = pd.read_sql(f'SELECT "Date/Time", Year, "Mean Max Temp (°C)", "Mean Min Temp (°C)", "Mean Temp (°C)", '
                         f'"Extr Max Temp (°C)", "Extr Min Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", '
                         f'"Snow Grnd Last Day (cm)", "Spd of Max Gust (km/h)" FROM {"historical_weather"} WHERE Year '
                         f'BETWEEN 1993 AND 2015', con)
# checking how many tables in db
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())
print("Weather Dataframe\n", weather_df)
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
grains_missing_percentage = (grains_missing_values_count[grain_columns] / len(grains_df)) * 100
print(grains_missing_percentage[:])

# Dropping the columns with 26% missing data
grains_df = grains_df.drop(columns=['Chickpeas 1CW - Kabuli 9mm ($ per cwt)', 'Chickpeas 1CW - Desi ($ per cwt)',
                                    'Lentils Small Red ($ per cwt)'])

grain_columns.remove('Chickpeas 1CW - Kabuli 9mm ($ per cwt)')
grain_columns.remove('Chickpeas 1CW - Desi ($ per cwt)')
grain_columns.remove('Lentils Small Red ($ per cwt)')

# checking for rows where they are all null
cols_to_check = grains_df.columns.drop('Date')
condition = grains_df[cols_to_check].isnull().all(axis=1)
rows_null = grains_df[condition]
print("ROWS NULL \n", rows_null)

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

yearly_missing_percentage = (yearly_missing_values / total_observations_per_year) * 100
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
                      'Real Field Peas 1CAN - Green Price (2015 $ per bu)',
                      'Real Field Peas 1CAN - Feed Price (2015 $ per bu)',
                      'Real Canada Canary Seed Price (2015 $ per cwt)',
                      'Real Lentils Large Green Price (2015 $ per cwt)',
                      'Real Lentils Small Green Price (2015 $ per cwt)',
                      'Real Lentils Medium Green Price (2015 $ per cwt)',
                      'Real Lentils French Green Price (2015 $ per cwt)',
                      'Real Mustard 1CAN - Yellow Price (2015 $ per cwt)',
                      'Real Mustard 1CAN - Brown (2015 $ per cwt) Price',
                      'Real Mustard 1CAN - Oriental (2015 $ per cwt) Price']

# Selecting a base year for real prices calculations
base_year = 2015
base_cpi = cpi_yearly_values_df[cpi_yearly_values_df['year'] == base_year]['VALUE'].values[0]
merged_df = pd.merge(grains_yearly_df, cpi_yearly_values_df, on='year')
print("Merged df\n", merged_df)

# Inserting real price columns for each grain
for real_col, grain_col in zip(real_grain_columns, grain_columns):
    merged_df[real_col] = merged_df[grain_col] * (base_cpi / merged_df['VALUE'])

print(merged_df)

# inserting new df to db
merged_df.to_sql("grain_real_prices", con, if_exists='append', index=False)

# Processing the weather dataset
print("Weather Dataframe\n", weather_df.info())

# Checking for null values in the dataset
weather_missing_values_count = weather_df.isnull().sum()
print(weather_missing_values_count[:])

# Finding row where snow grnd last day is null
condition = weather_df["Snow Grnd Last Day (cm)"].isnull()
print(weather_df[condition])

# Finding all rows with february in the past five years to fill the missing data
february_value = \
weather_df[(weather_df['Date/Time'].str.contains('-02')) & (weather_df['Year'] < 2015) & (weather_df['Year'] >= 2011)][
    "Snow Grnd Last Day (cm)"]
average_feb_values = february_value.mean()
print(average_feb_values)
weather_df['Snow Grnd Last Day (cm)'].fillna(average_feb_values, inplace=True)
print(weather_df)

# finding rows where speed of max gust wind is null
wind_condition = weather_df["Spd of Max Gust (km/h)"].isnull()
print(weather_df[wind_condition])

# Finding null values per month
weather_df["Month"] = weather_df['Date/Time'].str[5:7].astype(int)
missing_count_by_year_month = weather_df[weather_df['Spd of Max Gust (km/h)'].isna()].groupby(
    ["Year", "Month"]).size().reset_index(name='MissingCount')
print(missing_count_by_year_month)

# Convert 'Spd of Max Gust (km/h)' to numeric, coercing errors to NaN
weather_df['Spd of Max Gust (km/h)'] = pd.to_numeric(weather_df['Spd of Max Gust (km/h)'], errors='coerce')

# Impute missing values by seasonal average
monthly_average = weather_df.groupby('Month')['Spd of Max Gust (km/h)'].mean()
print(monthly_average)

def impute_monthly_average(df, target_col, group_averages):
    """
    Takes a dataframe a target column, and a dataframe of averages per month to impute missing values with seasonal average
    """
    # Apply to each row
    def impute_row(row):
        if pd.isnull(row[target_col]):
            return group_averages[row['Month']]
        else:
            return row[target_col]

    # Apply the function to impute missing values
    df[target_col] = df.apply(impute_row, axis=1)


# Use the generalized function to impute missing values in 'Spd of Max Gust (km/h)' based on 'Month'
impute_monthly_average(weather_df, 'Spd of Max Gust (km/h)',monthly_average)

# Checking for null values in the dataset
weather_missing_values_count = weather_df.isnull().sum()
print(weather_missing_values_count[:])

print(weather_df.iloc[171])