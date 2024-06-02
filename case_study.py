# Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
grain_years = pd.to_datetime(grains_df['Date']).dt.year.rename('Year')
grains_yearly_df = grains_df.groupby(grain_years)[grain_columns].mean().reset_index()
pd.options.display.max_columns = None
print(grains_yearly_df.head())

# Get cpi values per year
cpi_years = pd.to_datetime(cpi_df['REF_DATE']).dt.year.rename('Year')
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
base_cpi = cpi_yearly_values_df[cpi_yearly_values_df['Year'] == base_year]['VALUE'].values[0]
merged_df = pd.merge(grains_yearly_df, cpi_yearly_values_df, on='Year')
print("Merged df\n", merged_df)

# Inserting real price columns for each grain
for real_col, grain_col in zip(real_grain_columns, grain_columns):
    merged_df[real_col] = merged_df[grain_col] * (base_cpi / merged_df['VALUE'])

print(merged_df)

# inserting new df to db
#merged_df.to_sql("grain_real_prices", con, if_exists='append', index=False)


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

# Checking how many months in a year
months_per_year = weather_df.groupby('Year')['Month'].nunique().reset_index(name='UniqueMonths')
print(months_per_year)

# Group by years for analysis
weather_columns = ["Mean Max Temp (°C)", "Mean Min Temp (°C)", "Mean Temp (°C)", "Extr Max Temp (°C)", "Extr Min Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Snow Grnd Last Day (cm)", "Spd of Max Gust (km/h)"]
weather_years = pd.to_datetime(weather_df['Date/Time']).dt.year.rename('Year')
weather_yearly_df = weather_df.groupby(weather_years)[weather_columns].mean().reset_index()
print(weather_yearly_df)

# Since 1997 was dropped from the grains and cpi df and 2015 only has 2 months in weather

# Drop 1997 from weather yearly df
weather_yearly_df = weather_yearly_df.drop(4)

# Drop 2015 from the merged df and the weather_df
weather_yearly_df = weather_yearly_df.drop(22)
merged_df = merged_df.drop(21)
merged_df = merged_df.drop(columns=grain_columns)

weather_yearly_df = weather_yearly_df.reset_index(drop=True)
merged_df = merged_df.reset_index(drop=True)
print(merged_df)
print(weather_yearly_df)

# Analyze phase

# visualizing the changes in weather overtime using a line graph for trends overtime
plt.figure(figsize=(10,6))
plt.plot(weather_yearly_df['Year'], weather_yearly_df['Mean Temp (°C)'], marker='o', linestyle='-', label='Mean Temp (°C)')
plt.plot(weather_yearly_df['Year'], weather_yearly_df['Mean Max Temp (°C)'], marker='o', linestyle='-', label='Mean Max Temp (°C)')
plt.plot(weather_yearly_df['Year'], weather_yearly_df['Mean Min Temp (°C)'], marker='o', linestyle='-', label='Mean Min Temp (°C)')
print(weather_yearly_df['Year'])

# Customize the plot
plt.title('Mean Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Temp (°C)')
plt.legend()
# Display the plot
plt.show()

# Line graphs for precipitation
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(weather_yearly_df['Year'], weather_yearly_df['Total Rain (mm)'], marker='o', linestyle='-', label='Total Rain (mm)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Rain (mm)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(weather_yearly_df['Year'], weather_yearly_df['Total Snow (cm)'], marker='o', linestyle='-', color='red', label='Total Snow (cm)')
ax2.plot(weather_yearly_df['Year'], weather_yearly_df['Snow Grnd Last Day (cm)'], marker='o', linestyle='-',color='green', label='Snow Grnd Last Day (cm)')
ax2.set_ylabel('Snow (cm)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Mean Precipitation Over Time')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

plt.show()

# Line graph for wind over time
plt.figure(figsize=(10,6))
plt.plot(weather_yearly_df['Year'], weather_yearly_df['Spd of Max Gust (km/h)'], marker='o', linestyle='-', label='Spd of Max Gust (km/h)')

plt.title('Mean Wind Speed Over Time')
plt.xlabel('Year')
plt.ylabel('Speed of Gust (km/h)')
plt.show()

# Initial visualizations for crop prices
# initial visualization for crops with $ per tonne
plt.figure(figsize=(10,6))
plt.plot(merged_df['Year'], merged_df['Real Oats 2CW Price (2015 $ per tonne)'], marker='o', linestyle='-', label='Oats')
plt.plot(merged_df['Year'], merged_df['Real CW Feed Price (2015 $ per tonne)'], marker='o', linestyle='-', label='Wheat')
plt.plot(merged_df['Year'], merged_df['Real Flax 1CAN Price (2015 $ per tonne)'], marker='o', linestyle='-', label='Flax')
plt.plot(merged_df['Year'], merged_df['Real Canola 1CAN Price (2015 $ per tonne)'], marker='o', linestyle='-', label='Canola')

# Customize the plot
plt.title('Mean Price Over Time')
plt.xlabel('Year')
plt.ylabel('2015 $ per Tonne')
plt.legend()
# Display the plot
plt.show()

# Initial visualization for field peas
plt.figure(figsize=(10,6))
plt.plot(merged_df['Year'], merged_df['Real Field Peas 1CAN - Yellow Price (2015 $ per bu)'], marker='o', color='yellow', linestyle='-', label='Yellow')
plt.plot(merged_df['Year'], merged_df['Real Field Peas 1CAN - Green Price (2015 $ per bu)'], marker='o', color='green', linestyle='-', label='Green')
plt.plot(merged_df['Year'], merged_df['Real Field Peas 1CAN - Feed Price (2015 $ per bu)'], marker='o', linestyle='-', label='Feed')

# Customize the plot
plt.title('Mean Price of Peas Over Time')
plt.xlabel('Year')
plt.ylabel('2015 $ per Bushel')
plt.legend()
# Display the plot
plt.show()

# Initial visualization for lentils
plt.figure(figsize=(10,6))
plt.plot(merged_df['Year'], merged_df['Real Lentils Large Green Price (2015 $ per cwt)'], marker='o', linestyle='-', label='Large Green')
plt.plot(merged_df['Year'], merged_df['Real Lentils Small Green Price (2015 $ per cwt)'], marker='o', linestyle='-', label='Small Green')
plt.plot(merged_df['Year'], merged_df['Real Lentils Medium Green Price (2015 $ per cwt)'], marker='o', linestyle='-', label='Medium Green')
plt.plot(merged_df['Year'], merged_df['Real Lentils French Green Price (2015 $ per cwt)'], marker='o', linestyle='-', label='French Green')

# Customize the plot
plt.title('Mean Price of Lentils Over Time')
plt.xlabel('Year')
plt.ylabel('2015 $ per CWT')
plt.legend()

plt.show()

# Initial visualization for mustard prices
plt.figure(figsize=(10,6))
plt.plot(merged_df['Year'], merged_df['Real Mustard 1CAN - Yellow Price (2015 $ per cwt)'], marker='o', linestyle='-', label='Yellow')
plt.plot(merged_df['Year'], merged_df['Real Mustard 1CAN - Brown (2015 $ per cwt) Price'], marker='o', linestyle='-', label='Brown')
plt.plot(merged_df['Year'], merged_df['Real Mustard 1CAN - Oriental (2015 $ per cwt) Price'], marker='o', linestyle='-', label='Oriental')

# Customize the plot
plt.title('Mean Price of Mustard Over Time')
plt.xlabel('Year')
plt.ylabel('2015 $ per CWT')
plt.legend()

plt.show()

# Initial visualization for canary seed
plt.figure(figsize=(10,6))
plt.plot(merged_df['Year'], merged_df['Real Canada Canary Seed Price (2015 $ per cwt)'], marker='o', linestyle='-', label='Canary Seed')

plt.title('Mean Price of Canary Seed Over Time')
plt.xlabel('Year')
plt.ylabel('2015 $ per CWT')
plt.legend()

plt.show()


# initial visualization to see correlation of mean temp for crop prices

# to provide a more precise analysis I will use a linear regression model
# Fitting linear regression models
x_years = weather_yearly_df['Year']

# Mean Temp
