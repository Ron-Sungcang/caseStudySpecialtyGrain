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
# merged_df.to_sql("grain_real_prices", con, if_exists='append', index=False)


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
    weather_df[
        (weather_df['Date/Time'].str.contains('-02')) & (weather_df['Year'] < 2015) & (weather_df['Year'] >= 2011)][
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
impute_monthly_average(weather_df, 'Spd of Max Gust (km/h)', monthly_average)

# Checking for null values in the dataset
weather_missing_values_count = weather_df.isnull().sum()
print(weather_missing_values_count[:])

print(weather_df.iloc[171])

# Checking how many months in a year
months_per_year = weather_df.groupby('Year')['Month'].nunique().reset_index(name='UniqueMonths')
print(months_per_year)

# Group by years for analysis
weather_columns = ["Mean Max Temp (°C)", "Mean Min Temp (°C)", "Mean Temp (°C)", "Extr Max Temp (°C)",
                   "Extr Min Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Snow Grnd Last Day (cm)",
                   "Spd of Max Gust (km/h)"]
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
# Columns for weather
mean_temp_cols = weather_columns[0:3]
extreme_temp_cols = weather_columns[3:5]

# Columns for grains prices
price_per_tonnes = real_grain_columns[0:4]
price_peas_columns = real_grain_columns[4:7]
price_lentils_columns = real_grain_columns[8:12]
price_mustard_columns = real_grain_columns[12:15]


# visualizing the changes in weather overtime using a line graph for trends overtime
def plot_trends(df, x_col, y_cols, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    for y in y_cols:
        plt.plot(df[x_col], df[y], marker='o', linestyle='-', label=y)

    # Customize the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    # Display the plot
    plt.show()


plot_trends(weather_yearly_df, "Year", mean_temp_cols, "Mean Temp Over Time", "Year", "Temp in Celsius")
plot_trends(weather_yearly_df, "Year", extreme_temp_cols, "Mean Extreme Temp Over Time", "Year", "Temp in Celsius")

# Line graphs for precipitation
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(weather_yearly_df['Year'], weather_yearly_df['Total Rain (mm)'], marker='o', linestyle='-',
         label='Total Rain (mm)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Rain (mm)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(weather_yearly_df['Year'], weather_yearly_df['Total Snow (cm)'], marker='o', linestyle='-', color='red',
         label='Total Snow (cm)')
ax2.plot(weather_yearly_df['Year'], weather_yearly_df['Snow Grnd Last Day (cm)'], marker='o', linestyle='-',
         color='green', label='Snow Grnd Last Day (cm)')
ax2.set_ylabel('Snow (cm)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Mean Precipitation Over Time')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.show()

# Line graph for wind over time
plot_trends(weather_yearly_df, "Year", ["Spd of Max Gust (km/h)"], "Mean Wind Speed Over Time", "Year", "Wind Speed (km/h)")


# Initial visualizations for crop prices

# initial visualization for crops with $ per tonne
plot_trends(merged_df,"Year", price_per_tonnes, 'Mean Price Over Time', 'Year', '2015 $ per Tonne')

# Initial visualization for field peas
plot_trends(merged_df,"Year", price_peas_columns, "Mean Price of Peas Over Time", "Year", "2015 $ per bu")

# Initial visualization for lentils
plot_trends(merged_df,"Year",price_lentils_columns,"Mean Price of Lentils Over Time", "Year", "2015 $ per cwt")

# Initial visualization for mustard prices
plot_trends(merged_df, "Year", price_mustard_columns, "Mean Price of Mustard Over Time", "Year", "2015 $ per cwt")

# Initial visualization for canary seed
plot_trends(merged_df, "Year", ["Real Canada Canary Seed Price (2015 $ per cwt)"], "Mean Price of Canary Seeds Over Time", "Year", "2015 $ per cwt")



# initial visualization to see correlation of mean temp for crop prices

# Since the other columns are variable in the weather dataset other than temp I will be focusing on the temps
# to provide a more precise analysis I will use a linear regression model
# Fitting linear regression models using a function
def linear_regression_plot(df, x_col, y_col, title, label):
    plt.figure(figsize=(10, 6))
    for y in y_col:
        x_years = df[[x_col]]
        y_mean_temp = df[y]
        model_mean = LinearRegression().fit(x_years, y_mean_temp)
        trend_mean = model_mean.predict(x_years)

        plt.plot(x_years, y_mean_temp, marker='o', linestyle='-', label=y)
        plt.plot(df[x_col], trend_mean, linestyle='--')

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(label)
    plt.legend()
    plt.show()


linear_regression_plot(weather_yearly_df, 'Year', mean_temp_cols, "Mean Temp Over Time", "Mean Temp (°C)")
linear_regression_plot(weather_yearly_df,"Year", extreme_temp_cols,"Mean Extreme Temp Over Time", "Temp (°C)")

# Mean Temp
print(price_mustard_columns)