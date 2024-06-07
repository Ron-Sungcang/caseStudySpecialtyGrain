# Preparing the environment
import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Constants
DATABASE_PATH = "grains.db"
GRAINS_FILE = 'merged_specialty_grain_revenue.xlsx'
WEATHER_FILE = 'en_climate_monthly_SK_estevan.csv'
CPI_FILE = 'cpi.csv'


def load_datasets(db_path, grain_file, weather_file, cpi_file):
    """
    Loads files into a sql database using sqlite
    :param db_path: A string that represents the path to connect to the database
    :param grain_file: A string that represents the path to the merged_specialty_grain_revenue.xlsx file
    :param weather_file: A string that represents the path to the en_climate_monthly_SK_estevan.csv file
    :param cpi_file: A string that represents the path to the cpi.csv file
    :return: Three dataframes = grains_df, cpi_df, weather_df
    """
    # connect to database
    con = sql.connect(db_path)

    # create a cursor
    cur = con.cursor()
    # Load spreadsheets into sql
    pd.read_excel(grain_file).to_sql("specialty_grains", con, if_exists='append',
                                     index=False)
    pd.read_csv(weather_file).to_sql("historical_weather", con, if_exists='append', index=False)
    pd.read_csv(cpi_file).to_sql("cpi", con, if_exists='append', index=False)

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
    return grains_df, cpi_df, weather_df


def impute_monthly_average(df, target_col, group_averages):
    """
    Takes a dataframe a target column, and a dataframe of averages per month to impute missing values with seasonal average
    :param df: dataframe to be imputed
    :param target_col: A string that represents the column to check
    :param group_averages: monthly average
    :return: None
    """

    # Internal function that applies to each row in df
    def impute_row(row):
        if pd.isnull(row[target_col]):
            return group_averages[row['Month']]
        else:
            return row[target_col]

    # Apply the function to impute missing values
    df[target_col] = df.apply(impute_row, axis=1)


def preprocess_grains(df, cols, rows, grain_cols):
    """
    Preprocesses the grains dataframe by dropping selected columns and dropping rows where all values except date are null.
    Then groups the dataframe by years
    :param df: The grains dataframe
    :param cols: A list of columns to remove
    :param rows: A list of rows to be removed
    :param grain_cols: A list of the name of the columns from the grain dataset
    :return: a data frame grouped by years with yearly averages, and the years in the new dataframe
    """
    grains_df = df.drop(columns=cols)
    # Dropping rows that are all null
    for row in rows:
        grains_df = grains_df.drop(row)
    grain_years = pd.to_datetime(grains_df['Date']).dt.year.rename('Year')
    grains_yearly_df = grains_df.groupby(grain_years)[grain_cols].mean().reset_index()

    return grains_yearly_df, grain_years


def preprocess_cpi(df):
    """
    Groups the cpi dataframe into years
    :param df: the cpi dataframe
    :return: A new dataframe grouped years with yearly averages
    """
    cpi_years = pd.to_datetime(df['REF_DATE']).dt.year.rename('Year')
    cpi_yearly_values_df = df.groupby(cpi_years)["VALUE"].mean().reset_index()

    return cpi_yearly_values_df


def merge_dfs(grains_df, cpi_df, base_year, real_columns, grain_cols):
    """
    Calculates the real prices of the yearly grains dataframe by taking the cpi of the base year.
    Merges the two dataframes then inserts the real price
    :param grains_df: The grains dataframe
    :param cpi_df: The cpi dataframe
    :param base_year: the year chosen to calculate real prices
    :param real_columns: a list of strings containing the new column names for the real price
    :param grain_cols: A list of strings containing the column names for the grain prices
    :return: a merged dataframe
    """
    base_cpi = cpi_df[cpi_df['Year'] == base_year]['VALUE'].values[0]
    merged_df = pd.merge(grains_df, cpi_df, on='Year')

    # Inserting real price columns for each grain
    for real_col, grain_col in zip(real_columns, grain_cols):
        merged_df[real_col] = merged_df[grain_col] * (base_cpi / merged_df['VALUE'])

    return merged_df


def plot_trends(df, x_col, y_cols, title, x_label, y_label):
    """
    Plot trends over time for multiple columns.
    :param df: dataframe to be used for plotting
    :param x_col: A string that represents column name to use for the x values
    :param y_cols: A list of strings that represents column names to use for the y values
    :param title: The title of the visualization
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :return: none
    """
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


def linear_regression_plot(df, x_col, y_col, title, label):
    """
    Uses linear regression to predict the trend line, then plots trends for multiple columns
    :param df: dataframe to be used for plotting
    :param x_col: A string that represents the column name to be used for the x values
    :param y_col: A string that represents the column name to be used for the y values
    :param title: A string to be used for the title
    :param label: The string label for the y axis
    """
    plt.figure(figsize=(10, 6))
    for y in y_col:
        x_years = df[[x_col]]
        y_mean_temp = df[y]
        # fit the values into the model
        model_mean = LinearRegression().fit(x_years, y_mean_temp)
        # Predicting the trend line
        trend_mean = model_mean.predict(x_years)

        plt.plot(x_years, y_mean_temp, marker='o', linestyle='-', label=y)
        plt.plot(df[x_col], trend_mean, linestyle='--')

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(label)
    plt.legend()
    plt.show()


# Plotting the scatter plot
def scatter_plot(df, x_col, y_col, title):
    plt.figure(figsize=(10, 6))

    x_cols = df[[x_col]]
    y_mean_grain = df[y_col]
    model_mean = LinearRegression().fit(x_cols, y_mean_grain)
    trend_mean = model_mean.predict(x_cols)

    plt.scatter(df[x_col], df[y_col], color="blue", label="Data Points")
    plt.plot(df[x_col], trend_mean, color="red", linestyle='--', label="Trend Line")

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()

    # Calculate and print correlation
    correlation = df[x_col].corr(df[y_col])
    print(f"Correlation between {x_col} and {y_col}: {correlation}")


def main():
    # List of columns
    grain_columns = ['Oats 2CW ($ per tonne)', 'CW Feed ($ per tonne)', 'Flax 1CAN ($ per tonne)',
                     'Canola 1CAN ($ per tonne)',
                     'Chickpeas 1CW - Kabuli 9mm ($ per cwt)', 'Chickpeas 1CW - Desi ($ per cwt)',
                     'Field Peas 1CAN - Yellow ($ per bu)',
                     'Field Peas 1CAN - Green ($ per bu)', 'Field Peas 1CAN - Feed ($ per bu)',
                     'Canada Canary Seed ($ per cwt)',
                     'Lentils Small Red ($ per cwt)', 'Lentils Large Green ($ per cwt)',
                     'Lentils Small Green ($ per cwt)',
                     'Lentils Medium Green ($ per cwt)',
                     'Lentils French Green ($ per cwt)', 'Mustard 1CAN - Yellow ($ per cwt)',
                     'Mustard 1CAN - Brown ($ per cwt)', 'Mustard 1CAN - Oriental ($ per cwt)']

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

    # Group by years for analysis
    weather_columns = ["Mean Max Temp (°C)", "Mean Min Temp (°C)", "Mean Temp (°C)", "Extr Max Temp (°C)",
                       "Extr Min Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Snow Grnd Last Day (cm)",
                       "Spd of Max Gust (km/h)"]

    # Load datasets
    grains_df, cpi_df, weather_df = load_datasets(DATABASE_PATH, GRAINS_FILE, WEATHER_FILE, CPI_FILE)

    # Processing the grains data set
    # Checking for null values in grains dataset
    grains_missing_values_count = grains_df.isnull().sum()
    print(grains_missing_values_count[:])
    grains_missing_percentage = (grains_missing_values_count[grain_columns] / len(grains_df)) * 100
    print(grains_missing_percentage[:])

    # Removing columns from list
    grain_columns.remove('Chickpeas 1CW - Kabuli 9mm ($ per cwt)')
    grain_columns.remove('Chickpeas 1CW - Desi ($ per cwt)')
    grain_columns.remove('Lentils Small Red ($ per cwt)')

    # checking for rows where they are all null
    cols_to_check = grains_df.columns.drop('Date')
    condition = grains_df[cols_to_check].isnull().all(axis=1)
    rows_null = grains_df[condition]
    print("ROWS NULL \n", rows_null)

    # Dropping the columns with 26% missing data, Dropping rows that are all null, and Change grains to yearly
    grains_yearly_df, grain_years = preprocess_grains(grains_df,
                                                      ['Chickpeas 1CW - Kabuli 9mm ($ per cwt)',
                                                       'Chickpeas 1CW - Desi ($ per cwt)',
                                                       'Lentils Small Red ($ per cwt)'], [416, 521], grain_columns)
    print(grains_yearly_df.head())

    # Get cpi values per year
    cpi_yearly_values_df = preprocess_cpi(cpi_df)
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

    # Merging the yearly dfs
    merged_df = merge_dfs(grains_yearly_df, cpi_yearly_values_df, 2015, real_grain_columns, grain_columns)

    print(merged_df)

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

    # Use the generalized function to impute missing values in 'Spd of Max Gust (km/h)' based on 'Month'
    impute_monthly_average(weather_df, 'Spd of Max Gust (km/h)', monthly_average)

    # Checking for null values in the dataset
    weather_missing_values_count = weather_df.isnull().sum()
    print(weather_missing_values_count[:])

    print(weather_df.iloc[171])

    # Checking how many months in a year
    months_per_year = weather_df.groupby('Year')['Month'].nunique().reset_index(name='UniqueMonths')
    print(months_per_year)

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

    # Plotting the the temperatures
    plot_trends(weather_yearly_df, "Year", mean_temp_cols, "Mean Temp Over Time", "Year", "Temp in Celsius")
    plot_trends(weather_yearly_df, "Year", extreme_temp_cols, "Mean Extreme Temp Over Time", "Year", "Temp in Celsius")

    # Line graphs for precipitation, uses
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
    plot_trends(weather_yearly_df, "Year", ["Spd of Max Gust (km/h)"], "Mean Wind Speed Over Time", "Year",
                "Wind Speed (km/h)")

    # Initial visualizations for crop prices

    # initial visualization for crops with $ per tonne
    plot_trends(merged_df, "Year", price_per_tonnes, 'Mean Price Over Time', 'Year', '2015 $ per Tonne')

    # Initial visualization for field peas
    plot_trends(merged_df, "Year", price_peas_columns, "Mean Price of Peas Over Time", "Year", "2015 $ per bu")

    # Initial visualization for lentils
    plot_trends(merged_df, "Year", price_lentils_columns, "Mean Price of Lentils Over Time", "Year", "2015 $ per cwt")

    # Initial visualization for mustard prices
    plot_trends(merged_df, "Year", price_mustard_columns, "Mean Price of Mustard Over Time", "Year", "2015 $ per cwt")

    # Initial visualization for canary seed
    plot_trends(merged_df, "Year", ["Real Canada Canary Seed Price (2015 $ per cwt)"],
                "Mean Price of Canary Seeds Over Time", "Year", "2015 $ per cwt")

    # initial visualization to see correlation of mean temp for crop prices

    # Since the other columns are variable in the weather dataset other than temp I will be focusing on the temps
    # to provide a more precise analysis I will use a linear regression model

    # Plotting linear regression for temperatures
    linear_regression_plot(weather_yearly_df, 'Year', mean_temp_cols, "Mean Temp Over Time", "Mean Temp (°C)")
    linear_regression_plot(weather_yearly_df, "Year", extreme_temp_cols, "Mean Extreme Temp Over Time", "Temp (°C)")

    merged_grain_and_weather = pd.merge(weather_yearly_df, merged_df, on="Year")

    # Plotting the correlation of temperatures and grain prices
    for weather_col in mean_temp_cols:
        for grain_cols in real_grain_columns:
            scatter_plot(merged_grain_and_weather, weather_col, grain_cols,
                         f"{weather_col} Vs {grain_cols}")


main()
