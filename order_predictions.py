import pandas as pd
import kagglehub
import os
import numpy
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def fetch_weather(lat, lon, date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Europe/London",
    }

    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if "daily" in data and len(data["daily"]["time"]) > 0:
            return {
                "date": data["daily"]["time"][0],
                "temp_max": data["daily"]["temperature_2m_max"][0],
                "temp_min": data["daily"]["temperature_2m_min"][0],
                "precipitation": data["daily"]["precipitation_sum"][0],
            }
    return {"date": date, "temp_max": None, "temp_min": None, "precipitation": None}


# Example usage
date_to_check = datetime.strptime("2025-01-01", "%Y-%m-%d").date()

if __name__ == "__main__":
    if not os.path.isfile("full_bakery_data_kaggle.csv"):
        # Download latest version
        path = kagglehub.dataset_download("akashdeepkuila/bakery")

        files = os.listdir(path)
        # print(files)
        # ['Bakery.csv', 'bakery_sales_revised.csv']
        df = pd.read_csv(os.path.join(path, "bakery_sales_revised.csv"))

        bakery_locations = [
            {
                "store_id": "store_1",
                "name": "Aran Bakery",
                "postcode": "PH8 0AR",
                "lat": 56.5650,
                "lon": -3.5870,
            },
            {
                "store_id": "store_2",
                "name": "Lannan Bakery",
                "postcode": "EH4 1JP",
                "lat": 55.9560,
                "lon": -3.2190,
            },
            {
                "store_id": "store_3",
                "name": "Bad Girl Bakery",
                "postcode": "IV6 7TP",
                "lat": 57.5180,
                "lon": -4.4570,
            },
            {
                "store_id": "store_4",
                "name": "Cottonrake Bakery",
                "postcode": "G12 8HL",
                "lat": 55.8750,
                "lon": -4.2930,
            },
            {
                "store_id": "store_5",
                "name": "The Bakery Inverness",
                "postcode": "IV3 5DT",
                "lat": 57.4770,
                "lon": -4.2290,
            },
            {
                "store_id": "store_6",
                "name": "Deanston Bakery",
                "postcode": "G41 3LP",
                "lat": 55.8300,
                "lon": -4.2800,
            },
            {
                "store_id": "store_7",
                "name": "Bostock Bakery",
                "postcode": "EH39 4HQ",
                "lat": 56.0580,
                "lon": -2.7190,
            },
            {
                "store_id": "store_8",
                "name": "Mimi’s Bakehouse",
                "postcode": "EH6 6RA",
                "lat": 55.9760,
                "lon": -3.1690,
            },
            {
                "store_id": "store_9",
                "name": "Bandit Bakery",
                "postcode": "AB10 1UB",
                "lat": 57.1480,
                "lon": -2.0980,
            },
            {
                "store_id": "store_10",
                "name": "Casella & Polegato",
                "postcode": "PH1 5JL",
                "lat": 56.3960,
                "lon": -3.4370,
            },
        ]

        store_ids = [
            f"store_{i}" for i in range(1, 11)
        ]  # or use the actual IDs from the Scottish bakeries list

        # List of all transaction numbers
        unique_tx = df["Transaction"].dropna().unique()
        # For all unique IDs, assign a store_id at random
        tx_store_map = pd.DataFrame(
            {
                "Transaction": unique_tx,
                "store_id": numpy.random.choice(store_ids, size=len(unique_tx)),
            }
        )

        # Left join means by Transcation, order of transactions is preserved
        df = df.merge(tx_store_map, on="Transaction", how="left")

        df_locations = pd.DataFrame(bakery_locations)

        df = df.merge(df_locations, on="store_id")
        df["date"] = pd.to_datetime(df["date_time"]).dt.date

        df["lat"] = df["lat"].round(3)
        df["lon"] = df["lon"].round(3)

        # List of unique combinations of lat, long and date for weather API queries
        query_keys = df[["lat", "lon", "date"]].drop_duplicates()

        weather_data = []
        # Obtain index and dataframe row (dictionary/series object, obtain information by column key)
        print("Reading weather data")
        for _, row in query_keys.iterrows():
            # Returns dictionary
            result = fetch_weather(row["lat"], row["lon"], str(row["date"]))
            result["lat"] = row["lat"]
            result["lon"] = row["lon"]
            # List of dictionaries of temp_max, temp_min, precipitation e.t.c
            print(_, " ", result, "\n")
            weather_data.append(result)

        weather_df = pd.DataFrame(weather_data)
        weather_df["lat"] = weather_df["lat"].round(3)
        weather_df["lon"] = weather_df["lon"].round(3)
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

        df = df.merge(weather_df, on=["lat", "lon", "date"], how="left")
        df.to_csv("full_bakery_data_kaggle.csv", index=False)

    else:
        df = pd.read_csv("full_bakery_data_kaggle.csv")

    sales_per_day = df.groupby(["date"]).size().reset_index(name="units_sold_per_day")
    sales_per_day_item = (
        df.groupby(["date", "Item"]).size().reset_index(name="units_sold_per_day_item")
    )
    store_sales_per_day = (
        df.groupby(["store_id", "date"])
        .size()
        .reset_index(name="units_sold_per_store_day")
    )
    store_sales_per_item = (
        df.groupby(["store_id", "date", "Item"])
        .size()
        .reset_index(name="units_sold_per_store_day_item")
    )

    df = df.merge(sales_per_day, on=["date"], how="left")
    df = df.merge(sales_per_day_item, on=["date", "Item"], how="left")
    df = df.merge(store_sales_per_day, on=["store_id", "date"], how="left")
    df = df.merge(store_sales_per_item, on=["store_id", "date", "Item"], how="left")

    df["day_of_week"] = df["date"].apply(lambda x: pd.Timestamp(x).dayofweek)

    public_holiday_url = "https://www.gov.uk/bank-holidays.json"
    response = requests.get(public_holiday_url)
    if response.status_code == 200:
        print("Public holiday data fetched successfully.")
    else:
        print("Failed to fetch public holiday data.")
        exit

    data = response.json()
    holidays = data["scotland"]["events"]
    holiday_dates = [
        datetime.strptime(event["date"], "%Y-%m-%d").date() for event in holidays
    ]
    df["date"] = pd.to_datetime(df["date"])
    # Find the earliest and latest dates
    earliest_date = df["date"].min()
    latest_date = df["date"].max()

    # Print the results
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    holiday_day_months = [(int(date.month), int(date.day)) for date in holiday_dates]
    df["is_public_holiday"] = df[["day", "month"]].apply(
        lambda row: (row["day"], row["month"]) in holiday_day_months, axis=1
    )
    df.drop(["day", "month"], axis=1, inplace=True)

    df["lag_1"] = df.groupby(["store_id", "Item"])[
        "units_sold_per_store_day_item"
    ].shift(1)
    df["target"] = df["units_sold_per_store_day_item"]
    # df = df.dropna()
    df["rolling_week_avg"] = df.groupby(["store_id", "Item"])[
        "units_sold_per_store_day_item"
    ].transform(lambda x: x.shift(1).rolling(7).mean())
    df = pd.get_dummies(
        df,
        columns=["Item", "store_id", "period_day", "weekday_weekend"],
        drop_first=True,
    )
    # Check for overlapping dates
    # overlapping_dates = set(df["day_month"]).intersection(set(holiday_day_months))
    # print("Overlapping dates:", overlapping_dates)

    # print("Type of df column:", type(df["day"]))
    # print("Type of df column:", type(df["month"]))
    # print("Type of holiday_day_months:", type(holiday_day_months))
    # print(holiday_day_months[:5], "\n\n", df["day_month"].unique()[:5])

    # test_date = list(overlapping_dates)[0]  # Pick the first overlapping date

    # Drop the temporary "day_month" column if not needed
    # df.drop(columns=["day_month"], inplace=True)

    # sns.boxplot(x="weekday_weekend", y="units_sold_per_store_day_item", data = df)
    # plt.show()
    # coffee_data = df[(df["Item"] == "Coffee")]
    # coffee_sales = coffee_data.groupby("date")["units_sold_per_store_day_item"].sum()
    # coffee_outliers = coffee_sales[coffee_sales > coffee_sales.quantile(0.99)]
    # print(coffee_outliers)

    # coffee_data.groupby("is_public_holiday")["units_sold_per_store_day_item"].mean().plot(kind="bar")
    # plt.xlabel("Date")
    # plt.ylabel("Units Sold Per Store Per Day (Coffee)")
    # plt.title("Daily Coffee Sales for Store 1")
    # plt.show()

    # df.drop(["Transaction", "date_time", "lat", "lon"], axis=1, inplace=True)

    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

    # Sort by date to preserve order
    df = df.sort_values("date")

    # Choose a cutoff
    df.drop
    cutoff = df["date"].max() - pd.Timedelta(days=14)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    features = [
        "temp_max",
        "temp_min",
        "precipitation",
        "is_public_holiday",
        # , "weekday_weekend", "period_day",
        # "units_sold_per_day_item", "units_sold_per_day",
        # "rolling_week_avg", "lag_1", "store_id",  "day_of_week"
    ]

    grouping_keys = ["store_id", "Item", "date"]

    # Filter the features and group by the same keys
    input_train = train[features]
    input_test = test[features]

    # Align the target column
    output_train = train["target"]
    output_test = test["target"]
    # Drop the grouping keys to keep only the features and target

    # print("Input shape:", input_train.shape)
    # print("Output shape:", output_train.shape)

    model = lgb.LGBMRegressor()
    model.fit(input_train, output_train)

    prediction = model.predict(input_test)

    rmse = mean_squared_error(output_test, prediction)
    mae = mean_absolute_error(output_test, prediction)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

"""
sales_dataframe = pd.read_csv("mock_bakery_sales.csv")
meta_dataframe = pd.read_csv("mock_store_metadata.csv")

merged_dataframe = sales_dataframe.merge(meta_dataframe, on = "store_id")
filtered_merged_dataframe = merged_dataframe[(merged_dataframe["store_id"] == "store_1") & (merged_dataframe["sku_id"] == "croissant")].copy()
filtered_merged_dataframe["date"] = pd.to_datetime(filtered_merged_dataframe["date"])
filtered_merged_dataframe.sort_values("date", inplace=True)
# Average last 7 days of units sold every day
filtered_merged_dataframe["rolling_average"] = filtered_merged_dataframe["units_sold"].rolling(7).mean()
# Sum units sold by region
summary = merged_dataframe.groupby(["region"])["units_sold"].sum().reset_index()
# Look at rolling averages grouped by store_id and date after a certain time
print(filtered_merged_dataframe[["store_id", "date", "rolling_average"]][filtered_merged_dataframe['date'] > '2023-01-07'])
print(summary)
"""
