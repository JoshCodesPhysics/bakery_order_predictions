import pandas as pd
import sqlite3

print("\nPandas\n\n")

dataframe = pd.read_csv("sales_data.csv")
dataframe_filtered = dataframe[dataframe["date"] >= "2023-01-05"]

summary = (
    dataframe_filtered.groupby(["store_id", "sku_id"])["units_sold"].sum().reset_index()
)
print(summary)
print("\nSQLite\n\n")


database = sqlite3.connect("bakery_sales.db")

query = """
SELECT store_id, AVG(units_sold), MIN(date), sku_id
FROM sales_history
GROUP BY store_id, sku_id
"""
dataframe_sql = pd.read_sql(query, database)
"""
ORDER BY total_sales DESC
SELECT store_id, sku_id, SUM(units_sold) AS total_sales
WHERE date >= '2023-01-05'
"""
