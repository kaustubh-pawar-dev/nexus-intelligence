-- Inventory turnover by year and category: cost of goods shipped divided by average daily inventory value
WITH daily AS (
    SELECT extract(year FROM i."date")::int AS yr,
           p.category,
           i."date" AS dt,
           sum(i.inventory_value) AS inv_value,
           sum(i.units_fulfilled * p.unit_cost) AS cogs
    FROM analytics.inventory_daily i
    JOIN analytics.products p ON p.product_id = i.product_id
    GROUP BY 1, 2, 3
)
SELECT yr AS year,
       category,
       round(sum(cogs), 0) AS cogs,
       round(avg(inv_value), 0) AS avg_inventory_value,
       round(sum(cogs) / avg(inv_value), 2) AS turnover,
       round(365 / (sum(cogs) / avg(inv_value)), 1) AS days_of_inventory
FROM daily
GROUP BY yr, category
ORDER BY yr, category