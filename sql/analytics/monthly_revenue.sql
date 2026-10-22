-- Monthly revenue with month-over-month, year-over-year and a 3-month moving average
WITH monthly AS (
    SELECT date_trunc('month', o.order_date)::date AS month,
           count(DISTINCT o.order_id) AS orders,
           sum(i.line_total) AS revenue
    FROM analytics.order_items i
    JOIN analytics.orders o ON o.order_id = i.order_id
    WHERE o.order_status = 'Completed'
    GROUP BY 1
)
SELECT month,
       orders,
       round(revenue, 2) AS revenue,
       round(100 * (revenue / lag(revenue, 1) OVER (ORDER BY month) - 1), 1) AS mom_pct,
       round(100 * (revenue / lag(revenue, 12) OVER (ORDER BY month) - 1), 1) AS yoy_pct,
       round(avg(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m,
       count(*) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS window_months
FROM monthly
ORDER BY month