-- Monthly cohort retention: cohort = month of first completed order, active = completed order in a month
WITH active_months AS (
    SELECT customer_id, date_trunc('month', order_date)::date AS month
    FROM analytics.orders
    WHERE order_status = 'Completed'
    GROUP BY 1, 2
), first_month AS (
    SELECT customer_id, min(month) AS cohort_month
    FROM active_months
    GROUP BY 1
), activity AS (
    SELECT f.cohort_month, a.customer_id,
           ((extract(year FROM a.month) - extract(year FROM f.cohort_month)) * 12
            + extract(month FROM a.month) - extract(month FROM f.cohort_month))::int AS months_since
    FROM active_months a
    JOIN first_month f ON f.customer_id = a.customer_id
), sized AS (
    SELECT cohort_month, count(*) AS cohort_size
    FROM first_month
    GROUP BY 1
)
SELECT a.cohort_month,
       a.months_since,
       count(DISTINCT a.customer_id) AS active_customers,
       s.cohort_size,
       round(100.0 * count(DISTINCT a.customer_id) / s.cohort_size, 1) AS retention_pct
FROM activity a
JOIN sized s ON s.cohort_month = a.cohort_month
GROUP BY a.cohort_month, a.months_since, s.cohort_size
ORDER BY a.cohort_month, a.months_since