-- Executive KPIs for completed orders placed from :start to :end (inclusive). Uses the same line_facts CTE as the analytics queries.
WITH @LINE_FACTS@
SELECT count(DISTINCT order_id) AS orders,
       count(DISTINCT customer_id) AS customers,
       coalesce(sum(revenue), 0) AS revenue,
       coalesce(sum(cogs), 0) AS cogs,
       coalesce(sum(shipping), 0) AS shipping,
       coalesce(sum(refunded), 0) AS refunds
FROM line_facts
WHERE order_date BETWEEN :start AND :end