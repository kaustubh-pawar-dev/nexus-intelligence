-- Revenue, COGS, allocated shipping and allocated refunds by product category for completed orders placed from :start to :end.
WITH @LINE_FACTS@
SELECT category,
       sum(revenue) AS revenue,
       sum(cogs) AS cogs,
       sum(shipping) AS shipping,
       sum(refunded) AS refunds
FROM line_facts
WHERE order_date BETWEEN :start AND :end
GROUP BY category
ORDER BY category