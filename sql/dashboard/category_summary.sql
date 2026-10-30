-- Revenue and contribution margin by product category for completed orders placed from :start to :end.
WITH @LINE_FACTS@
SELECT category,
       round(sum(revenue), 2) AS revenue,
       round(sum(revenue - cogs - shipping - refunded), 2) AS contribution
FROM line_facts
WHERE order_date BETWEEN :start AND :end
GROUP BY category
ORDER BY revenue DESC