-- Contribution margin by product category (revenue - cost of goods sold - shipping - refunds)
WITH @LINE_FACTS@
SELECT category,
       round(sum(revenue), 2) AS revenue,
       round(sum(cogs), 2) AS cogs,
       round(sum(shipping), 2) AS shipping,
       round(sum(refunded), 2) AS refunds,
       round(sum(revenue - cogs - shipping - refunded), 2) AS contribution,
       round(100 * sum(revenue - cogs - shipping - refunded) / sum(revenue), 1) AS contribution_pct,
       round(sum(failed), 2) AS uncollected_failed_payments
FROM line_facts
GROUP BY category
ORDER BY contribution DESC