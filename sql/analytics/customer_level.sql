-- One row per customer: attributes plus observed orders, revenue and contribution (0 if never ordered)
WITH @LINE_FACTS@
SELECT c.customer_id,
       c.country,
       c.segment,
       c.acquisition_channel AS channel,
       c.marketing_opt_in,
       c.signup_date,
       coalesce(f.orders, 0) AS orders,
       coalesce(f.revenue, 0) AS revenue,
       coalesce(f.contribution, 0) AS contribution
FROM analytics.customers c
LEFT JOIN (
    SELECT customer_id,
           count(DISTINCT order_id) AS orders,
           sum(revenue) AS revenue,
           sum(revenue - cogs - shipping - refunded) AS contribution
    FROM line_facts
    GROUP BY customer_id
) f ON f.customer_id = c.customer_id
ORDER BY c.customer_id