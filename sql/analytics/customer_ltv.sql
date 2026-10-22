-- Observed lifetime value (contribution margin to date) per customer, summarised by segment and by channel
WITH @LINE_FACTS@,
per_customer AS (
    SELECT lf.customer_id,
           lf.segment,
           c.acquisition_channel AS channel,
           count(DISTINCT lf.order_id) AS orders,
           sum(lf.revenue) AS revenue,
           sum(lf.revenue - lf.cogs - lf.shipping - lf.refunded) AS contribution
    FROM line_facts lf
    JOIN analytics.customers c ON c.customer_id = lf.customer_id
    GROUP BY 1, 2, 3
)
SELECT 'segment' AS dimension,
       segment AS value,
       count(*) AS customers,
       round(avg(orders), 2) AS avg_orders,
       round(avg(revenue), 2) AS avg_revenue,
       round(avg(contribution), 2) AS avg_contribution_ltv,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY contribution))::numeric, 2) AS median_contribution_ltv
FROM per_customer
GROUP BY segment
UNION ALL
SELECT 'channel',
       channel,
       count(*),
       round(avg(orders), 2),
       round(avg(revenue), 2),
       round(avg(contribution), 2),
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY contribution))::numeric, 2)
FROM per_customer
GROUP BY channel
ORDER BY 1, 2