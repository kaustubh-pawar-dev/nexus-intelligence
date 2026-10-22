-- RFM segmentation as of 2025-12-31: recency, frequency and monetary quartiles with NTILE
WITH per_customer AS (
    SELECT o.customer_id,
           max(o.order_date) AS last_order,
           count(DISTINCT o.order_id) AS frequency,
           sum(i.line_total) AS monetary
    FROM analytics.orders o
    JOIN analytics.order_items i ON i.order_id = o.order_id
    WHERE o.order_status = 'Completed'
    GROUP BY 1
), scored AS (
    SELECT customer_id,
           (DATE '2025-12-31' - last_order) AS recency_days,
           frequency,
           monetary,
           ntile(4) OVER (ORDER BY (DATE '2025-12-31' - last_order) DESC, customer_id) AS r_score,
           ntile(4) OVER (ORDER BY frequency, customer_id) AS f_score,
           ntile(4) OVER (ORDER BY monetary, customer_id) AS m_score
    FROM per_customer
), labelled AS (
    SELECT s.*,
           CASE WHEN f_score >= 3 AND r_score >= 3 THEN 'Champions'
                WHEN f_score >= 3 THEN 'At risk'
                WHEN r_score >= 3 THEN 'New / promising'
                ELSE 'Dormant' END AS rfm_segment
    FROM scored s
)
SELECT rfm_segment,
       count(*) AS customers,
       round(avg(recency_days), 1) AS avg_recency_days,
       round(avg(frequency), 2) AS avg_orders,
       round(avg(monetary), 2) AS avg_revenue,
       round(sum(monetary), 2) AS revenue,
       round(100 * sum(monetary) / sum(sum(monetary)) OVER (), 1) AS revenue_share_pct
FROM labelled
GROUP BY rfm_segment
ORDER BY revenue DESC