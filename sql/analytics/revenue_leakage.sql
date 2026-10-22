-- Order funnel and revenue leakage by order year: cancellations, failed payments and refunds
WITH order_value AS (
    SELECT o.order_id,
           extract(year FROM o.order_date)::int AS yr,
           o.order_status,
           sum(i.line_total) AS total
    FROM analytics.orders o
    JOIN analytics.order_items i ON i.order_id = o.order_id
    GROUP BY o.order_id, o.order_date, o.order_status
), joined AS (
    SELECT v.yr, v.order_status, v.total, p.payment_status,
           coalesce(r.refund_amount, 0) AS refunded
    FROM order_value v
    LEFT JOIN analytics.payments p ON p.order_id = v.order_id
    LEFT JOIN analytics.refunds r ON r.order_id = v.order_id
), yearly AS (
    SELECT yr AS year,
           count(*) AS orders_placed,
           count(*) FILTER (WHERE order_status = 'Completed') AS completed,
           count(*) FILTER (WHERE payment_status = 'Captured') AS paid,
           count(*) FILTER (WHERE payment_status = 'Captured' AND refunded = 0) AS paid_not_refunded,
           sum(total) AS placed_value,
           coalesce(sum(total) FILTER (WHERE order_status = 'Cancelled'), 0) AS cancelled_value,
           coalesce(sum(total) FILTER (WHERE payment_status = 'Failed'), 0) AS failed_value,
           sum(refunded) AS refunded_value
    FROM joined
    GROUP BY yr
)
SELECT year, orders_placed, completed, paid, paid_not_refunded,
       round(placed_value, 2) AS placed_value,
       round(cancelled_value, 2) AS cancelled_value,
       round(failed_value, 2) AS failed_value,
       round(refunded_value, 2) AS refunded_value,
       round(100 * (cancelled_value + failed_value + refunded_value) / placed_value, 1) AS leakage_pct
FROM yearly
ORDER BY year