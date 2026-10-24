-- Shared CTE: one row per completed order line, with order-level amounts allocated by line value
line_facts AS (
    SELECT o.order_id,
           o.customer_id,
           o.order_date,
           i.product_id,
           p.product_name,
           p.category,
           c.segment,
           i.quantity,
           i.line_total AS revenue,
           i.quantity * i.unit_cost AS cogs,
           coalesce(s.shipping_cost, 0) * i.line_total / t.order_total AS shipping,
           coalesce(r.refund_amount, 0) * i.line_total / t.order_total AS refunded,
           CASE WHEN pay.payment_status = 'Failed' THEN pay.amount ELSE 0 END
               * i.line_total / t.order_total AS failed
    FROM analytics.order_items i
    JOIN analytics.orders o ON o.order_id = i.order_id
    JOIN (SELECT order_id, sum(line_total) AS order_total
          FROM analytics.order_items GROUP BY order_id) t ON t.order_id = i.order_id
    JOIN analytics.products p ON p.product_id = i.product_id
    JOIN analytics.customers c ON c.customer_id = o.customer_id
    LEFT JOIN analytics.shipments s ON s.order_id = o.order_id
    LEFT JOIN analytics.payments pay ON pay.order_id = o.order_id
    LEFT JOIN analytics.refunds r ON r.order_id = o.order_id
    WHERE o.order_status = 'Completed'
)
