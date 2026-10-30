-- First and last order date of the completed orders in the cleaned layer.
SELECT min(order_date) AS first_order, max(order_date) AS last_order
FROM analytics.orders
WHERE order_status = 'Completed'