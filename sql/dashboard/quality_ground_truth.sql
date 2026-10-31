-- Evaluation only: staged rows compared with the ground-truth (core) rows. The pipeline never reads core.
SELECT 'orders: staged rows missing from ground truth' AS check_name, count(*) AS rows
    FROM staging.orders s LEFT JOIN core.orders c USING (order_id) WHERE c.order_id IS NULL
UNION ALL SELECT 'orders: any column differs', count(*)
    FROM staging.orders s JOIN core.orders c USING (order_id)
    WHERE s.customer_id <> c.customer_id OR s.order_date <> c.order_date OR s.order_status <> c.order_status
       OR s.discount_pct <> c.discount_pct OR s.sales_channel <> c.sales_channel
UNION ALL SELECT 'customers: country, region or segment differs', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
    WHERE s.country <> c.country OR s.region <> c.region OR s.segment <> c.segment
UNION ALL SELECT 'customers: email present but different', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
    WHERE s.email IS NOT NULL AND s.email <> c.email
UNION ALL SELECT 'customers: signup_date differs (imputed dates)', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id) WHERE s.signup_date <> c.signup_date
UNION ALL SELECT 'customers: largest signup_date error in days', coalesce(max(abs(s.signup_date - c.signup_date)), 0)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
UNION ALL SELECT 'order_items: any column differs', count(*)
    FROM staging.order_items s JOIN core.order_items c USING (order_id, line_number)
    WHERE s.product_id <> c.product_id OR s.quantity <> c.quantity OR s.unit_price <> c.unit_price
       OR s.unit_cost <> c.unit_cost OR s.discount_pct <> c.discount_pct OR s.line_total <> c.line_total
UNION ALL SELECT 'payments: any column differs', count(*)
    FROM staging.payments s JOIN core.payments c USING (payment_id)
    WHERE s.order_id <> c.order_id OR s.payment_date <> c.payment_date OR s.payment_method <> c.payment_method
       OR s.payment_status <> c.payment_status OR s.amount <> c.amount