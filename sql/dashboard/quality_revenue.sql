-- Revenue of completed orders in staging, in the ground truth and inside quarantined orders.
SELECT
  round((SELECT sum(i.line_total) FROM staging.order_items i JOIN staging.orders o USING (order_id)
         WHERE o.order_status = 'Completed'), 2) AS staged_revenue,
  round((SELECT sum(i.line_total) FROM core.order_items i JOIN core.orders o USING (order_id)
         WHERE o.order_status = 'Completed'), 2) AS ground_truth_revenue,
  round((SELECT coalesce(sum(i.line_total), 0) FROM core.order_items i JOIN core.orders o USING (order_id)
         WHERE o.order_status = 'Completed'
           AND o.order_id IN (SELECT row_key FROM staging.row_issues
                              WHERE table_name = 'orders' AND action = 'quarantined')), 2)
         AS revenue_in_quarantined_orders