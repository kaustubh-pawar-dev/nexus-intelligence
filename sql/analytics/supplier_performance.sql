-- Supplier scorecard from delivered purchase orders, ranked by on-time share
WITH delivered AS (
    SELECT po.supplier_id,
           (po.actual_delivery_date - po.order_date) AS actual_lead,
           (po.expected_delivery_date - po.order_date) AS planned_lead,
           (po.actual_delivery_date - po.expected_delivery_date) AS delay_days,
           po.quantity_ordered,
           po.quantity_received,
           po.defective_units
    FROM analytics.purchase_orders po
    WHERE po.delivery_status = 'Delivered'
), per_supplier AS (
    SELECT supplier_id,
           count(*) AS deliveries,
           round(100.0 * avg((delay_days <= 0)::int), 1) AS on_time_pct,
           round(avg(delay_days) FILTER (WHERE delay_days > 0), 2) AS avg_delay_when_late,
           round(avg(planned_lead), 2) AS avg_planned_lead,
           round(avg(actual_lead), 2) AS avg_actual_lead,
           round(100.0 * sum(defective_units) / sum(quantity_received), 2) AS defect_pct,
           round(100.0 * avg((quantity_received < quantity_ordered)::int), 1) AS short_shipment_pct
    FROM delivered
    GROUP BY supplier_id
)
SELECT p.supplier_id,
       s.supplier_name,
       s.region,
       p.deliveries,
       p.on_time_pct,
       round(100 * s.on_time_delivery_rate, 1) AS stated_on_time_pct,
       p.avg_delay_when_late,
       p.avg_planned_lead,
       p.avg_actual_lead,
       p.defect_pct,
       round(100 * s.defect_rate, 2) AS stated_defect_pct,
       p.short_shipment_pct,
       rank() OVER (ORDER BY p.on_time_pct DESC) AS on_time_rank
FROM per_supplier p
JOIN analytics.suppliers s ON s.supplier_id = p.supplier_id
ORDER BY on_time_rank, p.supplier_id