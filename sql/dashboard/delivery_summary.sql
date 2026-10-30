-- Delivery performance of completed orders placed from :start to :end. Late means delivered after the promised date.
SELECT count(*) AS delivered,
       coalesce(sum((s.delivered_date > s.promised_delivery_date)::int), 0) AS late
FROM analytics.shipments s
JOIN analytics.orders o ON o.order_id = s.order_id
WHERE o.order_status = 'Completed'
  AND s.shipment_status = 'Delivered'
  AND o.order_date BETWEEN :start AND :end