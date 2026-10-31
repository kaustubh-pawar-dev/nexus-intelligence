-- Every raw row of the four cleaned tables must be staged, dropped as a duplicate or quarantined.
SELECT t.table_name, t.raw_rows, t.staged_rows,
       coalesce(i.dropped, 0) AS dropped_duplicates, coalesce(i.quarantined, 0) AS quarantined,
       t.raw_rows - t.staged_rows - coalesce(i.dropped, 0) - coalesce(i.quarantined, 0) AS unaccounted
FROM (
    SELECT 'customers' AS table_name, (SELECT count(*) FROM raw.customers) AS raw_rows,
           (SELECT count(*) FROM staging.customers) AS staged_rows
    UNION ALL SELECT 'orders', (SELECT count(*) FROM raw.orders), (SELECT count(*) FROM staging.orders)
    UNION ALL SELECT 'order_items', (SELECT count(*) FROM raw.order_items), (SELECT count(*) FROM staging.order_items)
    UNION ALL SELECT 'payments', (SELECT count(*) FROM raw.payments), (SELECT count(*) FROM staging.payments)
) t
LEFT JOIN (
    SELECT table_name,
           count(DISTINCT source_row_id) FILTER (WHERE action = 'dropped') AS dropped,
           count(DISTINCT source_row_id) FILTER (WHERE action = 'quarantined') AS quarantined
    FROM staging.row_issues GROUP BY table_name
) i USING (table_name)
ORDER BY t.table_name