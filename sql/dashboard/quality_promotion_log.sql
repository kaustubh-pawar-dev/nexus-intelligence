-- Rows quarantined while promoting staging into the analytics schema (knock-on effects).
SELECT table_name, rule, action, count(*) AS rows
FROM analytics.promotion_log
GROUP BY 1, 2, 3
ORDER BY 1