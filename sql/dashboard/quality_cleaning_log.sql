-- Row-level cleaning actions of the staging step, grouped by table, action and rule.
SELECT table_name, action, rule, rows
FROM staging.cleaning_summary
ORDER BY table_name, action, rule