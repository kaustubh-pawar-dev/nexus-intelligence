-- Top 3 products in each category by contribution margin, with RANK and share of the category
WITH @LINE_FACTS@,
product_totals AS (
    SELECT category, product_id, product_name,
           sum(revenue) AS revenue,
           sum(revenue - cogs - shipping - refunded) AS contribution
    FROM line_facts
    GROUP BY category, product_id, product_name
),
ranked AS (
    SELECT category, product_id, product_name, revenue, contribution,
           rank() OVER (PARTITION BY category ORDER BY contribution DESC) AS rank_in_category,
           round(100 * contribution / sum(contribution) OVER (PARTITION BY category), 1) AS share_of_category
    FROM product_totals
)
SELECT category, rank_in_category, product_id, product_name,
       round(revenue, 2) AS revenue,
       round(contribution, 2) AS contribution,
       share_of_category
FROM ranked
WHERE rank_in_category <= 3
ORDER BY category, rank_in_category, product_id