-- NEXUS Intelligence: core schema, the cleaned relational model. All data is SYNTHETIC.
-- Applied by scripts/init_database.py. Safe to run more than once.

CREATE SCHEMA IF NOT EXISTS core;

CREATE TABLE IF NOT EXISTS core.suppliers (
    supplier_id text PRIMARY KEY,
    supplier_name text NOT NULL UNIQUE,
    country text NOT NULL,
    region text NOT NULL,
    lead_time_mean_days integer NOT NULL CHECK (lead_time_mean_days >= 1),
    on_time_delivery_rate numeric(6,4) NOT NULL CHECK (on_time_delivery_rate BETWEEN 0 AND 1),
    defect_rate numeric(6,4) NOT NULL CHECK (defect_rate BETWEEN 0 AND 1),
    payment_terms_days integer NOT NULL CHECK (payment_terms_days > 0),
    onboarded_date date NOT NULL
);

CREATE TABLE IF NOT EXISTS core.products (
    product_id text PRIMARY KEY,
    sku text NOT NULL UNIQUE,
    product_name text NOT NULL,
    category text NOT NULL,
    supplier_id text NOT NULL REFERENCES core.suppliers (supplier_id),
    unit_cost numeric(12,2) NOT NULL CHECK (unit_cost > 0),
    list_price numeric(12,2) NOT NULL,
    launch_date date NOT NULL,
    is_active boolean NOT NULL,
    CONSTRAINT products_price_above_cost CHECK (list_price > unit_cost)
);

CREATE TABLE IF NOT EXISTS core.customers (
    customer_id text PRIMARY KEY,
    email text UNIQUE,
    country text NOT NULL,
    region text NOT NULL,
    segment text NOT NULL CHECK (segment IN ('Consumer', 'Small Business', 'Enterprise')),
    acquisition_channel text NOT NULL CHECK (acquisition_channel IN ('Organic Search', 'Paid Search', 'Social Media', 'Email', 'Referral', 'Direct')),
    signup_date date NOT NULL CHECK (signup_date BETWEEN '2021-01-01' AND '2025-12-31'),
    marketing_opt_in boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS core.orders (
    order_id text PRIMARY KEY,
    customer_id text NOT NULL REFERENCES core.customers (customer_id),
    order_date date NOT NULL CHECK (order_date BETWEEN '2023-01-01' AND '2025-12-31'),
    sales_channel text NOT NULL CHECK (sales_channel IN ('Web', 'Mobile App', 'Marketplace')),
    order_status text NOT NULL CHECK (order_status IN ('Completed', 'Cancelled')),
    discount_pct numeric(4,2) NOT NULL CHECK (discount_pct BETWEEN 0 AND 0.20)
);

CREATE TABLE IF NOT EXISTS core.order_items (
    order_id text NOT NULL REFERENCES core.orders (order_id),
    line_number integer NOT NULL CHECK (line_number >= 1),
    product_id text NOT NULL REFERENCES core.products (product_id),
    quantity integer NOT NULL CHECK (quantity BETWEEN 1 AND 20),
    unit_price numeric(12,2) NOT NULL CHECK (unit_price > 0),
    unit_cost numeric(12,2) NOT NULL CHECK (unit_cost > 0),
    discount_pct numeric(4,2) NOT NULL CHECK (discount_pct BETWEEN 0 AND 0.20),
    line_total numeric(14,2) NOT NULL CHECK (line_total > 0),
    PRIMARY KEY (order_id, line_number),
    CONSTRAINT order_items_unique_product UNIQUE (order_id, product_id)
);

CREATE TABLE IF NOT EXISTS core.payments (
    payment_id text PRIMARY KEY,
    order_id text NOT NULL UNIQUE REFERENCES core.orders (order_id),
    payment_date date NOT NULL,
    payment_method text NOT NULL CHECK (payment_method IN ('Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Digital Wallet', 'Bank Transfer')),
    payment_status text NOT NULL CHECK (payment_status IN ('Captured', 'Failed')),
    amount numeric(14,2) NOT NULL CHECK (amount > 0)
);

CREATE TABLE IF NOT EXISTS core.refunds (
    refund_id text PRIMARY KEY,
    payment_id text NOT NULL UNIQUE REFERENCES core.payments (payment_id),
    order_id text NOT NULL REFERENCES core.orders (order_id),
    refund_date date NOT NULL,
    refund_type text NOT NULL CHECK (refund_type IN ('Full', 'Partial')),
    refund_reason text NOT NULL CHECK (refund_reason IN ('Defective', 'Not As Described', 'Changed Mind', 'Late Delivery', 'Wrong Item')),
    refund_amount numeric(14,2) NOT NULL CHECK (refund_amount > 0)
);

CREATE TABLE IF NOT EXISTS core.purchase_orders (
    po_id text PRIMARY KEY,
    supplier_id text NOT NULL REFERENCES core.suppliers (supplier_id),
    product_id text NOT NULL REFERENCES core.products (product_id),
    order_date date NOT NULL CHECK (order_date BETWEEN '2023-01-01' AND '2025-12-31'),
    expected_delivery_date date NOT NULL,
    actual_delivery_date date,
    delivery_status text NOT NULL CHECK (delivery_status IN ('Delivered', 'In Transit')),
    quantity_ordered integer NOT NULL CHECK (quantity_ordered >= 5),
    quantity_received integer,
    defective_units integer,
    unit_cost numeric(12,2) NOT NULL CHECK (unit_cost > 0),
    order_cost numeric(14,2) NOT NULL CHECK (order_cost > 0),
    CONSTRAINT po_dates_ordered CHECK (expected_delivery_date > order_date),
    CONSTRAINT po_received_range CHECK (quantity_received IS NULL OR quantity_received BETWEEN 1 AND quantity_ordered),
    CONSTRAINT po_defective_range CHECK (defective_units IS NULL OR (quantity_received IS NOT NULL AND defective_units BETWEEN 0 AND quantity_received)),
    CONSTRAINT po_delivery_fields CHECK (
        (delivery_status = 'Delivered' AND actual_delivery_date IS NOT NULL AND quantity_received IS NOT NULL AND defective_units IS NOT NULL)
        OR (delivery_status = 'In Transit' AND actual_delivery_date IS NULL AND quantity_received IS NULL AND defective_units IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS core.inventory_daily (
    product_id text NOT NULL REFERENCES core.products (product_id),
    date date NOT NULL CHECK (date BETWEEN '2023-01-01' AND '2025-12-31'),
    opening_stock integer NOT NULL CHECK (opening_stock >= 0),
    initial_stock_units integer NOT NULL CHECK (initial_stock_units >= 0),
    po_receipts_units integer NOT NULL CHECK (po_receipts_units >= 0),
    defective_writeoff_units integer NOT NULL CHECK (defective_writeoff_units >= 0),
    units_demanded integer NOT NULL CHECK (units_demanded >= 0),
    units_fulfilled integer NOT NULL CHECK (units_fulfilled >= 0),
    closing_stock integer NOT NULL CHECK (closing_stock >= 0),
    backorder_units integer NOT NULL CHECK (backorder_units >= 0),
    stockout_flag smallint NOT NULL CHECK (stockout_flag IN (0, 1)),
    inventory_value numeric(16,2) NOT NULL CHECK (inventory_value >= 0),
    PRIMARY KEY (product_id, date),
    CONSTRAINT inventory_stock_identity CHECK (closing_stock = opening_stock + initial_stock_units + po_receipts_units - defective_writeoff_units - units_fulfilled),
    CONSTRAINT inventory_not_both CHECK (NOT (closing_stock > 0 AND backorder_units > 0)),
    CONSTRAINT inventory_flag_matches CHECK ((stockout_flag = 1) = (backorder_units > 0))
);

CREATE TABLE IF NOT EXISTS core.shipments (
    shipment_id text PRIMARY KEY,
    order_id text NOT NULL UNIQUE REFERENCES core.orders (order_id),
    carrier text NOT NULL CHECK (carrier IN ('SwiftShip', 'BlueLine Express', 'GlobalPost', 'RapidRoute')),
    ship_date date,
    promised_delivery_date date NOT NULL,
    delivered_date date,
    shipment_status text NOT NULL CHECK (shipment_status IN ('Delivered', 'In Transit', 'Awaiting Shipment')),
    had_backorder smallint NOT NULL CHECK (had_backorder IN (0, 1)),
    handling_days integer CHECK (handling_days >= 0),
    transit_days integer CHECK (transit_days >= 1),
    delivery_delay_days integer,
    shipping_cost numeric(10,2) NOT NULL CHECK (shipping_cost > 0),
    CONSTRAINT shipments_status_fields CHECK (
        (shipment_status = 'Delivered' AND ship_date IS NOT NULL AND delivered_date IS NOT NULL AND handling_days IS NOT NULL AND transit_days IS NOT NULL AND delivery_delay_days IS NOT NULL)
        OR (shipment_status = 'In Transit' AND ship_date IS NOT NULL AND handling_days IS NOT NULL AND delivered_date IS NULL AND transit_days IS NULL AND delivery_delay_days IS NULL)
        OR (shipment_status = 'Awaiting Shipment' AND ship_date IS NULL AND handling_days IS NULL AND delivered_date IS NULL AND transit_days IS NULL AND delivery_delay_days IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS core.daily_metrics (
    date date PRIMARY KEY CHECK (date BETWEEN '2023-01-01' AND '2025-12-31'),
    orders_placed integer NOT NULL CHECK (orders_placed >= 0),
    orders_completed integer NOT NULL CHECK (orders_completed >= 0),
    orders_cancelled integer NOT NULL CHECK (orders_cancelled >= 0),
    new_customers integer NOT NULL CHECK (new_customers >= 0),
    active_customers integer NOT NULL CHECK (active_customers >= 0),
    units_sold integer NOT NULL CHECK (units_sold >= 0),
    gross_revenue numeric(16,2) NOT NULL CHECK (gross_revenue >= 0),
    discount_amount numeric(16,2) NOT NULL CHECK (discount_amount >= 0),
    cogs numeric(16,2) NOT NULL CHECK (cogs >= 0),
    gross_margin numeric(16,2) NOT NULL,
    shipping_cost numeric(16,2) NOT NULL CHECK (shipping_cost >= 0),
    payments_captured numeric(16,2) NOT NULL CHECK (payments_captured >= 0),
    payments_failed numeric(16,2) NOT NULL CHECK (payments_failed >= 0),
    refunds_amount numeric(16,2) NOT NULL CHECK (refunds_amount >= 0),
    po_spend numeric(16,2) NOT NULL CHECK (po_spend >= 0),
    listed_products integer NOT NULL CHECK (listed_products >= 0),
    products_in_stockout integer NOT NULL CHECK (products_in_stockout >= 0),
    backorder_units integer NOT NULL CHECK (backorder_units >= 0),
    inventory_value numeric(16,2) NOT NULL CHECK (inventory_value >= 0),
    CONSTRAINT daily_orders_identity CHECK (orders_placed = orders_completed + orders_cancelled)
);

CREATE INDEX IF NOT EXISTS ix_products_supplier_id ON core.products (supplier_id);
CREATE INDEX IF NOT EXISTS ix_customers_signup_date ON core.customers (signup_date);
CREATE INDEX IF NOT EXISTS ix_orders_customer_id ON core.orders (customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_order_date ON core.orders (order_date);
CREATE INDEX IF NOT EXISTS ix_order_items_product_id ON core.order_items (product_id);
CREATE INDEX IF NOT EXISTS ix_payments_payment_date ON core.payments (payment_date);
CREATE INDEX IF NOT EXISTS ix_refunds_order_id ON core.refunds (order_id);
CREATE INDEX IF NOT EXISTS ix_refunds_refund_date ON core.refunds (refund_date);
CREATE INDEX IF NOT EXISTS ix_purchase_orders_supplier_id ON core.purchase_orders (supplier_id);
CREATE INDEX IF NOT EXISTS ix_purchase_orders_product_id ON core.purchase_orders (product_id);
CREATE INDEX IF NOT EXISTS ix_purchase_orders_order_date ON core.purchase_orders (order_date);
CREATE INDEX IF NOT EXISTS ix_inventory_daily_date ON core.inventory_daily (date);
CREATE INDEX IF NOT EXISTS ix_inventory_daily_stockouts ON core.inventory_daily (date, product_id) WHERE stockout_flag = 1;
CREATE INDEX IF NOT EXISTS ix_shipments_carrier ON core.shipments (carrier);