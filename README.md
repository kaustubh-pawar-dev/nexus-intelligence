# NEXUS Intelligence

**Enterprise Decision Intelligence & Predictive Analytics Platform** — a reproducible data engineering, business analytics, statistical analysis, and customer churn modeling project built entirely on synthetic enterprise data.

> **Data notice:** All data is synthetic. Results demonstrate analytical methods and are not evidence of performance on real businesses or customers.

## Business Problem

Enterprise data is often distributed across operational systems and affected by missing values, duplicates, inconsistent formats, and data-quality issues. NEXUS simulates an enterprise environment and builds a reproducible workflow from data generation and cleaning to business intelligence, statistical analysis, and predictive modeling.

## Implemented Features

- **Synthetic data generation:** 11 business tables covering customers, products, suppliers, orders, payments, refunds, purchase orders, inventory, shipments, and business metrics.
- **Data quality engineering:** Injected defects, typed staging tables, deduplication, validation, row-level issue logging, and quarantining.
- **PostgreSQL warehouse:** Raw, staging, analytics, and core schemas for data processing and reconciliation.
- **SQL business intelligence:** 11 analytics queries covering revenue trends, customer retention, customer lifetime value, RFM segmentation, supplier performance, inventory turnover, and revenue leakage.
- **Statistical analysis:** Seven hypothesis tests with effect sizes and Holm-adjusted p-values.
- **Multivariate analysis:** Regression models and confounding checks, without making unsupported causal claims.
- **Customer churn modeling:** 23,627 customer-cutoff records across 25 monthly cutoffs, using behavioral features, temporal evaluation, rule-based baselines, and logistic regression.
- **Model evaluation:** ROC-AUC, PR-AUC, top-20% precision and recall, lift, and Brier score.
- **Automated testing:** 243 pytest tests passed in the current local environment.

## Churn Modeling Results

The following results were obtained from the synthetic dataset using the temporal test cutoffs.

| Metric | Logistic regression |
|---|---:|
| ROC-AUC | 0.735 |
| PR-AUC | 0.605 |
| Precision among top 20% flagged | 0.654 |
| Recall among top 20% flagged | 0.337 |
| Lift among top 20% flagged | 1.686 |
| Brier score | 0.201 |

These results demonstrate the modeling workflow only. They should not be interpreted as expected performance on real customer data.

**Evaluation limitation:** Customers can appear at different monthly cutoffs in both training and test sets. The evaluation therefore does not guarantee customer-level independence. The churn label represents the absence of a completed order during the defined future observation window, not necessarily account cancellation.

## Technology Stack

- **Language:** Python
- **Database:** PostgreSQL, SQL
- **Data processing:** pandas, NumPy
- **Statistics:** SciPy, statsmodels
- **Machine learning:** scikit-learn
- **Database access:** SQLAlchemy, psycopg
- **Configuration and testing:** python-dotenv, pytest

## Architecture

```text
Synthetic data generators
          |
          v
Raw data and injected defects
          |
          v
PostgreSQL raw schema
          |
          v
Staging, cleaning and validation
          |
          v
Analytics and core schemas
          |
          +--> SQL business analytics
          +--> Statistical and multivariate analysis
          +--> Customer churn dataset and baseline models
```

## Project Structure

```text
src/nexus/    Core library: data generation, cleaning, database, analytics and models
scripts/      Pipeline entry points and analysis runners
sql/          Database schemas and analytics queries
tests/        Automated tests
docs/         Data dictionary and metric definitions
data/, logs/  Locally generated artifacts
```

## Setup

Requirements: Python 3.10+ and a configured PostgreSQL server.

```powershell
git clone https://github.com/kaustubh-pawar-dev/nexus-intelligence.git
cd nexus-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

Configure the database settings in `.env` using `.env.example` as a guide. Never commit `.env` or credentials.

Create the PostgreSQL role and database according to your local PostgreSQL setup. For example:

```powershell
psql -U postgres -c "CREATE ROLE nexus_app LOGIN PASSWORD 'your-password-here';"
psql -U postgres -c "CREATE DATABASE nexus_intelligence OWNER nexus_app;"
```

## Running the Pipeline

Check the environment and generate the synthetic datasets:

```powershell
python scripts/check_environment.py
python scripts/generate_reference_data.py
python scripts/generate_customer_data.py
python scripts/generate_order_data.py
python scripts/generate_payment_data.py
python scripts/generate_purchase_order_data.py
python scripts/generate_inventory_data.py
python scripts/generate_shipment_data.py
python scripts/generate_daily_metrics_data.py
python scripts/generate_dirty_data.py
```

Build the database layers:

```powershell
python scripts/init_database.py
python scripts/load_core.py
python scripts/load_raw.py
python scripts/build_staging.py
python scripts/build_analytics.py
```

Run the existing analyses and churn experiment:

```powershell
python scripts/run_sql_analytics.py
python scripts/run_sql_analytics_2.py
python scripts/run_stats.py
python scripts/run_multivariate.py
python scripts/run_churn.py
```

See `docs/data_dictionary.md` for table definitions, defect types, layer descriptions, and metric definitions.

## Testing

```powershell
python -m pytest -q
```

**Latest local verification:** 243 tests passed. Results depend on the local environment and database configuration.

## Roadmap

- Customer-level holdout evaluation for churn modeling
- Root-cause exploration and financial impact estimation
- What-if scenario analysis
- Intervention tracking
- Interactive analytics dashboard

## Disclaimer

This is an educational and portfolio project using synthetic data. It is not a production decision system, and its model results should not be generalized to real-world populations.
