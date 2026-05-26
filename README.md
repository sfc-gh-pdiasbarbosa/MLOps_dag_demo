# MLOps DAG Demo

Best-practices example of an ML retraining pipeline on Snowflake, showcasing
how to match each pipeline stage to its natural compute type.

---

## Pipelines

### 1. Batch retraining pipeline (`deploy.sh batch`)

A Task DAG with three stages, each on a different compute:

```
TASK_FEATURE_ENGINEERING          TASK_MODEL_TRAINING              TASK_BATCH_INFERENCE
  Serverless task             >>    ML Job (Compute Pool)    >>    Warehouse stored proc
  Auto-sized warehouse              Container Runtime               Predictable, short
  Billed per run                    GPU/large-memory capable        Standard WH credits
  Feature Store refresh             scikit-learn, xgboost, torch    Reads feature view
```

| Stage | Compute | Why |
|---|---|---|
| Feature engineering | **Serverless** | SQL-heavy Feature Store refresh; short-lived; Snowflake auto-sizes |
| Model training | **ML Job** (Compute Pool) | Memory/CPU-intensive; Container Runtime has full ML library set |
| Batch inference | **Warehouse** | Predictable duration; standard warehouse is the right fit |

### 2. Serving pipeline (`deploy.sh serving`)

Deploys the trained model as a persistent **SPCS inference service** using the
Model Registry's `create_service()`. No custom Docker image required — the
registry handles containerisation automatically.

After deployment, the model is callable as a SQL function:

```sql
SELECT CHURN_PREDICTION_SVC!PREDICT(TENURE, MONTHLY_CHARGES, TOTAL_CHARGES, CONTRACT_TYPE)
FROM DEV_ML_DB.FEATURES.CUSTOMER_FEATURES;
```

---

## Repository layout

```
.
├── deploy.sh                     # Entry point for both pipelines
├── requirements.txt
├── config/
│   └── environments.yml          # DB / schema / warehouse / compute pool per env
├── src/
│   └── ml_pipeline.py            # Core ML logic (feature engineering, training, inference)
├── scripts/
│   ├── deploy_batch.py           # Builds and deploys the mixed-compute DAG
│   └── deploy_serving.py         # Deploys the SPCS model inference service
└── sql/
    ├── 01_setup_environment.sql  # Provision DB, schemas, warehouse, stage, image repo
    └── 02_setup_compute_pool.sql # Compute pool (required for training + serving)
```

---

## Prerequisites

1. **Snow CLI** installed and a named connection configured:
   ```bash
   pip install snowflake-cli
   snow connection add          # follow prompts; use key-pair auth for automation
   snow connection test -c <name>
   ```

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Snowflake objects** provisioned (run once per environment):
   ```bash
   # Core objects: DB, schemas, warehouse, stage, image repository
   snow sql -c <conn> -f sql/01_setup_environment.sql \
            -D env_prefix=DEV -D wh_size=X-SMALL

   # Compute pool (required for ML Job training and SPCS serving)
   snow sql -c <conn> -f sql/02_setup_compute_pool.sql -D env_prefix=DEV
   ```

4. **Privilege**: the role that runs the batch pipeline needs
   `EXECUTE MANAGED TASK` (for the serverless feature engineering task).

---

## Deploy

```bash
chmod +x deploy.sh

# Deploy the mixed-compute retraining DAG
./deploy.sh batch my_dev_conn DEV

# Deploy the SPCS inference service
# (requires the batch pipeline to have run at least once to produce a trained model)
./deploy.sh serving my_dev_conn DEV
```

---

## Running the batch pipeline manually

After deploying, the DAG is left **suspended** in non-PRD environments.

```bash
# Trigger a manual run
snow sql -c <conn> -q "
  EXECUTE TASK DEV_ML_DB.PIPELINES.ML_RETRAINING_PIPELINE\$TASK_FEATURE_ENGINEERING;
"

# Check run history
snow sql -c <conn> -q "
  SELECT * FROM TABLE(DEV_ML_DB.INFORMATION_SCHEMA.TASK_HISTORY())
  ORDER BY SCHEDULED_TIME DESC LIMIT 20;
"
```

In **PRD**, the DAG schedule is automatically resumed after deploy (default:
daily at 02:00 UTC — set `schedule: null` in `environments.yml` to skip).

---

## Compute reference

| Compute type | When to use | Billing model |
|---|---|---|
| **Serverless task** | Short, SQL-heavy tasks with variable load | Per compute-hour of actual usage |
| **ML Job (Compute Pool)** | Memory/CPU-intensive training; GPU workloads | Per node-hour while the job runs |
| **Warehouse** | Predictable, user-controlled compute | Per credit while warehouse is active |
| **SPCS service** | Persistent online inference endpoint | Per node-hour while service is running |

---

## Adapting to your data

1. Update `config/environments.yml` → `tables.raw_data` to point to your source table.
2. Edit `feature_engineering_task()` in `src/ml_pipeline.py` with your feature transformations.
3. Swap the `LogisticRegression` in `model_training_task()` for your model of choice.
4. Re-run `./deploy.sh batch <conn> <env>` to redeploy.
