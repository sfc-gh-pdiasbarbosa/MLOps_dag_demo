# MLOps DAG Demo

Best-practices example of an ML retraining pipeline on Snowflake, built with
**Task DAGs**, **Feature Store**, and **Model Registry**.

The pipeline has three stages wired together as a DAG:

```
TASK_FEATURE_ENGINEERING  >>  TASK_MODEL_TRAINING  >>  TASK_INFERENCE
```

Each task can run as either a **Stored Procedure** (warehouse compute) or an
**ML Job** (container compute pool) — chosen at deploy time with `--mode`.

---

## Repository layout

```
.
├── deploy.sh                     # One-command manual deploy (Snow CLI)
├── requirements.txt
├── config/
│   └── environments.yml          # DB / schema / warehouse per environment
├── src/
│   └── ml_pipeline.py            # Feature engineering, training, inference logic
├── scripts/
│   └── deploy_pipeline.py        # DAG builder (called by deploy.sh)
└── sql/
    ├── 01_setup_environment.sql  # Provision DB, schemas, warehouse, stage
    └── 02_setup_compute_pool.sql # Compute pool (--mode mljobs only)
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
   # Core objects (DB, schemas, warehouse, stage)
   snow sql -c <conn> -f sql/01_setup_environment.sql \
            -D env_prefix=DEV -D wh_size=X-SMALL

   # Compute pool — only if using --mode mljobs
   snow sql -c <conn> -f sql/02_setup_compute_pool.sql -D env_prefix=DEV
   ```

---

## Deploy

```bash
chmod +x deploy.sh

# Stored Procedures mode (default — runs on a virtual warehouse)
./deploy.sh <connection_name> DEV

# ML Jobs mode (runs on a Compute Pool — better for GPU/large-memory training)
./deploy.sh <connection_name> DEV --mode mljobs
```

---

## Execution modes

| Mode | Compute | Best for |
|------|---------|----------|
| `sprocs` (default) | Virtual Warehouse | Most ML workloads; no extra infrastructure |
| `mljobs` | Compute Pool (container) | GPU training, large datasets, custom container images |

Both modes deploy the same three-task DAG. The only difference is what each
task does internally:

- **sprocs**: the DAG task calls a stored procedure that runs `ml_pipeline.py`
  directly on the warehouse.
- **mljobs**: the DAG task calls a stored procedure that submits a containerised
  ML Job to the compute pool and waits for it to finish.

---

## Running the pipeline manually

After deploying, the DAG is left **suspended** in non-PRD environments.
To trigger a run:

```bash
snow sql -c <conn> -q "
  EXECUTE TASK <DB>.PIPELINES.ML_RETRAINING_PIPELINE\$TASK_FEATURE_ENGINEERING;
"
```

Check run history:

```bash
snow sql -c <conn> -q "
  SELECT * FROM TABLE(<DB>.INFORMATION_SCHEMA.TASK_HISTORY())
  ORDER BY SCHEDULED_TIME DESC
  LIMIT 20;
"
```

In **PRD**, the DAG schedule is automatically resumed after deploy (default:
daily at 02:00 UTC — set `schedule` in `environments.yml` to `null` to skip).

---

## Adapting to your data

The pipeline ships with a placeholder `CUSTOMERS` table schema. To use your
own data:

1. Update `config/environments.yml` → `tables.raw_data` to point to your source table.
2. Edit `feature_engineering_task()` in `src/ml_pipeline.py` to define your
   feature transformations.
3. Update `model_training_task()` with your model choice and target column.
4. Re-run `deploy.sh` to redeploy.
