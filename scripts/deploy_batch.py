"""
deploy_batch.py — Deploys the ML retraining DAG with per-task compute.

Each stage of the pipeline uses the compute type that best matches its workload:

  TASK_FEATURE_ENGINEERING  →  Serverless task
    Snowflake auto-sizes the warehouse based on historical runs.
    Good fit for Feature Store refresh: it's SQL-heavy and short-lived,
    so paying per-run is cheaper than keeping a warehouse alive.

  TASK_MODEL_TRAINING  →  ML Job on a Compute Pool (SPCS)
    A thin stored procedure submits an ML Job to the compute pool and
    waits for completion. The actual training runs in a container with
    access to GPU/high-memory nodes and the full Container Runtime
    package set (XGBoost, PyTorch, etc.).

  TASK_BATCH_INFERENCE  →  Warehouse stored procedure
    Batch scoring reads the feature view and writes predictions.
    Predictable, short duration — warehouse compute is the right fit.

Usage (called by deploy.sh, not directly):
    SNOW_CONNECTION=<name> python scripts/deploy_batch.py <ENV>

Example:
    SNOW_CONNECTION=dev_conn python scripts/deploy_batch.py DEV
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from snowflake.core import Root
from snowflake.core.task import Cron
from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask
from snowflake.snowpark import Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_FILE  = REPO_ROOT / "src" / "ml_pipeline.py"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def get_session() -> Session:
    conn_name = os.environ.get("SNOW_CONNECTION")
    if not conn_name:
        raise EnvironmentError(
            "SNOW_CONNECTION env var is not set. Call this script via deploy.sh."
        )
    return Session.builder.config("connection_name", conn_name).create()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(env_name: str) -> dict:
    config_path = REPO_ROOT / "config" / "environments.yml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if env_name not in raw:
        raise ValueError(f"Environment '{env_name}' not found in environments.yml")
    return {**raw.get("default", {}), **raw[env_name]}


# ---------------------------------------------------------------------------
# Stored procedure registration helpers
# ---------------------------------------------------------------------------

def _register_warehouse_sproc(
    session: Session,
    sp_name: str,
    func_name: str,
    packages: list[str],
    stage: str,
) -> None:
    """Registers a stored procedure that runs directly on a warehouse."""
    session.sproc.register_from_file(
        file_path=str(SRC_FILE),
        func_name=func_name,
        name=sp_name,
        is_permanent=True,
        stage_location=stage,
        packages=packages,
        replace=True,
        execute_as="caller",
        imports=[str(SRC_FILE)],
    )
    logger.info("  Registered warehouse sproc: %s", sp_name)


def _register_mljob_sproc(
    session: Session,
    sp_name: str,
    func_name: str,
    compute_pool: str,
    stage: str,
    packages: list[str],
) -> None:
    """
    Registers a stored procedure that submits an ML Job to a Compute Pool.

    The sproc itself is lightweight (runs on the warehouse that hosts the
    TASK_MODEL_TRAINING task). All heavy compute happens inside the ML Job
    container, which runs on the Compute Pool.
    """
    body = f'''\
import importlib.util
from snowflake.ml.jobs import remote
from snowflake.snowpark import Session


def handler(session: Session) -> str:
    # ml_pipeline.py is injected into /tmp/ via the IMPORTS clause
    spec = importlib.util.spec_from_file_location("ml_pipeline", "/tmp/ml_pipeline.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn        = getattr(mod, "{func_name}")
    decorated = remote(compute_pool="{compute_pool}", stage_name="{stage}")(fn)
    return decorated(session)
'''
    packages_sql = ", ".join(repr(p) for p in packages)
    session.sql(f"""
        CREATE OR REPLACE PROCEDURE {sp_name}()
        RETURNS VARCHAR
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.11'
        PACKAGES = ({packages_sql})
        IMPORTS = ('{stage}/ml_pipeline.py')
        HANDLER = 'handler'
        EXECUTE AS CALLER
        AS $$
{body}
        $$
    """).collect()
    logger.info("  Registered ML Job sproc: %s  (compute pool: %s)", sp_name, compute_pool)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Feature engineering: serverless — auto-sized warehouse, billed per run.
TASK_FE = {
    "name":     "TASK_FEATURE_ENGINEERING",
    "func":     "feature_engineering_main",
    "compute":  "serverless",
    "packages": ["snowflake-snowpark-python", "pandas", "snowflake-ml-python"],
}

# Model training: ML Job — container compute, can target GPU pools.
TASK_TRAIN = {
    "name":     "TASK_MODEL_TRAINING",
    "func":     "model_training_main",
    "compute":  "mljob",
    "packages": ["snowflake-snowpark-python", "pandas", "scikit-learn", "snowflake-ml-python"],
}

# Batch inference: warehouse — predictable, short duration.
TASK_INFER = {
    "name":     "TASK_BATCH_INFERENCE",
    "func":     "inference_main",
    "compute":  "warehouse",
    "packages": ["snowflake-snowpark-python", "pandas", "snowflake-ml-python"],
}

PIPELINE_TASKS = [TASK_FE, TASK_TRAIN, TASK_INFER]


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

def deploy(env_name: str) -> None:
    cfg = load_config(env_name)

    db            = cfg["database"]
    schema        = cfg["schema"]
    warehouse     = cfg["warehouse"]
    compute_pool  = cfg.get("compute_pool", f"{env_name}_ML_COMPUTE_POOL")
    pipeline_name = cfg["pipeline_name"]
    schedule_cron = cfg.get("schedule")

    code_stage = f"@{db}.{schema}.ML_CODE_STAGE"

    logger.info("=== Deploying batch pipeline: %s ===", pipeline_name)
    logger.info("Environment : %s", env_name)
    logger.info("Target      : %s.%s", db, schema)
    logger.info("Warehouse   : %s  (training/inference tasks)", warehouse)
    logger.info("Compute pool: %s  (training ML Job)", compute_pool)

    session  = get_session()
    api_root = Root(session)

    # ------------------------------------------------------------------
    # Register stored procedures
    # ------------------------------------------------------------------
    logger.info("\nRegistering stored procedures...")

    for task in PIPELINE_TASKS:
        sp_name = f"{db}.{schema}.SP_{task['name']}"

        if task["compute"] == "mljob":
            _register_mljob_sproc(
                session,
                sp_name=sp_name,
                func_name=task["func"],
                compute_pool=compute_pool,
                stage=code_stage,
                packages=task["packages"] + ["snowflake-ml-python"],
            )
        else:
            # Both "serverless" and "warehouse" tasks call plain sprocs.
            # The compute type is determined by the DAGTask configuration,
            # not the stored procedure itself.
            _register_warehouse_sproc(
                session,
                sp_name=sp_name,
                func_name=task["func"],
                packages=task["packages"],
                stage=code_stage,
            )

    # ------------------------------------------------------------------
    # Build and deploy DAG with per-task compute
    # ------------------------------------------------------------------
    logger.info("\nBuilding DAG with mixed compute...")

    schema_obj = api_root.databases[db].schemas[schema]
    dag_op     = DAGOperation(schema_obj)

    schedule = Cron(schedule_cron, "UTC") if schedule_cron else None

    with DAG(
        pipeline_name,
        stage_location=code_stage,
        schedule=schedule,
        # No default warehouse: each task declares its own compute.
    ) as dag:

        # Serverless: omit warehouse → Snowflake auto-sizes based on history.
        # Requires EXECUTE MANAGED TASK privilege on the role.
        task_fe = DAGTask(
            TASK_FE["name"],
            definition=f"CALL {db}.{schema}.SP_{TASK_FE['name']}()",
        )

        # Warehouse: the sproc submits an ML Job; the task itself is lightweight.
        task_train = DAGTask(
            TASK_TRAIN["name"],
            definition=f"CALL {db}.{schema}.SP_{TASK_TRAIN['name']}()",
            warehouse=warehouse,
        )

        # Warehouse: standard batch scoring sproc.
        task_infer = DAGTask(
            TASK_INFER["name"],
            definition=f"CALL {db}.{schema}.SP_{TASK_INFER['name']}()",
            warehouse=warehouse,
        )

        # Linear dependency chain
        task_fe >> task_train >> task_infer

    dag_op.deploy(dag, mode="orreplace")
    logger.info("DAG '%s' deployed (suspended)", pipeline_name)

    # ------------------------------------------------------------------
    # Resume in PRD; leave suspended in lower environments
    # ------------------------------------------------------------------
    if env_name == "PRD" and schedule_cron:
        root_task = f"{db}.{schema}.{pipeline_name}${TASK_FE['name']}"
        session.sql(f"ALTER TASK {root_task} RESUME").collect()
        logger.info("Schedule resumed for PRD: %s UTC", schedule_cron)
    else:
        logger.info(
            "\nDAG is suspended. To trigger a manual run:\n"
            "  EXECUTE TASK %s.%s.%s$%s;",
            db, schema, pipeline_name, TASK_FE["name"],
        )

    logger.info(
        "\nCompute summary:\n"
        "  %-30s  serverless  (auto-sized, billed per run)\n"
        "  %-30s  ML Job      (compute pool: %s)\n"
        "  %-30s  warehouse   (%s)",
        TASK_FE["name"], TASK_TRAIN["name"], compute_pool,
        TASK_INFER["name"], warehouse,
    )

    session.close()
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy ML retraining DAG with per-task compute to Snowflake"
    )
    parser.add_argument("env", choices=["DEV", "SIT", "UAT", "PRD"],
                        help="Target environment")
    args = parser.parse_args()
    deploy(args.env)
