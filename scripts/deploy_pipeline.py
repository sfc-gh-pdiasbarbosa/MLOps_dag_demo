"""
deploy_pipeline.py — Builds and deploys the ML retraining DAG to Snowflake.

Reads connection details from the Snow CLI connection specified via the
SNOW_CONNECTION environment variable (set by deploy.sh).

Supports two execution modes:
  sprocs  (default) — each DAG task calls a registered Stored Procedure that
                      runs on a standard warehouse. Best for most workloads.
  mljobs            — each DAG task calls a Stored Procedure that submits an
                      ML Job to a Compute Pool (containerised). Best for
                      GPU-heavy or large-memory training.

Usage (called by deploy.sh, not directly):
    SNOW_CONNECTION=<name> python scripts/deploy_pipeline.py <ENV> [--mode sprocs|mljobs]

Examples:
    SNOW_CONNECTION=dev_conn python scripts/deploy_pipeline.py DEV
    SNOW_CONNECTION=dev_conn python scripts/deploy_pipeline.py DEV --mode mljobs
"""

import argparse
import logging
import os
import sys
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
    """
    Creates a Snowpark session from a Snow CLI named connection.
    The connection must exist in ~/.snowflake/connections.toml and be
    configured with the appropriate auth (key-pair, SSO, etc.).
    """
    conn_name = os.environ.get("SNOW_CONNECTION")
    if not conn_name:
        raise EnvironmentError(
            "SNOW_CONNECTION env var is not set. "
            "Call this script via deploy.sh or set it manually."
        )
    return Session.builder.config("connection_name", conn_name).create()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(env_name: str) -> dict:
    config_path = REPO_ROOT / "config" / "environments.yml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if env_name not in raw:
        raise ValueError(f"Environment '{env_name}' not found in environments.yml")

    # Merge: defaults first, then env-specific values win
    cfg = {**raw.get("default", {}), **raw[env_name]}
    return cfg


# ---------------------------------------------------------------------------
# ML Jobs helpers
# ---------------------------------------------------------------------------

def _make_mljob_sproc_body(func_name: str, compute_pool: str, stage: str) -> str:
    """
    Returns the Python source for a thin stored procedure that submits an
    ML Job for the given entry-point function and waits for completion.

    The ML Job runs ml_pipeline.<func_name>(session) on the compute pool.
    """
    return f'''\
import sys
from snowflake.ml.jobs import remote
from snowflake.snowpark import Session

def handler(session: Session) -> str:
    import importlib.util, os

    # Load ml_pipeline from the stage import
    spec = importlib.util.spec_from_file_location(
        "ml_pipeline",
        "/tmp/ml_pipeline.py",    # Snow injects stage imports into /tmp/
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, "{func_name}")

    decorated = remote(
        compute_pool="{compute_pool}",
        stage_name="{stage}",
    )(fn)
    return decorated(session)
'''


def register_mljob_sproc(
    session: Session,
    sp_name: str,
    func_name: str,
    compute_pool: str,
    stage: str,
    packages: list[str],
) -> None:
    """Registers a stored procedure that submits an ML Job for func_name."""
    logger.info("  Registering ML Job sproc: %s", sp_name)
    body = _make_mljob_sproc_body(func_name, compute_pool, stage)
    session.sql(f"""
        CREATE OR REPLACE PROCEDURE {sp_name}()
        RETURNS VARCHAR
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.11'
        PACKAGES = ({', '.join(repr(p) for p in packages)})
        IMPORTS = ('{stage}/ml_pipeline.py')
        HANDLER = 'handler'
        EXECUTE AS CALLER
        AS $$
{body}
        $$
    """).collect()


# ---------------------------------------------------------------------------
# Main deploy
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name":      "TASK_FEATURE_ENGINEERING",
        "func":      "feature_engineering_main",
        "packages":  ["snowflake-snowpark-python", "pandas", "snowflake-ml-python"],
    },
    {
        "name":      "TASK_MODEL_TRAINING",
        "func":      "model_training_main",
        "packages":  ["snowflake-snowpark-python", "pandas", "scikit-learn", "snowflake-ml-python"],
    },
    {
        "name":      "TASK_INFERENCE",
        "func":      "inference_main",
        "packages":  ["snowflake-snowpark-python", "pandas", "snowflake-ml-python"],
    },
]


def deploy(env_name: str, mode: str) -> None:
    cfg = load_config(env_name)

    db            = cfg["database"]
    schema        = cfg["schema"]
    warehouse     = cfg["warehouse"]
    compute_pool  = cfg.get("compute_pool", f"{env_name}_ML_COMPUTE_POOL")
    pipeline_name = cfg["pipeline_name"]
    schedule_cron = cfg.get("schedule")

    code_stage = f"@{db}.{schema}.ML_CODE_STAGE"

    logger.info("Deploying %s  env=%s  mode=%s", pipeline_name, env_name, mode.upper())
    logger.info("Target: %s.%s  warehouse: %s", db, schema, warehouse)

    session  = get_session()
    api_root = Root(session)

    # ------------------------------------------------------------------
    # Register stored procedures
    # ------------------------------------------------------------------
    logger.info("Registering stored procedures (mode=%s)...", mode)

    for task in TASKS:
        sp_name = f"{db}.{schema}.SP_{task['name']}"

        if mode == "mljobs":
            # Sproc submits an ML Job to the compute pool
            register_mljob_sproc(
                session,
                sp_name=sp_name,
                func_name=task["func"],
                compute_pool=compute_pool,
                stage=code_stage,
                packages=task["packages"] + ["snowflake-ml-python"],
            )
        else:
            # Sproc runs the task directly on the warehouse
            session.sproc.register_from_file(
                file_path=str(SRC_FILE),
                func_name=task["func"],
                name=sp_name,
                is_permanent=True,
                stage_location=code_stage,
                packages=task["packages"],
                replace=True,
                execute_as="caller",
                imports=[str(SRC_FILE)],
            )
            logger.info("  Registered: %s", sp_name)

    # ------------------------------------------------------------------
    # Build and deploy the DAG
    # ------------------------------------------------------------------
    logger.info("Building DAG: %s", pipeline_name)

    schema_obj = api_root.databases[db].schemas[schema]
    dag_op     = DAGOperation(schema_obj)

    schedule = Cron(schedule_cron, "UTC") if schedule_cron else None

    with DAG(
        pipeline_name,
        stage_location=code_stage,
        schedule=schedule,
        warehouse=warehouse,
        packages=["snowflake-snowpark-python"],
    ) as dag:
        dag_tasks = []
        for task in TASKS:
            sp_fqn   = f"{db}.{schema}.SP_{task['name']}"
            dag_task = DAGTask(
                task["name"],
                definition=f"CALL {sp_fqn}()",
                warehouse=warehouse,
            )
            dag_tasks.append(dag_task)

        # Chain: feature engineering >> training >> inference
        for i in range(len(dag_tasks) - 1):
            dag_tasks[i] >> dag_tasks[i + 1]

    dag_op.deploy(dag, mode="orreplace")
    logger.info("DAG '%s' deployed (suspended)", pipeline_name)

    # ------------------------------------------------------------------
    # Resume schedule in PRD; leave suspended elsewhere
    # ------------------------------------------------------------------
    if env_name == "PRD" and schedule_cron:
        root_task_fqn = f"{db}.{schema}.{pipeline_name}${TASKS[0]['name']}"
        session.sql(f"ALTER TASK {root_task_fqn} RESUME").collect()
        logger.info("Schedule resumed for PRD: %s", schedule_cron)
    else:
        logger.info(
            "DAG is suspended. To run manually:\n"
            "  EXECUTE TASK %s.%s.%s$%s;",
            db, schema, pipeline_name, TASKS[0]["name"],
        )

    session.close()
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy ML retraining DAG to Snowflake")
    parser.add_argument("env", choices=["DEV", "SIT", "UAT", "PRD"],
                        help="Target environment")
    parser.add_argument("--mode", choices=["sprocs", "mljobs"], default="sprocs",
                        help="sprocs: run tasks as Stored Procedures on a warehouse (default). "
                             "mljobs: run tasks as ML Jobs on a Compute Pool.")
    args = parser.parse_args()
    deploy(args.env, args.mode)
