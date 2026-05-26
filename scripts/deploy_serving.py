"""
deploy_serving.py — Deploys the trained model as an SPCS inference service.

Uses the Snowflake Model Registry's create_service() to launch the model as a
persistent SPCS service, with no custom Docker image required. The Model Registry
handles containerisation automatically.

The resulting service exposes the model as a SQL function:

    SELECT CHURN_PREDICTION_SVC!PREDICT(TENURE, MONTHLY_CHARGES, ...)
    FROM <DB>.FEATURES.CUSTOMER_FEATURES;

Or via the REST endpoint if ingress is enabled:

    POST https://<ingress-url>/predict
    {"data": [[tenure, monthly_charges, total_charges, ...]]}

Prerequisites:
  - Model must already be registered (run deploy_batch.py at least once and
    trigger the pipeline so that TASK_MODEL_TRAINING completes).
  - Compute pool must exist (run sql/02_setup_compute_pool.sql).
  - Image repository must exist (run sql/01_setup_environment.sql).

Usage (called by deploy.sh, not directly):
    SNOW_CONNECTION=<name> python scripts/deploy_serving.py <ENV>

Example:
    SNOW_CONNECTION=dev_conn python scripts/deploy_serving.py DEV
"""

import argparse
import logging
import os
import time
from pathlib import Path

import yaml
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


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
# Deploy
# ---------------------------------------------------------------------------

def deploy(env_name: str) -> None:
    cfg = load_config(env_name)

    db            = cfg["database"]
    schema        = cfg["schema"]
    compute_pool  = cfg.get("compute_pool", f"{env_name}_ML_COMPUTE_POOL")
    model_name    = cfg["model_name"]
    service_name  = cfg.get("service_name", "CHURN_PREDICTION_SVC")
    image_repo    = cfg.get("image_repo", f"{db}.{schema}.ML_IMAGE_REPO")

    service_fqn = f"{db}.{schema}.{service_name}"

    logger.info("=== Deploying SPCS inference service ===")
    logger.info("Environment  : %s", env_name)
    logger.info("Model        : %s (v_latest)", model_name)
    logger.info("Service      : %s", service_fqn)
    logger.info("Compute pool : %s", compute_pool)
    logger.info("Image repo   : %s", image_repo)

    session = get_session()

    # ------------------------------------------------------------------
    # Retrieve model version from Registry
    # ------------------------------------------------------------------
    reg = Registry(session=session)

    try:
        mv = reg.get_model(model_name).version("v_latest")
    except Exception as e:
        raise RuntimeError(
            f"Model '{model_name}' (v_latest) not found in the Registry. "
            "Run the batch pipeline at least once to train and register the model."
        ) from e

    logger.info("\nModel found. Deploying as SPCS service...")

    # ------------------------------------------------------------------
    # Deploy model as SPCS service
    #
    # create_service() builds a container image from the registered model
    # artefacts, pushes it to the image repository, and creates an SPCS
    # service backed by the compute pool.
    #
    # ingress_enabled=True exposes a REST endpoint for HTTP scoring;
    # set to False if you only need the SQL function interface.
    # ------------------------------------------------------------------
    mv.create_service(
        service_name=service_fqn,
        service_compute_pool=compute_pool,
        image_repo=image_repo,
        ingress_enabled=True,
        max_instances=1,
        # force_rebuild=True can be used to push updated model artefacts
    )

    # ------------------------------------------------------------------
    # Wait for service to reach RUNNING state
    # ------------------------------------------------------------------
    logger.info("Waiting for service to start (this may take a few minutes)...")
    _wait_for_service(session, service_fqn)

    # ------------------------------------------------------------------
    # Print usage instructions
    # ------------------------------------------------------------------
    logger.info("\n=== Service is RUNNING ===")
    logger.info(
        "\nSQL batch scoring:\n"
        "  SELECT %s!PREDICT(TENURE, MONTHLY_CHARGES, TOTAL_CHARGES, CONTRACT_TYPE)\n"
        "  FROM %s.FEATURES.CUSTOMER_FEATURES;",
        service_fqn, db,
    )
    logger.info(
        "\nTo check service status:\n"
        "  SHOW SERVICES IN SCHEMA %s.%s;\n"
        "  CALL SYSTEM$GET_SERVICE_STATUS('%s');",
        db, schema, service_fqn,
    )
    logger.info(
        "\nTo tear down the service when no longer needed:\n"
        "  DROP SERVICE IF EXISTS %s;",
        service_fqn,
    )

    session.close()
    logger.info("Done.")


def _wait_for_service(
    session: Session,
    service_fqn: str,
    poll_interval_s: int = 15,
    timeout_s: int = 600,
) -> None:
    """Polls until the service reaches RUNNING or raises on timeout/error."""
    elapsed = 0
    while elapsed < timeout_s:
        rows = session.sql(
            f"CALL SYSTEM$GET_SERVICE_STATUS('{service_fqn}')"
        ).collect()
        status_text = rows[0][0] if rows else ""

        if "RUNNING" in status_text.upper():
            return

        if any(s in status_text.upper() for s in ("FAILED", "SUSPENDED", "DELETING")):
            raise RuntimeError(
                f"Service entered unexpected state. Status: {status_text}\n"
                f"Check SHOW SERVICES IN SCHEMA for details."
            )

        logger.info("  Status: %s — retrying in %ds...", status_text.strip(), poll_interval_s)
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s

    raise TimeoutError(
        f"Service did not reach RUNNING within {timeout_s}s. "
        "Check compute pool capacity and image repository access."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy trained model as SPCS inference service"
    )
    parser.add_argument("env", choices=["DEV", "SIT", "UAT", "PRD"],
                        help="Target environment")
    args = parser.parse_args()
    deploy(args.env)
