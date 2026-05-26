#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Manual ML pipeline deployment via Snow CLI
#
# Prerequisites:
#   1. Snow CLI installed:  pip install snowflake-cli
#   2. A named connection in ~/.snowflake/connections.toml
#   3. Python dependencies:  pip install -r requirements.txt
#   4. Environment objects provisioned:
#        snow sql -c <conn> -f sql/01_setup_environment.sql -D env_prefix=DEV -D wh_size=X-SMALL
#      For training on a compute pool (batch pipeline) and serving (both pipelines):
#        snow sql -c <conn> -f sql/02_setup_compute_pool.sql -D env_prefix=DEV
#
# Usage:
#   ./deploy.sh batch   <connection_name> <ENV>
#   ./deploy.sh serving <connection_name> <ENV>
#
# Examples:
#   ./deploy.sh batch   my_dev_conn DEV
#   ./deploy.sh batch   my_prd_conn PRD
#   ./deploy.sh serving my_dev_conn DEV
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <pipeline> <connection_name> <ENV>"
    echo ""
    echo "  pipeline         batch | serving"
    echo "  connection_name  Snow CLI connection from ~/.snowflake/connections.toml"
    echo "  ENV              DEV | SIT | UAT | PRD"
    echo ""
    echo "Examples:"
    echo "  $0 batch   my_dev_conn DEV    # Deploy mixed-compute retraining DAG"
    echo "  $0 serving my_dev_conn DEV    # Deploy SPCS model inference service"
    exit 1
fi

PIPELINE="$1"
CONN="$2"
ENV="$3"

if [[ "$PIPELINE" != "batch" && "$PIPELINE" != "serving" ]]; then
    echo "Error: pipeline must be 'batch' or 'serving'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve DB and schema from environments.yml
# ---------------------------------------------------------------------------
DB=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('config/environments.yml'))
print(cfg['$ENV']['database'])
")
SCHEMA=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('config/environments.yml'))
print(cfg['$ENV']['schema'])
")

STAGE="@${DB}.${SCHEMA}.ML_CODE_STAGE"

echo "======================================================="
echo " MLOps Pipeline Deploy"
echo " Pipeline   : $PIPELINE"
echo " Connection : $CONN"
echo " Environment: $ENV"
echo "======================================================="

# ---------------------------------------------------------------------------
# Batch pipeline: upload code + deploy mixed-compute DAG
# ---------------------------------------------------------------------------
if [[ "$PIPELINE" == "batch" ]]; then

    echo ""
    echo "[1/2] Uploading src/ml_pipeline.py -> $STAGE"
    snow stage copy src/ml_pipeline.py "$STAGE" \
        --connection "$CONN" \
        --overwrite \
        --auto-compress false
    echo "      Upload complete."

    echo ""
    echo "[2/2] Deploying batch DAG (serverless FE → ML Job training → warehouse inference)..."
    SNOW_CONNECTION="$CONN" python3 scripts/deploy_batch.py "$ENV"

    echo ""
    echo "======================================================="
    echo " Batch pipeline deployed."
    echo ""
    echo " Trigger a manual run:"
    echo "   snow sql -c $CONN -q \\"
    echo "     \"EXECUTE TASK ${DB}.${SCHEMA}.ML_RETRAINING_PIPELINE\\\$TASK_FEATURE_ENGINEERING;\""
    echo ""
    echo " Check run history:"
    echo "   snow sql -c $CONN -q \\"
    echo "     \"SELECT * FROM TABLE(${DB}.INFORMATION_SCHEMA.TASK_HISTORY()) ORDER BY SCHEDULED_TIME DESC LIMIT 20;\""
    echo "======================================================="

# ---------------------------------------------------------------------------
# Serving pipeline: deploy model as SPCS inference service
# ---------------------------------------------------------------------------
elif [[ "$PIPELINE" == "serving" ]]; then

    echo ""
    echo "[1/1] Deploying SPCS inference service from Model Registry..."
    echo "      (the batch pipeline must have run at least once to produce a trained model)"
    SNOW_CONNECTION="$CONN" python3 scripts/deploy_serving.py "$ENV"

    echo ""
    echo "======================================================="
    echo " Serving pipeline deployed."
    echo ""
    echo " Check service status:"
    echo "   snow sql -c $CONN -q \\"
    echo "     \"SHOW SERVICES IN SCHEMA ${DB}.${SCHEMA};\""
    echo ""
    echo " Tear down service when no longer needed:"
    echo "   snow sql -c $CONN -q \\"
    echo "     \"DROP SERVICE IF EXISTS ${DB}.${SCHEMA}.CHURN_PREDICTION_SVC;\""
    echo "======================================================="

fi
