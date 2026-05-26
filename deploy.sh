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
#      For ML Jobs mode also run:
#        snow sql -c <conn> -f sql/02_setup_compute_pool.sql -D env_prefix=DEV
#
# Usage:
#   ./deploy.sh <connection_name> <ENV> [--mode sprocs|mljobs]
#
# Examples:
#   ./deploy.sh my_dev_conn DEV
#   ./deploy.sh my_dev_conn DEV --mode mljobs
#   ./deploy.sh my_prd_conn PRD
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <connection_name> <ENV> [--mode sprocs|mljobs]"
    echo ""
    echo "  connection_name  Snow CLI connection from ~/.snowflake/connections.toml"
    echo "  ENV              DEV | SIT | UAT | PRD"
    echo "  --mode           sprocs (default) or mljobs"
    exit 1
fi

CONN="$1"
ENV="$2"
MODE="sprocs"

# Parse optional --mode flag
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "sprocs" && "$MODE" != "mljobs" ]]; then
    echo "Error: --mode must be 'sprocs' or 'mljobs'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve environment config (DB and schema from environments.yml)
# ---------------------------------------------------------------------------
# Requires python3 + pyyaml (installed via requirements.txt)
DB=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('config/environments.yml'))
print(cfg['$ENV']['database'])
")
SCHEMA=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('config/environments.yml'))
print(cfg['$ENV']['schema'])
")

STAGE="@${DB}.${SCHEMA}.ML_CODE_STAGE"

echo "======================================================="
echo " MLOps DAG Deploy"
echo " Connection : $CONN"
echo " Environment: $ENV"
echo " Mode       : $MODE"
echo " Stage      : $STAGE"
echo "======================================================="

# ---------------------------------------------------------------------------
# Upload source code to stage
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Uploading src/ml_pipeline.py -> $STAGE"
snow stage copy src/ml_pipeline.py "$STAGE" \
    --connection "$CONN" \
    --overwrite \
    --auto-compress false

echo "      Upload complete."

# ---------------------------------------------------------------------------
# Deploy DAG (stored procedures + task graph)
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Deploying DAG (mode=$MODE)..."
SNOW_CONNECTION="$CONN" python3 scripts/deploy_pipeline.py "$ENV" --mode "$MODE"

echo ""
echo "======================================================="
echo " Deployment complete."
echo ""
echo " To run the pipeline manually:"
echo "   snow sql -c $CONN -q \\"
echo "     \"EXECUTE TASK ${DB}.${SCHEMA}.ML_RETRAINING_PIPELINE\\\$TASK_FEATURE_ENGINEERING;\""
echo ""
echo " To check task run history:"
echo "   snow sql -c $CONN -q \\"
echo "     \"SELECT * FROM TABLE(${DB}.INFORMATION_SCHEMA.TASK_HISTORY()) ORDER BY SCHEDULED_TIME DESC LIMIT 20;\""
echo "======================================================="
