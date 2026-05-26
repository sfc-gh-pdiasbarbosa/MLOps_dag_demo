"""
ml_pipeline.py — Core ML logic for the churn prediction pipeline.

Contains three task functions that form the DAG:
  1. feature_engineering_task  – creates/refreshes Feature Store feature views
  2. model_training_task       – trains a model and registers it in the Model Registry
  3. inference_task            – runs batch predictions and writes to an output table

Each task also has a no-argument entry point (e.g. feature_engineering_main)
used by stored procedures and ML Jobs, which derive config from the session context.
"""

import logging
from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode
from sklearn.linear_model import LogisticRegression
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Task functions  (called by the entry points below)
# =============================================================================

def feature_engineering_task(session: Session, source_table: str, feature_view_fqn: str) -> str:
    """
    Creates or refreshes a Feature View in the Snowflake Feature Store.

    The Feature View is backed by a Dynamic Table that auto-refreshes daily,
    so downstream training always reads up-to-date features without manual
    intervention.

    Args:
        session:          Active Snowpark session.
        source_table:     Fully-qualified raw data table  (DB.SCHEMA.TABLE).
        feature_view_fqn: Fully-qualified target Feature View  (DB.SCHEMA.VIEW_NAME).
    """
    logger.info("Feature engineering: %s -> %s", source_table, feature_view_fqn)

    parts = feature_view_fqn.split(".")
    if len(parts) == 3:
        db, schema, fv_name = parts
    else:
        db     = session.get_current_database().strip('"')
        schema = session.get_current_schema().strip('"')
        fv_name = feature_view_fqn

    fs = FeatureStore(
        session=session,
        database=db,
        name=schema,
        default_warehouse=session.get_current_warehouse().strip('"'),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    entity = Entity(name="CUSTOMER_ENTITY", join_keys=["CUSTOMER_ID"])
    fs.register_entity(entity, if_exists=CreationMode.CREATE_IF_NOT_EXIST)

    df_raw      = session.table(source_table)
    df_features = df_raw.fillna(0)

    fv = FeatureView(
        name=fv_name,
        entities=[entity],
        feature_df=df_features,
        refresh_freq="1 day",
        desc="Customer churn features – auto-refreshed daily",
    )

    fs.register_feature_view(fv, version="v1", if_exists=CreationMode.CREATE_OR_OVERWRITE)

    return f"OK: Feature View {feature_view_fqn} (v1) registered in {db}.{schema}"


def model_training_task(
    session: Session,
    feature_view_fqn: str,
    model_name: str,
) -> str:
    """
    Reads features from the Feature Store, trains a classifier, and registers
    it in the Snowflake Model Registry.

    The Model Registry stores versioned model artefacts alongside metadata
    (sample input, conda dependencies, comment) for reproducibility.

    Args:
        session:          Active Snowpark session.
        feature_view_fqn: Fully-qualified Feature View table to read from.
        model_name:       Name to register the model under in the Registry.
    """
    logger.info("Model training: features=%s model=%s", feature_view_fqn, model_name)

    df = session.table(feature_view_fqn).to_pandas()

    if "TARGET_LABEL" not in df.columns:
        raise ValueError("TARGET_LABEL column not found in feature view")

    X = df.drop(columns=["TARGET_LABEL", "CUSTOMER_ID"])
    y = df["TARGET_LABEL"]

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    reg = Registry(session=session)
    reg.log_model(
        model=clf,
        model_name=model_name,
        version_name="v_latest",
        conda_dependencies=["scikit-learn", "pandas"],
        comment="Logistic regression trained on Feature Store data",
        sample_input_data=X.head(),
    )

    return f"OK: Model '{model_name}' (v_latest) registered"


def inference_task(
    session: Session,
    feature_view_fqn: str,
    model_name: str,
    output_table: str,
) -> str:
    """
    Loads the latest registered model and writes batch predictions to an
    output table, appending each run for historical tracking.

    Args:
        session:          Active Snowpark session.
        feature_view_fqn: Fully-qualified Feature View to score.
        model_name:       Model name in the Registry.
        output_table:     Fully-qualified output table  (DB.SCHEMA.TABLE).
    """
    logger.info("Inference: model=%s -> %s", model_name, output_table)

    reg       = Registry(session=session)
    model_ref = reg.get_model(model_name).version("v_latest")

    df_features = session.table(feature_view_fqn)
    predictions = model_ref.run(df_features, function_name="predict")
    predictions.write.mode("append").save_as_table(output_table)

    return f"OK: Predictions appended to {output_table}"


# =============================================================================
# Entry points  (used by stored procedures and ML Jobs)
#
# These zero-argument functions derive config from the current session context
# (database / schema), making them environment-agnostic at call time.
# =============================================================================

MODEL_NAME = "CHURN_PREDICTION_MODEL"


def feature_engineering_main(session: Session) -> str:
    """Stored procedure / ML Job entry point for feature engineering."""
    db     = session.get_current_database().strip('"')
    schema = session.get_current_schema().strip('"')
    return feature_engineering_task(
        session,
        source_table=f"{db}.PUBLIC.CUSTOMERS",
        feature_view_fqn=f"{db}.FEATURES.CUSTOMER_FEATURES",
    )


def model_training_main(session: Session) -> str:
    """Stored procedure / ML Job entry point for model training."""
    db = session.get_current_database().strip('"')
    return model_training_task(
        session,
        feature_view_fqn=f"{db}.FEATURES.CUSTOMER_FEATURES",
        model_name=MODEL_NAME,
    )


def inference_main(session: Session) -> str:
    """Stored procedure / ML Job entry point for batch inference."""
    db     = session.get_current_database().strip('"')
    schema = session.get_current_schema().strip('"')
    return inference_task(
        session,
        feature_view_fqn=f"{db}.FEATURES.CUSTOMER_FEATURES",
        model_name=MODEL_NAME,
        output_table=f"{db}.OUTPUT.CHURN_PREDICTIONS",
    )


def main(session: Session) -> str:
    """
    Runs the full pipeline sequentially in a single session.
    Used by ML Jobs when a single job executes all three stages.
    """
    results = [
        feature_engineering_main(session),
        model_training_main(session),
        inference_main(session),
    ]
    for r in results:
        logger.info(r)
    return "Pipeline complete: " + " | ".join(results)
