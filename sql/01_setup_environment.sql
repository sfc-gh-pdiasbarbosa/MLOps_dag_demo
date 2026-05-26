-- =============================================================================
-- 01_setup_environment.sql
-- Run once per environment to provision the required Snowflake objects.
--
-- Usage (Snow CLI):
--   snow sql -c <connection> -f sql/01_setup_environment.sql \
--            -D env_prefix=DEV -D wh_size=X-SMALL
--
-- Parameters (passed with -D key=value):
--   env_prefix  : e.g. DEV, SIT, UAT, PRD
--   wh_size     : warehouse size, e.g. X-SMALL, SMALL, MEDIUM, LARGE
-- =============================================================================

-- Warehouse
CREATE WAREHOUSE IF NOT EXISTS &{env_prefix}_ML_WH
    WAREHOUSE_SIZE = '&{wh_size}'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE
    COMMENT        = 'ML pipeline warehouse – &{env_prefix}';

-- Databases
CREATE DATABASE IF NOT EXISTS &{env_prefix}_ML_DB
    COMMENT = 'ML platform database – &{env_prefix}';

CREATE DATABASE IF NOT EXISTS &{env_prefix}_RAW_DB
    COMMENT = 'Raw data source – &{env_prefix}';

-- Schemas inside ML DB
CREATE SCHEMA IF NOT EXISTS &{env_prefix}_ML_DB.PIPELINES
    COMMENT = 'Stores DAG tasks, stored procedures and code stages';

CREATE SCHEMA IF NOT EXISTS &{env_prefix}_ML_DB.FEATURES
    COMMENT = 'Feature Store feature views (Dynamic Tables)';

CREATE SCHEMA IF NOT EXISTS &{env_prefix}_ML_DB.OUTPUT
    COMMENT = 'Inference output tables';

-- Schema inside raw DB
CREATE SCHEMA IF NOT EXISTS &{env_prefix}_RAW_DB.PUBLIC;

-- Internal stage for pipeline code
CREATE STAGE IF NOT EXISTS &{env_prefix}_ML_DB.PIPELINES.ML_CODE_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT    = 'Holds uploaded Python source files for stored procedures';

-- Placeholder raw data table (replace with your actual source)
CREATE TABLE IF NOT EXISTS &{env_prefix}_RAW_DB.PUBLIC.CUSTOMERS (
    CUSTOMER_ID   NUMBER        NOT NULL,
    TENURE        NUMBER,
    MONTHLY_CHARGES FLOAT,
    TOTAL_CHARGES   FLOAT,
    CONTRACT_TYPE   VARCHAR(50),
    TARGET_LABEL    NUMBER        -- 1 = churned, 0 = retained
);
