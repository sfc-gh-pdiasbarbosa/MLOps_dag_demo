-- =============================================================================
-- 02_setup_compute_pool.sql
-- Only required when deploying with --mode mljobs.
--
-- Usage (Snow CLI):
--   snow sql -c <connection> -f sql/02_setup_compute_pool.sql \
--            -D env_prefix=DEV
-- =============================================================================

CREATE COMPUTE POOL IF NOT EXISTS &{env_prefix}_ML_COMPUTE_POOL
    MIN_NODES        = 1
    MAX_NODES        = 3
    INSTANCE_FAMILY  = CPU_X64_S
    AUTO_RESUME      = TRUE
    AUTO_SUSPEND_SECS = 300
    COMMENT          = 'Compute pool for ML Jobs – &{env_prefix}';

-- Grant usage to the role that runs the pipeline
-- GRANT USAGE ON COMPUTE POOL &{env_prefix}_ML_COMPUTE_POOL TO ROLE <your_ml_role>;
