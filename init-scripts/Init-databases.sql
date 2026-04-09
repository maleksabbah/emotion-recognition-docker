-- ============================================================================
-- Creates the three databases needed by the platform.
-- Runs automatically on first PostgreSQL startup via docker-entrypoint-initdb.d
-- ============================================================================

-- orchestrator_db is created by POSTGRES_DB env var (default db)
-- We just need to create the other two:

CREATE DATABASE gateway_db;
CREATE DATABASE storage_db;

-- Grant full access to the emotion user on all databases
GRANT ALL PRIVILEGES ON DATABASE orchestrator_db TO emotion;
GRANT ALL PRIVILEGES ON DATABASE gateway_db TO emotion;
GRANT ALL PRIVILEGES ON DATABASE storage_db TO emotion;