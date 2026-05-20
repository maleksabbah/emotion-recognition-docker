-- ============================================================================
-- Mntis — initialize per-service databases on first Postgres boot
-- Mounted by docker-compose into:
--   /docker-entrypoint-initdb.d/01-init-databases.sql
-- ============================================================================

CREATE DATABASE gateway_db;
CREATE DATABASE orchestrator_db;
CREATE DATABASE storage_db;