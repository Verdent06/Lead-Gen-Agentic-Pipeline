"""Asynchronous PostgreSQL data access via asyncpg + pgvector."""

import logging
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


class DatabaseService:
    """Async connection pool for deal-sourcing PostgreSQL with pgvector support."""

    def __init__(
        self,
        *,
        user: str = "vedant",
        password: str = "rootpassword",
        database: str = "deal_sourcing_db",
        host: str = "localhost",
        port: int = 5432,
    ) -> None:
        self._user = user
        self._password = password
        self._database = database
        self._host = host
        self._port = port
        self._pool: Optional[asyncpg.Pool] = None

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Register pgvector on each new pool connection (3072-dim embeddings)."""
        await register_vector(conn)

    async def init_pool(self) -> None:
        """Create the async connection pool with pgvector registered on every connection."""
        if self._pool is not None:
            logger.debug("Database pool already initialized")
            return

        logger.info(
            "Initializing asyncpg pool host=%s port=%s db=%s user=%s",
            self._host,
            self._port,
            self._database,
            self._user,
        )
        try:
            self._pool = await asyncpg.create_pool(
                user=self._user,
                password=self._password,
                database=self._database,
                host=self._host,
                port=self._port,
                init=self._init_connection,
            )
            logger.info("Database connection pool ready")
        except Exception:
            logger.exception("Failed to initialize database connection pool")
            raise

    async def insert_target_entity(
        self,
        url: str,
        company_name: str,
        raw_content: str,
        embedding: list[float],
    ) -> None:
        """
        Upsert a target entity row keyed by URL.

        Uses parameterized queries and ON CONFLICT (url) DO UPDATE for idempotent pipeline runs.
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized; call init_pool() first")

        query = """
            INSERT INTO target_entities (url, company_name, raw_content, embedding)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (url) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                raw_content = EXCLUDED.raw_content,
                embedding = EXCLUDED.embedding,
                scraped_at = CURRENT_TIMESTAMP
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, url, company_name, raw_content, embedding)
            logger.info("Inserted/updated target entity url=%s company=%s", url, company_name)
        except Exception:
            logger.exception(
                "Failed to insert target entity url=%s company=%s",
                url,
                company_name,
            )
            raise
