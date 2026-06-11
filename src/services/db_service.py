"""Asynchronous PostgreSQL data access via asyncpg + pgvector."""

import json
import logging
import os
from typing import Any, Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


class DatabaseService:
    """Async connection pool for deal-sourcing PostgreSQL with pgvector support."""

    def __init__(self) -> None:
        self._host = os.getenv("DB_HOST", "db")
        self._port = int(os.getenv("DB_PORT", "5432"))
        self._user = os.getenv("DB_USER", "postgres")
        self._password = os.getenv("DB_PASSWORD", "postgrespassword")
        self._database = os.getenv("DB_NAME", "lead_gen")
        self._pool: Optional[asyncpg.Pool] = None

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Register pgvector on each new pool connection (3072-dim embeddings)."""
        await register_vector(conn)

    async def init_pool(self) -> None:
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                user=self._user,
                password=self._password,
                database=self._database,
                host=self._host,
                port=self._port,
                init=self._init_connection,
                ssl=False,              # ← fixes the SSL crash
            )
            logger.info("Database connection pool ready")
        except Exception:
            logger.exception("Failed to initialize database connection pool")
            raise

    async def close_pool(self) -> None:
        """Close the connection pool gracefully."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None
        logger.info("Database connection pool closed")

    async def fetch_all_leads(self) -> list[dict[str, Any]]:
        """Return all rows from target_entities (embeddings omitted for payload size)."""
        if self._pool is None:
            raise RuntimeError("Database pool not initialized; call init_pool() first")

        query = """
            SELECT id, url, company_name, primary_contact, all_contacts,
                   raw_content, scraped_at
            FROM target_entities
            ORDER BY scraped_at DESC NULLS LAST, id DESC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)

        def _parse_jsonb(value: Any) -> Any:
            if value is None or isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                return json.loads(value)
            return value

        return [
            {
                "id": row["id"],
                "url": row["url"],
                "company_name": row["company_name"],
                "primary_contact": _parse_jsonb(row["primary_contact"]),
                "all_contacts": _parse_jsonb(row["all_contacts"]),
                "raw_content": row["raw_content"],
                "scraped_at": row["scraped_at"],
            }
            for row in rows
        ]

    async def insert_target_entity(
        self,
        url: str,
        company_name: str,
        raw_content: str,
        embedding: Optional[list[float]] = None,
        primary_contact: Optional[dict[str, Any]] = None,
        all_contacts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Upsert a target entity row keyed by URL.

        Uses parameterized queries and ON CONFLICT (url) DO UPDATE for idempotent pipeline runs.
        embedding may be None when the orchestrator bypasses web crawl (direct_to_enrichment).
        primary_contact and all_contacts are stored as JSONB from Hunter enrichment.
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized; call init_pool() first")

        query = """
            INSERT INTO target_entities (
                url, company_name, primary_contact, all_contacts, raw_content, embedding
            )
            VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
            ON CONFLICT (url) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                primary_contact = EXCLUDED.primary_contact,
                all_contacts = EXCLUDED.all_contacts,
                raw_content = EXCLUDED.raw_content,
                embedding = EXCLUDED.embedding,
                scraped_at = CURRENT_TIMESTAMP
        """

        primary_json = json.dumps(primary_contact) if primary_contact is not None else None
        all_contacts_json = json.dumps(all_contacts) if all_contacts is not None else None

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    url,
                    company_name,
                    primary_json,
                    all_contacts_json,
                    raw_content,
                    embedding,
                )
            logger.info("Inserted/updated target entity url=%s company=%s", url, company_name)
        except Exception:
            logger.exception(
                "Failed to insert target entity url=%s company=%s",
                url,
                company_name,
            )
            raise
