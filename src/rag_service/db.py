from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_service.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton-style manager for the async engine and session factory."""

    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @classmethod
    def _initialize(cls) -> None:
        """Create the engine and session factory if they don't exist."""
        if cls._engine is not None:
            return

        database_url = cls._get_database_url()

        pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

        logger.info(
            "Creating database engine with pool_size=%s max_overflow=%s",
            pool_size,
            max_overflow,
        )

        cls._engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        cls._session_factory = async_sessionmaker(
            bind=cls._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info(
            "Initialized database engine for db='%s' host='%s'",
            cls._engine.url.database,
            cls._engine.url.host,
        )

    @classmethod
    def _get_database_url(cls) -> str:
        """
        Resolve the database URL from environment variables, preferring DATABASE_URL.
        Falls back to composing from POSTGRES_* variables or the settings default.
        """
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url

        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        db = os.getenv("POSTGRES_DB")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")

        if user and password and db:
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

        if settings.database_url:
            return settings.database_url

        raise ValueError(
            "Database configuration not found. Set DATABASE_URL or POSTGRES_* environment variables."
        )

    @classmethod
    def get_session_factory(cls) -> async_sessionmaker[AsyncSession]:
        """Return the shared session factory, initializing if needed."""
        cls._initialize()
        if not cls._session_factory:
            raise RuntimeError("Session factory could not be created.")
        return cls._session_factory

    @classmethod
    async def get_session(cls) -> AsyncSession:
        """Convenience helper to get a single session."""
        factory = cls.get_session_factory()
        return factory()

    @classmethod
    async def close_engine(cls) -> None:
        """Dispose the engine and reset the manager."""
        if cls._engine:
            logger.info("Closing database engine")
            await cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None
            logger.info("Database engine closed")


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a session for dependency injection."""
    session = await DatabaseManager.get_session()
    try:
        yield session
    finally:
        await session.close()
