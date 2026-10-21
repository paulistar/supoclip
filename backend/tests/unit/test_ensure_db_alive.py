"""Regression: long ffmpeg waves must not leave a dead AsyncSession attached."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import InterfaceError

from src.database import ensure_db_alive


@pytest.mark.asyncio
async def test_ensure_db_alive_noops_when_ping_succeeds():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock())
    db.info = {}
    db.commit = AsyncMock()
    await ensure_db_alive(db)
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_db_alive_skips_commit_while_editing():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock())
    db.info = {"supoclip_edit_task": "task-1"}
    db.commit = AsyncMock()
    await ensure_db_alive(db)
    db.execute.assert_awaited_once()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_db_alive_recovers_closed_connection():
    db = AsyncMock()
    closed = InterfaceError(
        "connection",
        None,
        Exception("cannot call PreparedStatement.fetch(): the underlying connection is closed"),
    )
    db.execute = AsyncMock(side_effect=[closed, MagicMock()])
    db.rollback = AsyncMock()
    db.info = {}
    db.commit = AsyncMock()
    conn = AsyncMock()
    db.connection = AsyncMock(return_value=conn)

    await ensure_db_alive(db)

    assert db.execute.await_count == 2
    db.rollback.assert_awaited()
    conn.invalidate.assert_awaited_once()
    db.commit.assert_awaited_once()
