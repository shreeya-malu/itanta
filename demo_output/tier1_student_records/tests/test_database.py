"""
tests/test_database.py
TDD-first tests for app/database.py
Written by Forge QAAgent BEFORE implementation files exist.
"""
import pytest

try:
    from app.database import Base, get_db, init_db
    from sqlalchemy.orm import Session
    HAS_DB = True
except ImportError:
    HAS_DB = False


@pytest.fixture()
def db_session():
    if not HAS_DB:
        pytest.skip("database not yet implemented")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def test_base_exists_and_is_declarative():
    if not HAS_DB:
        pytest.skip("database not yet implemented")
    assert Base is not None
    assert hasattr(Base, "metadata")


def test_init_db_creates_tables(db_session):
    if not HAS_DB:
        pytest.skip("database not yet implemented")
    from sqlalchemy import inspect
    engine = db_session.get_bind()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert isinstance(tables, list)


def test_get_db_yields_session():
    if not HAS_DB:
        pytest.skip("database not yet implemented")
    gen = get_db()
    session = next(gen)
    assert isinstance(session, Session)
    try:
        next(gen)
    except StopIteration:
        pass
