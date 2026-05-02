"""
tests/test_models.py
TDD-first tests for app/models.py (Student SQLAlchemy model)
Written by Forge QAAgent BEFORE implementation files exist.
"""
import pytest

try:
    from app.models import Student
    from app.database import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


@pytest.fixture()
def db():
    if not HAS_MODELS:
        pytest.skip("models not yet implemented")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def test_student_model_has_required_fields():
    if not HAS_MODELS:
        pytest.skip("models not yet implemented")
    student = Student.__table__.columns
    column_names = [c.name for c in student]
    assert "id" in column_names
    assert "name" in column_names
    assert "email" in column_names


def test_student_can_be_created_and_queried(db):
    if not HAS_MODELS:
        pytest.skip("models not yet implemented")
    student = Student(name="Alice Smith", email="alice@example.com", grade="A")
    db.add(student)
    db.commit()
    db.refresh(student)
    assert student.id is not None
    fetched = db.query(Student).filter(Student.email == "alice@example.com").first()
    assert fetched is not None
    assert fetched.name == "Alice Smith"


def test_student_email_is_unique(db):
    if not HAS_MODELS:
        pytest.skip("models not yet implemented")
    from sqlalchemy.exc import IntegrityError
    s1 = Student(name="Bob", email="duplicate@example.com", grade="B")
    s2 = Student(name="Carol", email="duplicate@example.com", grade="B")
    db.add(s1)
    db.commit()
    db.add(s2)
    with pytest.raises(IntegrityError):
        db.commit()


def test_student_deletion(db):
    if not HAS_MODELS:
        pytest.skip("models not yet implemented")
    student = Student(name="Delete Me", email="deleteme@example.com", grade="C")
    db.add(student)
    db.commit()
    student_id = student.id
    db.delete(student)
    db.commit()
    assert db.query(Student).filter(Student.id == student_id).first() is None
