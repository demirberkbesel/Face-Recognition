import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Identity(Base):
    __tablename__ = "identities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(20), nullable=False)
    name = Column(String(255), nullable=True)
    extra_data = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    face_embeddings = relationship("FaceEmbedding", back_populates="identity", cascade="all, delete-orphan")
    process_faces = relationship("ProcessFace", back_populates="identity", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identity_id = Column(UUID(as_uuid=True), ForeignKey("identities.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    identity = relationship("Identity", back_populates="face_embeddings")


class Process(Base):
    __tablename__ = "processes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow)
    task_details = Column(JSONB, nullable=False)

    process_faces = relationship("ProcessFace", back_populates="process", cascade="all, delete-orphan")


class ProcessFace(Base):
    __tablename__ = "process_faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    process_id = Column(UUID(as_uuid=True), ForeignKey("processes.id", ondelete="CASCADE"), nullable=False)
    identity_id = Column(UUID(as_uuid=True), ForeignKey("identities.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    process = relationship("Process", back_populates="process_faces")
    identity = relationship("Identity", back_populates="process_faces")
