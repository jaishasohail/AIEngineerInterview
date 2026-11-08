"""Store dense vectors in pgvector on PostgreSQL and JSON on SQLite."""

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator


class PortableVector(TypeDecorator):
    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):  # noqa: ANN001
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return Vector(self.dimensions)
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        vector = [float(item) for item in value]
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, received {len(vector)}")
        return vector

    def process_result_value(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        if isinstance(value, str):
            import json

            value = json.loads(value)
        return [float(item) for item in list(value)]
