import uuid
from typing import Any, Optional

from sqlalchemy import Dialect, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql.type_api import _T, TypeEngine
from sqlalchemy.types import TypeDecorator


Base = declarative_base()


class UUIDType(TypeDecorator):
    """
    Custom UUID type for handling UUIDs in databases that don't natively support UUID.
    """

    impl = UUID

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(value))

    def process_result_value(
        self, value: Optional[Any], dialect: Dialect
    ) -> Optional[_T]:
        if value is None:
            return None
        return uuid.UUID(value)

