"""用户持仓的 ORM 模型。"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, DECIMAL, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class Portfolio(Base):
    """用户的股票或基金持仓。"""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    shares: Mapped[Decimal] = mapped_column(DECIMAL(20, 6), nullable=False)
    cost_price: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), nullable=False)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
