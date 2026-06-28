"""数据库引擎与会话管理。"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """所有 SQLAlchemy ORM 模型的基类。"""


settings = get_settings()

engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,
    # 远程 MySQL 下，连接池里的连接在长 agent 运行期间会空闲几分钟，容易被
    # 中间 NAT/防火墙悄悄掐断。把 pool_recycle 从 1h 调到 5min，主动回收老连接，
    # 减少拿到半开连接的概率。
    pool_recycle=300,
    # connect_timeout：连不上时快速失败（默认会卡很久）。
    # read/write_timeout：半开连接（TCP 被 NAT 静默掐断）读写时快速失败，
    # 让 pool_pre_ping 能及时识别并换新连接，而不是卡死。
    # ssl_disabled：服务端开了 SSL 能力，pymysql 默认 SSL 握手在 Python 3.14 +
    # 该服务端配置下会卡死超时；开发环境不强制 SSL，禁用绕开。
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 10,
        "write_timeout": 10,
        "ssl_disabled": True,
    },
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """为 FastAPI 依赖注入产出一个数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
