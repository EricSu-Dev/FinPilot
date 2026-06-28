"""从环境变量加载的应用配置。"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """FinPilot 后端的运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API key.")
    MODEL_NAME: str = Field(default="deepseek-v4-flash", description="Default LLM model name.")
    TAVILY_API_KEY: str = Field(default="", description="Tavily 搜索 API key，用作 ddgs 失败时的兜底。")
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="JWT 签名密钥。生产环境务必从 .env 读取一个足够随机的值。",
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT 签名算法。")
    JWT_EXPIRE_MINUTES: int = Field(default=1440, description="JWT token 过期时间（分钟），默认 24 小时。")
    MYSQL_HOST: str = Field(default="localhost", description="MySQL host.")
    MYSQL_PORT: int = Field(default=3306, description="MySQL port.")
    MYSQL_USER: str = Field(default="root", description="MySQL username.")
    MYSQL_PASSWORD: str = Field(default="", description="MySQL password.")
    MYSQL_DATABASE: str = Field(default="finpilot", description="MySQL database name.")
    DASHSCOPE_API_KEY: str = Field(default="", description="阿里百炼 API key，用于财报 RAG 的文本嵌入。")

    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Local directory used by Chroma vector store.",
    )

    @property
    def mysql_url(self) -> str:
        """构建 SQLAlchemy 的 MySQL 连接 URL。"""
        user = quote_plus(self.MYSQL_USER)
        password = quote_plus(self.MYSQL_PASSWORD)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    """返回缓存的配置，使所有模块共享同一个配置实例。"""
    return Settings()
