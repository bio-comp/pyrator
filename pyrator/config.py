# pyrator/config.py
from typing import Literal
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import multiprocessing

class ComputeSettings(BaseSettings):
    """Computation and performance settings"""
    model_config = SettingsConfigDict(
        env_prefix="PYRATOR_COMPUTE_",
        case_sensitive=False,
    )

    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs (-1 = all CPUs)",
        json_schema_extra={"example": 4}
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Batch size for processing large datasets"
    )
    backend: Literal["numpy", "numba", "jax"] = Field(
        default="numpy",
        description="Computation backend"
    )
    max_memory_gb: float | None = Field(
        default=None,
        gt=0,
        description="Maximum memory usage in GB"
    )
    use_memmap: bool = Field(
        default=False,
        description="Use memory-mapped arrays for large datasets"
    )

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        if v == -1:
            return multiprocessing.cpu_count()
        elif v < 1:
            raise ValueError("n_jobs must be -1 or positive")
        return min(v, multiprocessing.cpu_count())

class BayesianSettings(BaseSettings):
    """Bayesian inference settings"""
    model_config = SettingsConfigDict(
        env_prefix="PYRATOR_BAYESIAN_",
        case_sensitive=False,
    )

    sampler: Literal["nuts", "metropolis", "numpyro", "blackjax"] = "nuts"
    chains: int = Field(default=4, ge=1, le=16)
    draws: int = Field(default=2000, ge=100)
    tune: int = Field(default=1000, ge=100)
    target_accept: float = Field(default=0.8, ge=0.5, le=0.99)

    # Convergence criteria
    check_convergence: bool = True
    r_hat_threshold: float = Field(default=1.01, gt=1.0)
    ess_threshold: int = Field(default=400, ge=100)

    # GPU settings
    use_gpu: bool = False
    gpu_device: int | None = None


    @model_validator(mode='after')
    def validate_gpu_settings(self):
        """Check GPU availability if use_gpu is True"""
        if self.use_gpu:
            try:
                import jax
                devices = jax.devices('gpu')
                if not devices:
                    print("Warning: GPU requested but not available, falling back to CPU")
                    self.use_gpu = False
                elif self.gpu_device is not None and self.gpu_device >= len(devices):
                    raise ValueError(f"GPU device {self.gpu_device} not available")
            except ImportError:
                print("Warning: JAX not installed, GPU acceleration unavailable")
                self.use_gpu = False
        return self

class DatabaseSettings(BaseSettings):
    """Database backend settings"""
    model_config = SettingsConfigDict(
        env_prefix="PYRATOR_DB_",
        case_sensitive=False,
        # Allow loading from .env files
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields
    )

    engine: Literal["duckdb", "sqlite", "pandas", "polars"] = "pandas"
    connection_string: Optional[str] = None

    # DuckDB specific
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: Optional[int] = None
    duckdb_temp_directory: Optional[Path] = None

    # Safety
    enable_sql_validation: bool = True
    allow_raw_sql: bool = False

    @field_validator("connection_string")
    @classmethod
    def validate_connection(cls, v: Optional[str], info) -> str:
        if info.data.get("engine") in ["duckdb", "sqlite"] and v is None:
            return ":memory:"
        return v

class PyratorSettings(BaseSettings):
    """Main configuration for Pyrator"""

    model_config = SettingsConfigDict(
        # Primary env prefix
        env_prefix="PYRATOR_",

        # Support for .env files
        env_file=".env",
        env_file_encoding="utf-8",

        # Support for multiple env files (in order of priority)
        env_file=(".env", ".env.local", ".env.prod"),

        # Allow nested env vars with double underscore
        env_nested_delimiter="__",

        # Case insensitive env vars
        case_sensitive=False,

        # Validate on assignment
        validate_assignment=True,

        # Allow extra fields (forward compatibility)
        extra="allow",

        # Support secrets directory (for Docker/K8s)
        secrets_dir="/run/secrets",
    )

    # Nested settings
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    bayesian: BayesianSettings = Field(default_factory=BayesianSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Global settings
    random_seed: Optional[int] = Field(default=None, ge=0)
    debug: bool = Field(default=False)
    profile: bool = Field(default=False)

    # Paths
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".pyrator" / "cache"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".pyrator" / "logs"
    )

    @field_validator("cache_dir", "log_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Define the priority of settings sources.
        Higher priority sources override lower priority ones.
        """
        return (
            init_settings,      # 1. Arguments passed to constructor
            env_settings,       # 2. Environment variables
            dotenv_settings,    # 3. .env file
            file_secret_settings,  # 4. Secret files (Docker/K8s)
        )
