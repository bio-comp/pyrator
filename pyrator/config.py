# pyrator/config.py
"""Configuration management for the pyrator library.

This module defines the configuration structure for pyrator using Pydantic,
allowing for settings to be loaded from environment variables, .env files,
and secrets directories. It centralizes settings for computation, Bayesian
inference, and data backends.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import ClassVar, Literal, Self

from loguru import logger
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ComputeSettings(BaseSettings):
    """Manages computation and performance settings.

    Attributes:
        n_jobs: Number of parallel jobs to use. Defaults to -1, which utilizes
            all available CPU cores.
        chunk_size: The batch size for processing large datasets in chunks.
        backend: The numerical computation backend to use.
        max_memory_gb: An optional limit on the maximum memory usage in GB.
        use_memmap: If True, uses memory-mapped arrays for large datasets to
            reduce RAM usage at the cost of slower disk I/O.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_COMPUTE_",
        case_sensitive=False,
    )

    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs (-1 = all available CPUs).",
        json_schema_extra={"example": 4},
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Batch size for processing large datasets.",
    )
    backend: Literal["numpy", "numba", "jax"] = Field(
        default="numpy",
        description="The numerical computation backend.",
    )
    max_memory_gb: float | None = Field(
        default=None,
        gt=0,
        description="Optional limit on maximum memory usage in GB.",
    )
    use_memmap: bool = Field(
        default=False,
        description="Use memory-mapped arrays for large datasets.",
    )

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int, info: ValidationInfo) -> int:
        """Validates and resolves the number of parallel jobs.

        If n_jobs is -1, it is resolved to the total number of available CPU
        cores. The final value is capped at the total number of CPUs.

        Args:
            v: The user-provided value for n_jobs.
            info: Pydantic validation info (unused).

        Returns:
            The validated and resolved number of jobs.

        Raises:
            ValueError: If n_jobs is less than 1 and not -1.
        """
        cpu_count = multiprocessing.cpu_count()
        if v == -1:
            return cpu_count
        if v < 1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        return min(v, cpu_count)


class OntologySettings(BaseSettings):
    """Manages settings for ontology processing and semantic metrics.

    Attributes:
        lca_policy: The tie-breaking policy for finding the Least Common
            Ancestor (LCA) in a DAG.
        path_policy: The policy for calculating path distance in a DAG.
        smoothing_alpha: The smoothing parameter (pseudocount) for calculating
            Information Content (IC).
        path_distance_normalization: The scope for normalizing path distance.
            'observed' uses the max distance found in the current dataset, while
            'global' would scan the entire ontology (more expensive).
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_ONTOLOGY_", case_sensitive=False
    )
    lca_policy: Literal["max_ic", "max_depth"] = "max_ic"
    path_policy: Literal["undirected_tr", "spanning_tree"] = "undirected_tr"
    smoothing_alpha: float = Field(default=1.0, gt=0)
    path_distance_normalization: Literal["observed", "global"] = "observed"


class AgreementSettings(BaseSettings):
    """Manages settings for Inter-Annotator Agreement calculations.

    Attributes:
        abstain_policy: How to treat `ABSTAIN` labels in Kappa calculations.
            'ignore' drops items with abstentions, 'penalize' treats them as
            a distinct category (a full disagreement). Krippendorff's alpha
            always treats them as missing data.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_AGREEMENT_", case_sensitive=False
    )
    abstain_policy: Literal["ignore", "penalize"] = "ignore"


class BayesianPriorSettings(BaseSettings):
    """Defines hyperparameters for the priors in Bayesian models."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_BAYESIAN__PRIORS_", case_sensitive=False
    )
    reliability_tau: float = Field(
        default=1.0, gt=0, description="Scale for HalfNormal prior on annotator reliability (τ)."
    )
    difficulty_delta: float = Field(
        default=1.0, gt=0, description="Scale for HalfNormal prior on item difficulty (δ)."
    )
    bias_beta: float = Field(
        default=1.0, gt=0, description="Scale for Normal prior on annotator specificity bias (β)."
    )
    hybrid_lambda_a: float = Field(
        default=2.0,
        gt=0,
        description="Alpha parameter for Beta prior on the hybrid specificity metric.",
    )
    hybrid_lambda_b: float = Field(
        default=2.0,
        gt=0,
        description="Beta parameter for Beta prior on the hybrid specificity metric.",
    )


class BayesianSettings(BaseSettings):
    """Manages settings for Bayesian inference and MCMC sampling.

    Attributes:
        sampler: The MCMC sampler to use for inference.
        chains: The number of parallel MCMC chains to run.
        draws: The number of posterior samples to generate per chain.
        tune: The number of tuning (warmup) steps per chain.
        target_accept: The target acceptance probability for NUTS samplers.
        check_convergence: If True, automatically checks for convergence using
            R-hat and ESS thresholds.
        r_hat_threshold: The acceptable threshold for the R-hat statistic.
        ess_threshold: The acceptable threshold for the effective sample size.
        use_gpu: If True, attempts to use a GPU for accelerated computation.
        gpu_device: The specific GPU device index to use.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
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
    pruning_k: int = Field(
        default=5, ge=1, description="Number of nearest neighbors for pruned candidate sets."
    )
    priors: BayesianPriorSettings = BayesianPriorSettings()

    @model_validator(mode="after")
    def validate_gpu_settings(self) -> Self:
        """Validates GPU availability and device settings.

        Checks if JAX is installed and if a GPU is available when `use_gpu`
        is requested. If not, it gracefully falls back to CPU and issues a
        warning.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If the specified `gpu_device` index is out of bounds.
        """
        if self.use_gpu:
            try:
                import jax

                devices = jax.devices("gpu")
                if not devices:
                    logger.warning("GPU requested but not available. Falling back to CPU.")
                    self.use_gpu = False
                elif self.gpu_device is not None and self.gpu_device >= len(devices):
                    raise ValueError(
                        f"GPU device index {self.gpu_device} is not available. "
                        f"Found {len(devices)} devices."
                    )
            except (ImportError, RuntimeError):
                logger.warning(
                    "JAX not installed or GPU unavailable. GPU acceleration is unavailable."
                )
                self.use_gpu = False
        return self


class DatabaseSettings(BaseSettings):
    """Manages settings for data backends and database connections.

    Attributes:
        engine: The data processing engine to use.
        connection_string: The database connection string. For in-memory
            SQLite or DuckDB, this can be omitted.
        duckdb_memory_limit: Memory limit for DuckDB instances.
        duckdb_threads: Number of threads for DuckDB. Defaults to DuckDB's internal
            heuristic if None.
        duckdb_temp_directory: Directory for DuckDB to spill temporary data.
        enable_sql_validation: If True, performs basic validation on SQL queries.
        allow_raw_sql: If True, allows execution of raw, unvalidated SQL strings.
            Use with caution.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_DB_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engine: Literal["duckdb", "sqlite", "pandas", "polars"] = "pandas"
    connection_string: str | None = None

    # DuckDB specific settings
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int | None = None
    duckdb_temp_directory: Path | None = None

    # Safety features
    enable_sql_validation: bool = True
    allow_raw_sql: bool = False

    @field_validator("connection_string")
    @classmethod
    def validate_connection(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Sets a default in-memory connection for SQLite and DuckDB.

        Args:
            v: The user-provided connection string.
            info: Pydantic validation info containing other model data.

        Returns:
            The validated connection string, or ":memory:" as a default.
        """
        if info.data.get("engine") in ("duckdb", "sqlite") and v is None:
            return ":memory:"
        return v


class PyratorSettings(BaseSettings):
    """Defines the main configuration structure for the pyrator library.

    This class aggregates all other settings classes and provides global
    application-level configurations. It defines the priority for loading
    settings from various sources like environment variables and .env files.

    Attributes:
        compute: Nested settings for computation and performance.
        bayesian: Nested settings for Bayesian inference.
        database: Nested settings for data backends.
        random_seed: An optional seed for all random number generators to
            ensure reproducibility.
        debug: If True, enables debug-level logging and other diagnostics.
        profile: If True, enables performance profiling hooks.
        cache_dir: The directory for storing cached data and artifacts.
        log_dir: The directory for storing log files.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="PYRATOR_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",
        secrets_dir="/run/secrets",
    )

    # Nested settings sections
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    bayesian: BayesianSettings = Field(default_factory=BayesianSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Global settings
    random_seed: int | None = Field(default=None, ge=0, description="Seed for reproducibility.")
    debug: bool = Field(default=False, description="Enable debug mode.")
    profile: bool = Field(default=False, description="Enable performance profiling.")

    # Path configurations
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".pyrator" / "cache")
    log_dir: Path = Field(default_factory=lambda: Path.home() / ".pyrator" / "logs")

    @field_validator("cache_dir", "log_dir")
    @classmethod
    def create_directories(cls, v: Path, info: ValidationInfo) -> Path:
        """Ensures that the cache and log directories exist.

        Args:
            v: The path to create.
            info: Pydantic validation info (unused).

        Returns:
            The validated path.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Defines the priority of configuration sources.

        Settings are loaded in the specified order, with later sources
        overriding earlier ones. The priority is:
        1. Direct initialization arguments.
        2. Environment variables.
        3. Values from .env files.
        4. Values from files in a secrets directory (e.g., Docker Secrets).

        Args:
            settings_cls: The settings class being initialized.
            init_settings: Settings from constructor arguments.
            env_settings: Settings from environment variables.
            dotenv_settings: Settings from .env files.
            file_secret_settings: Settings from secret files.

        Returns:
            A tuple of the setting sources in their desired order of priority.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


# Example of how to instantiate and use the settings
# if __name__ == "__main__":
#     config = PyratorSettings()
#     print("--- Pyrator Configuration ---")
#     print(f"Debug mode: {config.debug}")
#     print(f"Random seed: {config.random_seed}")
#     print("\n[Compute Settings]")
#     print(f"  Parallel jobs: {config.compute.n_jobs}")
#     print(f"  Backend: {config.compute.backend}")
#     print("\n[Bayesian Settings]")
#     print(f"  Sampler: {config.bayesian.sampler}")
#     print(f"  Chains: {config.bayesian.chains}")
#     print(f"  Use GPU: {config.bayesian.use_gpu}")
#     print("\n[Database Settings]")
#     print(f"  Engine: {config.database.engine}")
#     print(f"  Connection: {config.database.connection_string}")
