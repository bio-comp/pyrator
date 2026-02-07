"""Tests for configuration management."""

import multiprocessing
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pyrator.config import (
    AgreementSettings,
    BayesianPriorSettings,
    BayesianSettings,
    ComputeSettings,
    DatabaseSettings,
    OntologySettings,
    PyratorSettings,
)


class TestComputeSettings:
    """Test ComputeSettings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = ComputeSettings()
        assert settings.n_jobs == multiprocessing.cpu_count()
        assert settings.chunk_size == 1000
        assert settings.backend == "numpy"
        assert settings.max_memory_gb is None
        assert settings.use_memmap is False

    def test_n_jobs_validation(self):
        """Test n_jobs validation logic."""
        settings = ComputeSettings(n_jobs=2)
        assert settings.n_jobs == 2

        # Test -1 resolves to CPU count
        settings = ComputeSettings(n_jobs=-1)
        assert settings.n_jobs == multiprocessing.cpu_count()

        # Test invalid values
        with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
            ComputeSettings(n_jobs=0)

        with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
            ComputeSettings(n_jobs=-2)

    def test_n_jobs_capping(self):
        """Test that n_jobs is capped at available CPU count."""
        cpu_count = multiprocessing.cpu_count()
        settings = ComputeSettings(n_jobs=cpu_count + 10)
        assert settings.n_jobs == cpu_count

    def test_backend_validation(self):
        """Test backend literal validation."""
        for backend in ["numpy", "numba", "jax"]:
            settings = ComputeSettings(backend=backend)
            assert settings.backend == backend

    def test_chunk_size_validation(self):
        """Test chunk_size validation."""
        settings = ComputeSettings(chunk_size=500)
        assert settings.chunk_size == 500

        with pytest.raises(ValueError):
            ComputeSettings(chunk_size=0)

    def test_max_memory_validation(self):
        """Test max_memory_gb validation."""
        settings = ComputeSettings(max_memory_gb=8.0)
        assert settings.max_memory_gb == 8.0

        with pytest.raises(ValueError):
            ComputeSettings(max_memory_gb=0)


class TestOntologySettings:
    """Test OntologySettings configuration."""

    def test_default_values(self):
        """Test default ontology settings."""
        settings = OntologySettings()
        assert settings.lca_policy == "max_ic"
        assert settings.path_policy == "undirected_tr"
        assert settings.smoothing_alpha == 1.0
        assert settings.path_distance_normalization == "observed"

    def test_lca_policy_validation(self):
        """Test LCA policy validation."""
        for policy in ["max_ic", "max_depth"]:
            settings = OntologySettings(lca_policy=policy)
            assert settings.lca_policy == policy

    def test_path_policy_validation(self):
        """Test path policy validation."""
        for policy in ["undirected_tr", "spanning_tree"]:
            settings = OntologySettings(path_policy=policy)
            assert settings.path_policy == policy

    def test_smoothing_alpha_validation(self):
        """Test smoothing alpha validation."""
        settings = OntologySettings(smoothing_alpha=2.0)
        assert settings.smoothing_alpha == 2.0

        with pytest.raises(ValueError):
            OntologySettings(smoothing_alpha=0)


class TestAgreementSettings:
    """Test AgreementSettings configuration."""

    def test_default_values(self):
        """Test default agreement settings."""
        settings = AgreementSettings()
        assert settings.abstain_policy == "ignore"

    def test_abstain_policy_validation(self):
        """Test abstain policy validation."""
        for policy in ["ignore", "penalize"]:
            settings = AgreementSettings(abstain_policy=policy)
            assert settings.abstain_policy == policy


class TestBayesianPriorSettings:
    """Test BayesianPriorSettings configuration."""

    def test_default_values(self):
        """Test default prior settings."""
        settings = BayesianPriorSettings()
        assert settings.reliability_tau == 1.0
        assert settings.difficulty_delta == 1.0
        assert settings.bias_beta == 1.0
        assert settings.hybrid_lambda_a == 2.0
        assert settings.hybrid_lambda_b == 2.0

    def test_prior_validation(self):
        """Test prior parameter validation."""
        settings = BayesianPriorSettings(
            reliability_tau=2.0,
            difficulty_delta=1.5,
            bias_beta=0.5,
            hybrid_lambda_a=3.0,
            hybrid_lambda_b=1.0,
        )
        assert settings.reliability_tau == 2.0
        assert settings.difficulty_delta == 1.5
        assert settings.bias_beta == 0.5
        assert settings.hybrid_lambda_a == 3.0
        assert settings.hybrid_lambda_b == 1.0

        with pytest.raises(ValueError):
            BayesianPriorSettings(reliability_tau=0)


class TestBayesianSettings:
    """Test BayesianSettings configuration."""

    def test_default_values(self):
        """Test default Bayesian settings."""
        settings = BayesianSettings()
        assert settings.sampler == "nuts"
        assert settings.chains == 4
        assert settings.draws == 2000
        assert settings.tune == 1000
        assert settings.target_accept == 0.8
        assert settings.check_convergence is True
        assert settings.r_hat_threshold == 1.01
        assert settings.ess_threshold == 400
        assert settings.use_gpu is False
        assert settings.gpu_device is None
        assert settings.pruning_k == 5

    def test_sampler_validation(self):
        """Test sampler validation."""
        for sampler in ["nuts", "metropolis", "numpyro", "blackjax"]:
            settings = BayesianSettings(sampler=sampler)
            assert settings.sampler == sampler

    def test_chain_validation(self):
        """Test chain count validation."""
        settings = BayesianSettings(chains=8)
        assert settings.chains == 8

        with pytest.raises(ValueError):
            BayesianSettings(chains=0)

        with pytest.raises(ValueError):
            BayesianSettings(chains=17)

    def test_draws_and_tune_validation(self):
        """Test draws and tune validation."""
        settings = BayesianSettings(draws=5000, tune=2000)
        assert settings.draws == 5000
        assert settings.tune == 2000

        with pytest.raises(ValueError):
            BayesianSettings(draws=50)

        with pytest.raises(ValueError):
            BayesianSettings(tune=50)

    def test_target_accept_validation(self):
        """Test target accept validation."""
        settings = BayesianSettings(target_accept=0.9)
        assert settings.target_accept == 0.9

        with pytest.raises(ValueError):
            BayesianSettings(target_accept=0.4)

        with pytest.raises(ValueError):
            BayesianSettings(target_accept=1.0)

    def test_convergence_validation(self):
        """Test convergence criteria validation."""
        settings = BayesianSettings(r_hat_threshold=1.05, ess_threshold=800)
        assert settings.r_hat_threshold == 1.05
        assert settings.ess_threshold == 800

        with pytest.raises(ValueError):
            BayesianSettings(r_hat_threshold=1.0)

        with pytest.raises(ValueError):
            BayesianSettings(ess_threshold=50)

    def test_gpu_settings_validation(self):
        """Test GPU settings validation."""
        # Test with JAX not available - should fallback to CPU
        settings = BayesianSettings(use_gpu=True, gpu_device=0, pruning_k=10)
        assert settings.use_gpu is False  # Falls back to CPU when JAX not available
        assert settings.gpu_device == 0
        assert settings.pruning_k == 10

        with pytest.raises(ValueError):
            BayesianSettings(pruning_k=0)

    def test_gpu_validation_with_jax(self):
        """Test GPU validation when JAX is available."""
        with patch.dict("sys.modules", {"jax": MagicMock()}):
            import jax

            jax.devices.return_value = ["gpu:0", "gpu:1"]
            jax.__version__ = "1.0.0"

            settings = BayesianSettings(use_gpu=True, gpu_device=1)
            assert settings.use_gpu is True
            assert settings.gpu_device == 1

    def test_gpu_validation_no_devices(self):
        """Test GPU validation when no GPU devices are available."""
        with patch.dict("sys.modules", {"jax": MagicMock()}):
            import jax

            jax.devices.return_value = []
            jax.__version__ = "1.0.0"

            settings = BayesianSettings(use_gpu=True)
            assert settings.use_gpu is False  # Should fallback to CPU

    def test_gpu_validation_invalid_device(self):
        """Test GPU validation with invalid device index."""
        with patch.dict("sys.modules", {"jax": MagicMock()}):
            import jax

            jax.devices.return_value = ["gpu:0"]
            jax.__version__ = "1.0.0"

            with pytest.raises(ValueError, match="GPU device index 1 is not available"):
                BayesianSettings(use_gpu=True, gpu_device=1)

    def test_gpu_validation_without_jax(self):
        """Test GPU validation when JAX is not installed."""
        with patch.dict("sys.modules", {"jax": None}):
            settings = BayesianSettings(use_gpu=True)
            assert settings.use_gpu is False


class TestDatabaseSettings:
    """Test DatabaseSettings configuration."""

    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()
        assert settings.engine == "pandas"
        assert settings.connection_string is None
        assert settings.duckdb_memory_limit == "4GB"
        assert settings.duckdb_threads is None
        assert settings.duckdb_temp_directory is None
        assert settings.enable_sql_validation is True
        assert settings.allow_raw_sql is False

    def test_engine_validation(self):
        """Test engine validation."""
        for engine in ["duckdb", "sqlite", "pandas", "polars"]:
            settings = DatabaseSettings(engine=engine)
            assert settings.engine == engine

    def test_connection_string_defaults(self):
        """Test connection string defaults for in-memory databases."""
        # DuckDB should default to :memory:
        settings = DatabaseSettings(engine="duckdb")
        assert settings.connection_string == ":memory:"

        # SQLite should default to :memory:
        settings = DatabaseSettings(engine="sqlite")
        assert settings.connection_string == ":memory:"

        # Pandas and Polars should remain None
        settings = DatabaseSettings(engine="pandas")
        assert settings.connection_string is None

        settings = DatabaseSettings(engine="polars")
        assert settings.connection_string is None

    def test_duckdb_settings(self):
        """Test DuckDB-specific settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = DatabaseSettings(
                engine="duckdb",
                duckdb_memory_limit="8GB",
                duckdb_threads=4,
                duckdb_temp_directory=Path(temp_dir),
            )
            assert settings.duckdb_memory_limit == "8GB"
            assert settings.duckdb_threads == 4
            assert settings.duckdb_temp_directory == Path(temp_dir)

    def test_safety_settings(self):
        """Test safety feature settings."""
        settings = DatabaseSettings(enable_sql_validation=False, allow_raw_sql=True)
        assert settings.enable_sql_validation is False
        assert settings.allow_raw_sql is True


class TestPyratorSettings:
    """Test main PyratorSettings configuration."""

    def test_default_values(self):
        """Test default main settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override default paths for testing
            settings = PyratorSettings(
                cache_dir=Path(temp_dir) / "cache", log_dir=Path(temp_dir) / "logs"
            )

            assert settings.random_seed is None
            assert settings.debug is False
            assert settings.profile is False
            assert settings.cache_dir == Path(temp_dir) / "cache"
            assert settings.log_dir == Path(temp_dir) / "logs"

            # Check nested settings
            assert isinstance(settings.compute, ComputeSettings)
            assert isinstance(settings.bayesian, BayesianSettings)
            assert isinstance(settings.database, DatabaseSettings)

    def test_directory_creation(self):
        """Test that cache and log directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            log_dir = Path(temp_dir) / "logs"

            settings = PyratorSettings(cache_dir=cache_dir, log_dir=log_dir)

            assert cache_dir.exists()
            assert log_dir.exists()
            assert cache_dir.is_dir()
            assert log_dir.is_dir()

    def test_nested_settings_override(self):
        """Test overriding nested settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = PyratorSettings(
                compute=ComputeSettings(n_jobs=2, backend="jax"),
                bayesian=BayesianSettings(chains=8, sampler="metropolis"),
                database=DatabaseSettings(engine="duckdb"),
                cache_dir=Path(temp_dir) / "cache",
                log_dir=Path(temp_dir) / "logs",
            )

            assert settings.compute.n_jobs == 2
            assert settings.compute.backend == "jax"
            assert settings.bayesian.chains == 8
            assert settings.bayesian.sampler == "metropolis"
            assert settings.database.engine == "duckdb"

    def test_global_settings(self):
        """Test global settings validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = PyratorSettings(
                random_seed=42,
                debug=True,
                profile=True,
                cache_dir=Path(temp_dir) / "cache",
                log_dir=Path(temp_dir) / "logs",
            )

            assert settings.random_seed == 42
            assert settings.debug is True
            assert settings.profile is True

            with pytest.raises(ValueError):
                PyratorSettings(random_seed=-1)

    def test_settings_customise_sources(self):
        """Test settings source customization."""
        sources = PyratorSettings.settings_customise_sources(
            PyratorSettings,
            init_settings=None,
            env_settings=None,
            dotenv_settings=None,
            file_secret_settings=None,
        )

        assert len(sources) == 4
        # Check order: init, env, dotenv, secrets (all None when passed as None)
        assert sources[0] is None
        assert sources[1] is None
        assert sources[2] is None
        assert sources[3] is None


class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_env_variable_loading(self):
        """Test loading settings from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "PYRATOR_COMPUTE__N_JOBS": "4",
                "PYRATOR_COMPUTE__BACKEND": "jax",
                "PYRATOR_BAYESIAN__CHAINS": "8",
                "PYRATOR_BAYESIAN__USE_GPU": "true",
                "PYRATOR_DATABASE__ENGINE": "duckdb",
                "PYRATOR_DEBUG": "true",
                "PYRATOR_RANDOM_SEED": "123",
            },
        ):
            settings = PyratorSettings()

            assert settings.compute.n_jobs == 4
            assert settings.compute.backend == "jax"
            assert settings.bayesian.chains == 8
            assert settings.bayesian.use_gpu is False  # Falls back to CPU when JAX not available
            assert settings.database.engine == "duckdb"
            assert settings.debug is True
            assert settings.random_seed == 123
