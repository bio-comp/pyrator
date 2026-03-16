"""Tests for MonitorConfig classes."""

from pyrator.drift.monitor import (
    CramerVMonitorConfig,
    JsdMonitorConfig,
    MmdMonitorConfig,
    Monitor,
    PsiMonitorConfig,
    WassersteinMonitorConfig,
)


def test_psi_monitor_config_creation():
    config = PsiMonitorConfig(name="test", col="value", n_bins=20)
    assert config.name == "test"
    assert config.col == "value"
    assert config.n_bins == 20
    assert config.metric == "psi"


def test_cramer_v_monitor_config_creation():
    config = CramerVMonitorConfig(name="test", x="col1", y="col2")
    assert config.name == "test"
    assert config.x == "col1"
    assert config.y == "col2"
    assert config.metric == "cramer_v"


def test_jsd_monitor_config_creation():
    config = JsdMonitorConfig(name="test", dist_cols=["a", "b"])
    assert config.name == "test"
    assert config.dist_cols == ["a", "b"]
    assert config.metric == "jsd"


def test_wasserstein_monitor_config_creation():
    config = WassersteinMonitorConfig(name="test", col="value")
    assert config.name == "test"
    assert config.col == "value"
    assert config.metric == "wasserstein"


def test_mmd_monitor_config_creation():
    config = MmdMonitorConfig(name="test", emb_cols=["emb1", "emb2"])
    assert config.name == "test"
    assert config.emb_cols == ["emb1", "emb2"]
    assert config.metric == "mmd"


def test_monitor_dispatch_by_config_type():
    psi_config = PsiMonitorConfig(name="test", col="value")
    monitor = Monitor(psi_config)
    assert isinstance(monitor.config, PsiMonitorConfig)


def test_base_config_fields():
    config = PsiMonitorConfig(
        name="test",
        col="value",
        warn=0.2,
        crit=0.5,
        semantics="delta",
        window_col="window",
    )
    assert config.warn == 0.2
    assert config.crit == 0.5
    assert config.semantics == "delta"
    assert config.window_col == "window"
