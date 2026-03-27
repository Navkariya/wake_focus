"""Tests for Configuration system."""


import yaml

from wake_focus.config import Config


class TestConfig:
    """Test config loading and defaults."""

    def test_defaults_without_file(self):
        """Config should work with no file, using defaults."""
        config = Config(config_path="/nonexistent/path.yaml")
        assert config.window_width == 800
        assert config.window_height == 800
        assert config.camera_index == 0
        assert config.ear_threshold == 0.21

    def test_load_from_file(self, tmp_path):
        """Config should load values from YAML file."""
        config_data = {
            "app": {"window_width": 1024},
            "camera": {"device_index": 2},
            "alerts": {
                "drowsiness": {"ear_threshold": 0.25}
            },
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = Config(config_path=str(config_file))
        assert config.window_width == 1024
        assert config.camera_index == 2
        assert config.ear_threshold == 0.25
        # Unset values should use defaults
        assert config.window_height == 800

    def test_deep_merge(self):
        """Edge config should overlay on top of default config."""
        from wake_focus.config import _deep_merge

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        result = _deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}

    def test_get_nested(self):
        config = Config()
        # Non-existent nested path should return default
        assert config.get("nonexistent.deep.path", "fallback") == "fallback"

    def test_border_color_tuple(self, tmp_path):
        config_data = {
            "alerts": {
                "border": {
                    "drowsy_color": [200, 50, 50]
                }
            }
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = Config(config_path=str(config_file))
        assert config.border_color_drowsy == (200, 50, 50)

    def test_yandex_api_key_can_come_from_environment(self, monkeypatch):
        monkeypatch.setenv("YANDEX_MAPS_API_KEY", "env-key")

        config = Config(config_path="/nonexistent/path.yaml")

        assert config.map_provider == "yandex"
        assert config.yandex_maps_api_key == "env-key"
