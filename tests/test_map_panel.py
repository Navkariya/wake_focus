"""Tests for headless-safe map panel behavior."""

from wake_focus.ui.map_panel import MapPanel, _build_yandex_bootstrap_html


def test_map_panel_uses_fallback_when_webengine_disabled(qtbot, monkeypatch):
    monkeypatch.setenv("WAKE_FOCUS_DISABLE_WEBENGINE", "1")

    panel = MapPanel()
    qtbot.addWidget(panel)

    assert panel.is_interactive is False


def test_map_panel_uses_fallback_in_offscreen_mode(qtbot, monkeypatch):
    monkeypatch.delenv("WAKE_FOCUS_DISABLE_WEBENGINE", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    panel = MapPanel()
    qtbot.addWidget(panel)

    assert panel.is_interactive is False


def test_yandex_map_html_bootstrap_contains_api_loader():
    html = _build_yandex_bootstrap_html(
        api_key="test-key",
        lang="en_US",
        center=(41.311, 69.279),
        zoom=15,
        traffic_enabled=True,
        auto_follow=True,
    )

    assert "api-maps.yandex.ru/2.1/" in html
    assert "Wake Focus Nav" in html
    assert "trafficControl" in html
