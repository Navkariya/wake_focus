"""
Wake Focus - Map Panel (300x500)

Uses Yandex Maps JavaScript API in interactive environments and keeps the map
page loaded permanently. Track points, fleet markers, incidents, and reroute
polylines are updated incrementally through JavaScript instead of reloading the
entire HTML document on every refresh.
"""

import json
import logging
import os

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from wake_focus.constants import MAP_PANEL_H, MAP_PANEL_W

logger = logging.getLogger(__name__)


def _should_use_webengine() -> bool:
    """Return True when the interactive web map should be enabled."""
    if os.environ.get("WAKE_FOCUS_DISABLE_WEBENGINE") == "1":
        return False
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    return True


def _build_yandex_bootstrap_html(
    api_key: str,
    lang: str,
    center: tuple[float, float],
    zoom: int,
    traffic_enabled: bool,
    auto_follow: bool,
) -> str:
    """Return the HTML shell for the embedded Yandex map page."""
    controls = ["zoomControl", "typeSelector", "fullscreenControl", "geolocationControl"]
    if traffic_enabled:
        controls.insert(1, "trafficControl")
    initial_state = {
        "center": list(center),
        "zoom": zoom,
        "track_points": [],
        "fleet_devices": [],
        "incidents": [],
        "route_points": [],
        "traffic_enabled": traffic_enabled,
        "auto_follow": auto_follow,
    }
    initial_state_json = json.dumps(initial_state, ensure_ascii=True)
    controls_json = json.dumps(controls, ensure_ascii=True)
    api_key = api_key.replace("&", "&amp;").replace('"', "&quot;")
    lang = lang.replace("&", "&amp;").replace('"', "&quot;")

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://api-maps.yandex.ru/2.1/?apikey={api_key}&lang={lang}"></script>
    <style>
        html, body, #map {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #0f172a;
        }}
        .map-overlay {{
            position: absolute;
            left: 8px;
            right: 8px;
            top: 8px;
            z-index: 1000;
            pointer-events: none;
            display: flex;
            justify-content: space-between;
            gap: 8px;
            font-family: Arial, sans-serif;
        }}
        .overlay-pill {{
            background: rgba(15, 23, 42, 0.82);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 11px;
            line-height: 1.2;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="map-overlay">
        <div class="overlay-pill">Wake Focus Nav</div>
        <div class="overlay-pill" id="routeSummary">Live traffic route</div>
    </div>
    <script>
        const wakeFocusState = {initial_state_json};
        let map = null;
        let ownMarker = null;
        let ownTrack = null;
        let reroutePath = null;
        let fleetObjects = [];
        let incidentObjects = [];

        function clearObjects(objects) {{
            objects.forEach(function(item) {{
                if (item && map) {{
                    map.geoObjects.remove(item);
                }}
            }});
            objects.length = 0;
        }}

        function statusColor(status) {{
            const normalized = String(status || "").toLowerCase();
            if (normalized === "online") return "#10b981";
            if (normalized === "alerting" || normalized === "drowsy" || normalized === "distracted") return "#f59e0b";
            return "#ef4444";
        }}

        function updateSummary(state) {{
            const routeEl = document.getElementById("routeSummary");
            const routeCount = (state.route_points || []).length;
            const incidentCount = (state.incidents || []).length;
            routeEl.textContent = routeCount > 1
                ? "Route active • " + incidentCount + " incident(s)"
                : "Live traffic route";
        }}

        function applyState(state) {{
            Object.assign(wakeFocusState, state || {{}});
            if (!map) {{
                return true;
            }}

            const center = wakeFocusState.center || [{center[0]}, {center[1]}];
            const zoom = wakeFocusState.zoom || {zoom};
            const trackPoints = wakeFocusState.track_points || [];
            const routePoints = wakeFocusState.route_points || [];
            const fleetDevices = wakeFocusState.fleet_devices || [];
            const incidents = wakeFocusState.incidents || [];
            const autoFollow = !!wakeFocusState.auto_follow;

            if (!ownTrack) {{
                ownTrack = new ymaps.Polyline([], {{}}, {{
                    strokeColor: "#2563eb",
                    strokeWidth: 4,
                    strokeOpacity: 0.9
                }});
                map.geoObjects.add(ownTrack);
            }}
            ownTrack.geometry.setCoordinates(trackPoints);

            if (!reroutePath) {{
                reroutePath = new ymaps.Polyline([], {{}}, {{
                    strokeColor: "#22c55e",
                    strokeWidth: 5,
                    strokeOpacity: 0.85,
                    strokeStyle: "shortdash"
                }});
                map.geoObjects.add(reroutePath);
            }}
            reroutePath.geometry.setCoordinates(routePoints);

            if (trackPoints.length > 0) {{
                const lastPosition = trackPoints[trackPoints.length - 1];
                if (!ownMarker) {{
                    ownMarker = new ymaps.Placemark(lastPosition, {{
                        balloonContent: "<strong>You</strong>"
                    }}, {{
                        preset: "islands#blueCircleDotIcon"
                    }});
                    map.geoObjects.add(ownMarker);
                }} else {{
                    ownMarker.geometry.setCoordinates(lastPosition);
                }}

                if (autoFollow) {{
                    map.setCenter(lastPosition, zoom, {{ duration: 120 }});
                }}
            }} else {{
                map.setCenter(center, zoom, {{ duration: 120 }});
            }}

            clearObjects(fleetObjects);
            fleetDevices.forEach(function(dev) {{
                const marker = new ymaps.Placemark(
                    [dev.lat, dev.lon],
                    {{
                        balloonContent:
                            "<strong>" + dev.name + "</strong><br/>Status: " + dev.status
                    }},
                    {{
                        preset: "islands#circleDotIcon",
                        iconColor: statusColor(dev.status)
                    }}
                );
                fleetObjects.push(marker);
                map.geoObjects.add(marker);
            }});

            clearObjects(incidentObjects);
            incidents.forEach(function(inc) {{
                const marker = new ymaps.Placemark(
                    [inc.lat, inc.lon],
                    {{
                        balloonContent:
                            "<strong>Incident</strong><br/>" + (inc.desc || "Reported event")
                    }},
                    {{
                        preset: "islands#redWarningIcon"
                    }}
                );
                const zone = new ymaps.Circle(
                    [[inc.lat, inc.lon], 500],
                    {{}},
                    {{
                        fillColor: "rgba(239, 68, 68, 0.15)",
                        strokeColor: "#ef4444",
                        strokeOpacity: 0.75,
                        strokeWidth: 2
                    }}
                );
                incidentObjects.push(marker, zone);
                map.geoObjects.add(marker);
                map.geoObjects.add(zone);
            }});

            updateSummary(wakeFocusState);
            return true;
        }}

        window.wakeFocusSetState = function(state) {{
            return applyState(state);
        }};

        window.wakeFocusIsReady = function() {{
            return !!map;
        }};

        ymaps.ready(function() {{
            map = new ymaps.Map("map", {{
                center: wakeFocusState.center,
                zoom: wakeFocusState.zoom,
                controls: {controls_json}
            }}, {{
                suppressMapOpenBlock: true
            }});

            map.behaviors.disable("scrollZoom");
            applyState(wakeFocusState);
        }});
    </script>
</body>
</html>
"""


class _FallbackMapView(QLabel):
    """Headless-safe fallback widget for map status information."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setObjectName("mapFallbackView")

    def set_summary(
        self,
        center: tuple[float, float],
        track_count: int,
        fleet_count: int,
        incident_count: int,
        route_count: int,
        reason: str,
    ) -> None:
        lat, lon = center
        self.setText(
            "Map fallback active\n"
            f"{reason}\n\n"
            f"Center: {lat:.6f}, {lon:.6f}\n"
            f"Track points: {track_count}\n"
            f"Fleet devices: {fleet_count}\n"
            f"Incidents: {incident_count}\n"
            f"Route points: {route_count}\n"
        )


def _build_web_view(parent: QWidget | None) -> QWidget | None:
    """Create a QWebEngineView when supported, otherwise return None."""
    if not _should_use_webengine():
        return None

    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception as exc:  # pragma: no cover - depends on system packages
        logger.warning("Qt WebEngine unavailable, using map fallback: %s", exc)
        return None

    return QWebEngineView(parent)


class MapPanel(QFrame):
    """Map panel displaying GPS tracks, fleet devices, and routes."""

    def __init__(
        self,
        default_center: tuple[float, float] = (41.311, 69.279),
        default_zoom: int = 15,
        provider: str = "yandex",
        yandex_api_key: str = "",
        yandex_lang: str = "en_US",
        traffic_enabled: bool = True,
        auto_follow: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("mapPanel")
        self.setFixedSize(MAP_PANEL_W, MAP_PANEL_H)

        self._center = default_center
        self._zoom = default_zoom
        self._provider = provider.lower()
        self._yandex_api_key = yandex_api_key
        self._yandex_lang = yandex_lang
        self._traffic_enabled = traffic_enabled
        self._auto_follow = auto_follow
        self._page_ready = False
        self._fallback_reason = "Interactive map is disabled in this environment."

        self._track_points: list[tuple[float, float]] = []
        self._fleet_devices: dict[str, tuple[float, float, str, str]] = {}
        self._incidents: dict[str, tuple[float, float, str]] = {}
        self._route_points: list[tuple[float, float]] = []

        self._view = _build_web_view(self)
        self._interactive = self._view is not None and self._provider == "yandex"

        if self._interactive and not self._yandex_api_key:
            self._interactive = False
            self._fallback_reason = (
                "Yandex Maps API key is missing. Set map.yandex.api_key or "
                "YANDEX_MAPS_API_KEY to enable the navigator-style map."
            )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if self._interactive:
            self._view.setFixedSize(MAP_PANEL_W, MAP_PANEL_H)
            layout.addWidget(self._view)
            self._view.loadFinished.connect(self._on_map_loaded)
            self._view.setHtml(
                _build_yandex_bootstrap_html(
                    api_key=self._yandex_api_key,
                    lang=self._yandex_lang,
                    center=self._center,
                    zoom=self._zoom,
                    traffic_enabled=self._traffic_enabled,
                    auto_follow=self._auto_follow,
                ),
                baseUrl=QUrl("https://wake-focus.local/"),
            )
            logger.info("MapPanel using Yandex Maps interactive mode")
        else:
            self._view = _FallbackMapView(self)
            self._view.setFixedSize(MAP_PANEL_W, MAP_PANEL_H)
            layout.addWidget(self._view)
            logger.info("MapPanel using fallback view")
            self._render_fallback()

        logger.info(
            "MapPanel initialized: provider=%s, center=%s, zoom=%d, interactive=%s",
            self._provider,
            self._center,
            self._zoom,
            self._interactive,
        )

    @property
    def is_interactive(self) -> bool:
        return self._interactive

    def _build_state(self) -> dict:
        return {
            "center": list(self._center),
            "zoom": self._zoom,
            "track_points": self._track_points,
            "fleet_devices": [
                {"lat": lat, "lon": lon, "name": name, "status": status, "id": did}
                for did, (lat, lon, name, status) in self._fleet_devices.items()
            ],
            "incidents": [
                {"lat": lat, "lon": lon, "desc": desc, "id": iid}
                for iid, (lat, lon, desc) in self._incidents.items()
            ],
            "route_points": self._route_points,
            "traffic_enabled": self._traffic_enabled,
            "auto_follow": self._auto_follow,
        }

    def _render_fallback(self) -> None:
        if isinstance(self._view, _FallbackMapView):
            self._view.set_summary(
                center=self._center,
                track_count=len(self._track_points),
                fleet_count=len(self._fleet_devices),
                incident_count=len(self._incidents),
                route_count=len(self._route_points),
                reason=self._fallback_reason,
            )

    @Slot(bool)
    def _on_map_loaded(self, ok: bool) -> None:
        self._page_ready = ok
        if ok:
            self._push_state_to_webview()
        else:
            logger.warning("Yandex map page failed to load")

    def _push_state_to_webview(self) -> None:
        if not self._interactive or not self._page_ready:
            return
        state_json = json.dumps(self._build_state(), ensure_ascii=True)
        self._view.page().runJavaScript(f"window.wakeFocusSetState({state_json});")

    @Slot(float, float)
    def update_position(self, lat: float, lon: float) -> None:
        """Add a new GPS position and update the map center."""
        self._track_points.append((lat, lon))
        self._center = (lat, lon)

    @Slot(dict)
    def update_fleet_devices(self, devices: dict) -> None:
        """Update fleet device positions. devices = {id: (lat, lon, name, status)}"""
        self._fleet_devices = devices

    @Slot(str, float, float, str)
    def add_incident(self, incident_id: str, lat: float, lon: float, description: str) -> None:
        """Add an incident marker to the map."""
        self._incidents[incident_id] = (lat, lon, description)

    @Slot(str)
    def remove_incident(self, incident_id: str) -> None:
        """Remove an incident marker."""
        self._incidents.pop(incident_id, None)

    @Slot(list)
    def update_route(self, points: list[tuple[float, float]]) -> None:
        """Update the suggested route polyline."""
        self._route_points = points

    @Slot()
    def refresh_map(self) -> None:
        """Refresh visible map objects without reloading the entire page."""
        if self._interactive:
            self._push_state_to_webview()
        else:
            self._render_fallback()

    def clear_track(self) -> None:
        """Clear GPS track history."""
        self._track_points.clear()
        self.refresh_map()
