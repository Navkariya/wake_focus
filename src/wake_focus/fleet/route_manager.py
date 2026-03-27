"""
Wake Focus - Route Manager (OSRM Integration)

Queries OSRM for driving routes, supports alternative routes,
and provides incident-aware rerouting.
"""

import logging
from typing import Optional

import requests

from wake_focus.constants import OSRM_DEFAULT_URL

logger = logging.getLogger(__name__)


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode Google-encoded polyline string to lat/lon pairs."""
    try:
        import polyline as pl
        return [(lat, lon) for lat, lon in pl.decode(encoded)]
    except ImportError:
        # Manual decode fallback
        return _manual_decode_polyline(encoded)


def _manual_decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Manual polyline decoder without external dependency."""
    points = []
    index = 0
    lat = 0
    lng = 0

    while index < len(encoded):
        # Latitude
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += (~(result >> 1) if result & 1 else result >> 1)

        # Longitude
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lng += (~(result >> 1) if result & 1 else result >> 1)

        points.append((lat / 1e5, lng / 1e5))

    return points


class RouteResult:
    """Result from a routing query."""

    def __init__(
        self,
        points: list[tuple[float, float]] = None,
        distance_km: float = 0.0,
        duration_min: float = 0.0,
        is_alternative: bool = False,
    ):
        self.points = points or []
        self.distance_km = distance_km
        self.duration_min = duration_min
        self.is_alternative = is_alternative


class RouteManager:
    """OSRM routing with incident-aware alternative computation."""

    def __init__(self, osrm_url: str = OSRM_DEFAULT_URL, timeout: int = 10):
        self._osrm_url = osrm_url.rstrip("/")
        self._timeout = timeout
        logger.info("RouteManager: OSRM server=%s", self._osrm_url)

    def get_route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        alternatives: bool = True,
    ) -> list[RouteResult]:
        """Get driving route(s) from OSRM.

        Args:
            origin: (lat, lon)
            destination: (lat, lon)
            alternatives: Request alternative routes.

        Returns:
            List of RouteResult objects.
        """
        # OSRM expects lon,lat order
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        url = f"{self._osrm_url}/route/v1/driving/{coords}"

        params = {
            "overview": "full",
            "geometries": "polyline",
            "alternatives": "true" if alternatives else "false",
        }

        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "Ok":
                logger.warning("OSRM returned non-Ok: %s", data.get("code"))
                return []

            routes = []
            for i, route in enumerate(data.get("routes", [])):
                geometry = route.get("geometry", "")
                points = decode_polyline(geometry) if geometry else []
                distance_km = route.get("distance", 0) / 1000.0
                duration_min = route.get("duration", 0) / 60.0

                routes.append(
                    RouteResult(
                        points=points,
                        distance_km=distance_km,
                        duration_min=duration_min,
                        is_alternative=i > 0,
                    )
                )

            logger.info("OSRM returned %d route(s)", len(routes))
            return routes

        except requests.RequestException as e:
            logger.error("OSRM request failed: %s", e)
            return []

    def get_alternative_avoiding_area(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        avoid_center: tuple[float, float],
        avoid_radius_km: float = 0.5,
    ) -> Optional[RouteResult]:
        """Get route that avoids a specific area (incident zone).

        Strategy: Request multiple alternatives and pick the one
        furthest from the avoidance zone.
        """
        routes = self.get_route(origin, destination, alternatives=True)
        if not routes:
            return None

        from wake_focus.fleet.gps_manager import haversine_km

        best_route = None
        best_min_dist = -1

        for route in routes:
            # Find minimum distance from route to avoidance center
            min_dist = float("inf")
            for lat, lon in route.points:
                dist = haversine_km(lat, lon, avoid_center[0], avoid_center[1])
                min_dist = min(min_dist, dist)

            # Prefer routes that stay furthest from the incident
            if min_dist > avoid_radius_km and min_dist > best_min_dist:
                best_min_dist = min_dist
                best_route = route
                best_route.is_alternative = True

        if best_route:
            logger.info(
                "Found alternative route avoiding incident zone (min_dist=%.2fkm)",
                best_min_dist,
            )
        else:
            logger.warning("No route found avoiding the incident zone")
            # Fall back to the first route
            if routes:
                best_route = routes[0]

        return best_route
