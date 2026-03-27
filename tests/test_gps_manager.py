"""Tests for GPS Manager."""


from wake_focus.fleet.gps_manager import GPSPosition, SimulationGPSSource, haversine_km


class TestHaversine:
    def test_same_point(self):
        assert haversine_km(41.0, 69.0, 41.0, 69.0) == 0.0

    def test_known_distance(self):
        # Tashkent to Samarkand ~270km
        dist = haversine_km(41.299, 69.240, 39.654, 66.959)
        assert 250 < dist < 300

    def test_short_distance(self):
        # Two very close points
        dist = haversine_km(41.311, 69.279, 41.312, 69.280)
        assert 0.05 < dist < 0.5


class TestSimulationGPS:
    def test_returns_position(self):
        source = SimulationGPSSource()
        pos = source.get_position()
        assert pos.has_fix
        assert pos.lat != 0
        assert pos.lon != 0

    def test_position_changes(self):
        source = SimulationGPSSource()
        pos1 = source.get_position()
        pos2 = source.get_position()
        # Should have moved slightly
        assert pos1.lat != pos2.lat or pos1.lon != pos2.lon

    def test_has_speed(self):
        source = SimulationGPSSource(speed_kmh=50.0)
        pos = source.get_position()
        assert pos.speed_kmh > 0


class TestGPSPosition:
    def test_to_dict(self):
        pos = GPSPosition(lat=41.0, lon=69.0, speed_kmh=50.0)
        d = pos.to_dict()
        assert d["lat"] == 41.0
        assert d["speed_kmh"] == 50.0
