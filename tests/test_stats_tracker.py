"""Tests for Vehicle Stats Tracker."""


from wake_focus.fleet.gps_manager import GPSPosition
from wake_focus.vehicle.stats_tracker import StatsTracker


class TestStatsTracker:
    def test_initial_values(self):
        tracker = StatsTracker()
        assert tracker.total_distance_km == 0.0
        assert tracker.total_fuel_liters == 0.0

    def test_distance_accumulation(self):
        tracker = StatsTracker(fuel_economy_l_per_100km=10.0)

        # Two positions ~0.11km apart
        pos1 = GPSPosition(lat=41.311, lon=69.279, has_fix=True)
        pos2 = GPSPosition(lat=41.312, lon=69.280, has_fix=True)

        tracker.update(pos1)
        tracker.update(pos2)

        assert tracker.total_distance_km > 0
        assert tracker.total_fuel_liters > 0

    def test_no_fix_ignored(self):
        tracker = StatsTracker()
        pos = GPSPosition(has_fix=False)
        tracker.update(pos)
        assert tracker.total_distance_km == 0.0

    def test_reset(self):
        tracker = StatsTracker()
        pos1 = GPSPosition(lat=41.311, lon=69.279, has_fix=True)
        pos2 = GPSPosition(lat=41.315, lon=69.285, has_fix=True)
        tracker.update(pos1)
        tracker.update(pos2)
        assert tracker.total_distance_km > 0

        tracker.reset()
        assert tracker.total_distance_km == 0.0
        assert tracker.total_fuel_liters == 0.0

    def test_gps_noise_filtering(self):
        """Large jumps (>1km) should be filtered out."""
        tracker = StatsTracker()
        pos1 = GPSPosition(lat=41.311, lon=69.279, has_fix=True)
        pos2 = GPSPosition(lat=42.0, lon=70.0, has_fix=True)  # ~100km jump

        tracker.update(pos1)
        tracker.update(pos2)

        # Should NOT add ~100km
        assert tracker.total_distance_km < 1.0
