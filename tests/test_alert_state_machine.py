"""Tests for the Alert State Machine."""

import time


from wake_focus.constants import AlertState
from wake_focus.core.alert_state_machine import AlertStateMachine
from wake_focus.core.perception_engine import PerceptionResult
from wake_focus.ml.drowsiness_detector import DrowsinessState
from wake_focus.ml.gaze_analyzer import GazeResult
from wake_focus.ml.head_pose import HeadPose


def make_perception(
    is_drowsy=False,
    is_road_facing=True,
    has_object=False,
    face_detected=True,
) -> PerceptionResult:
    """Create a PerceptionResult with specific flags."""
    result = PerceptionResult(
        timestamp=time.monotonic(),
        face_detected=face_detected,
        drowsiness=DrowsinessState(is_drowsy=is_drowsy),
        head_pose=HeadPose(is_road_facing=is_road_facing),
        gaze=GazeResult(
            is_looking_forward=is_road_facing,
            is_attending_road=is_road_facing,
        ),
        has_distraction_object=has_object,
        is_attending_road=is_road_facing,
        is_holding_distraction=has_object and not is_road_facing,
    )
    return result


class TestAlertStateMachine:
    """Test alert state machine transitions."""

    def test_starts_normal(self):
        fsm = AlertStateMachine()
        assert fsm.current_state == AlertState.NORMAL

    def test_drowsiness_triggers_alert(self):
        fsm = AlertStateMachine()

        # Normal state
        p = make_perception(is_drowsy=False)
        status = fsm.update(p)
        assert status.state == AlertState.NORMAL
        assert not status.show_border

        # Drowsy detected → transitions through DETECTED → ALERTING
        p = make_perception(is_drowsy=True)
        fsm.update(p)  # DETECTED
        status = fsm.update(p)  # ALERTING
        assert status.state == AlertState.ALERTING_DROWSY
        assert status.show_border
        assert status.border_color == (255, 0, 0)  # Red

    def test_drowsiness_recovery(self):
        fsm = AlertStateMachine(drowsy_recovery_seconds=0.1)

        # Trigger drowsiness
        p = make_perception(is_drowsy=True)
        fsm.update(p)
        fsm.update(p)
        assert fsm.current_state == AlertState.ALERTING_DROWSY

        # Eyes open, attending road
        p = make_perception(is_drowsy=False, is_road_facing=True)
        fsm.update(p)
        assert fsm.current_state == AlertState.RECOVERY_CHECK_DROWSY

        # Wait for recovery stable period
        time.sleep(0.15)
        status = fsm.update(p)
        assert status.state == AlertState.NORMAL

    def test_distraction_needs_30s(self):
        fsm = AlertStateMachine(distraction_onset_seconds=0.1)  # Shortened for test

        # Start distraction: object + not facing road
        p = make_perception(has_object=True, is_road_facing=False)
        fsm.update(p)
        assert fsm.current_state == AlertState.DISTRACTION_TRACKING

        # Wait past onset threshold
        time.sleep(0.15)
        status = fsm.update(p)
        assert status.state == AlertState.ALERTING_DISTRACTION
        assert status.border_color == (255, 165, 0)  # Orange

    def test_distraction_cancelled_if_attention_returns(self):
        fsm = AlertStateMachine(distraction_onset_seconds=30.0)

        # Start tracking
        p = make_perception(has_object=True, is_road_facing=False)
        fsm.update(p)
        assert fsm.current_state == AlertState.DISTRACTION_TRACKING

        # Return attention before 30s
        p = make_perception(has_object=False, is_road_facing=True)
        status = fsm.update(p)
        assert status.state == AlertState.NORMAL

    def test_drowsiness_overrides_distraction(self):
        fsm = AlertStateMachine(distraction_onset_seconds=0.05)

        # Start distraction
        p = make_perception(has_object=True, is_road_facing=False)
        fsm.update(p)
        time.sleep(0.06)
        fsm.update(p)
        assert fsm.current_state == AlertState.ALERTING_DISTRACTION

        # Drowsiness kicks in — should override
        p = make_perception(is_drowsy=True, has_object=True, is_road_facing=False)
        fsm.update(p)
        fsm.update(p)
        assert fsm.current_state in (AlertState.DROWSINESS_DETECTED, AlertState.ALERTING_DROWSY)

    def test_reset(self):
        fsm = AlertStateMachine()

        p = make_perception(is_drowsy=True)
        fsm.update(p)
        fsm.update(p)
        assert fsm.current_state == AlertState.ALERTING_DROWSY

        fsm.reset()
        assert fsm.current_state == AlertState.NORMAL
