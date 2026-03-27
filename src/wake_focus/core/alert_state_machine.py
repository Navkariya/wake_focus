"""
Wake Focus - Alert State Machine

Implements the dual-alert system with deterministic priority:
  <<ogohlantirish>>  — Drowsiness (RED border, continuous beep)
  <<ogohlantirish2>> — Prolonged distraction (ORANGE border, 2 beeps / 5s)

State transitions use hysteresis and stable-window checks to avoid
flicker and false toggles. Drowsiness always has highest priority.
"""

import logging
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from wake_focus.constants import (
    DISTRACTION_BEEP_INTERVAL_MS,
    DISTRACTION_BEEPS_PER_ALERT,
    DISTRACTION_ONSET_SECONDS,
    DISTRACTION_RECOVERY_STABLE_SECONDS,
    DROWSY_BEEP_INTERVAL_MS,
    DROWSY_RECOVERY_STABLE_SECONDS,
    AlertPriority,
    AlertState,
)
from wake_focus.core.perception_engine import PerceptionResult

logger = logging.getLogger(__name__)


@dataclass
class AlertStatus:
    """Current alert output for UI rendering."""

    state: AlertState = AlertState.NORMAL
    active_priority: AlertPriority = AlertPriority.NONE
    # Border
    show_border: bool = False
    border_color: tuple[int, int, int] = (0, 0, 0)
    # Audio commands
    should_beep: bool = False
    beep_count: int = 0  # Number of beeps in this event
    # Debug info
    drowsy_frame_count: int = 0
    distraction_seconds: float = 0.0
    recovery_seconds: float = 0.0


class AlertStateMachine(QObject):
    """Implements <<ogohlantirish>> and <<ogohlantirish2>> alert behaviors."""

    # Emitted whenever alert status changes
    alert_changed = Signal(object)  # AlertStatus
    # Specific audio commands
    start_drowsy_beep = Signal()
    stop_drowsy_beep = Signal()
    play_distraction_beep = Signal()  # 2 beeps
    stop_distraction_beep = Signal()

    def __init__(
        self,
        drowsy_recovery_seconds: float = DROWSY_RECOVERY_STABLE_SECONDS,
        distraction_onset_seconds: float = DISTRACTION_ONSET_SECONDS,
        distraction_beep_interval_ms: int = DISTRACTION_BEEP_INTERVAL_MS,
        distraction_beeps_per_alert: int = DISTRACTION_BEEPS_PER_ALERT,
        distraction_recovery_seconds: float = DISTRACTION_RECOVERY_STABLE_SECONDS,
        drowsy_beep_interval_ms: int = DROWSY_BEEP_INTERVAL_MS,
        border_color_drowsy: tuple[int, int, int] = (255, 0, 0),
        border_color_distraction: tuple[int, int, int] = (255, 165, 0),
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        self._drowsy_recovery_s = drowsy_recovery_seconds
        self._distraction_onset_s = distraction_onset_seconds
        self._distraction_beep_interval_ms = distraction_beep_interval_ms
        self._distraction_beeps = distraction_beeps_per_alert
        self._distraction_recovery_s = distraction_recovery_seconds
        self._drowsy_beep_interval_ms = drowsy_beep_interval_ms
        self._color_drowsy = border_color_drowsy
        self._color_distraction = border_color_distraction

        # Internal state
        self._state = AlertState.NORMAL
        self._prev_state = AlertState.NORMAL

        # Drowsiness tracking
        self._drowsy_start_time: float | None = None
        self._drowsy_recovery_start: float | None = None
        self._last_drowsy_beep_time: float = 0.0

        # Distraction tracking
        self._distraction_start_time: float | None = None
        self._distraction_recovery_start: float | None = None
        self._last_distraction_beep_time: float = 0.0

        self._current_status = AlertStatus()

        logger.info(
            "AlertStateMachine: drowsy_recovery=%.1fs, distraction_onset=%.1fs, "
            "distraction_interval=%dms, distraction_recovery=%.1fs",
            drowsy_recovery_seconds,
            distraction_onset_seconds,
            distraction_beep_interval_ms,
            distraction_recovery_seconds,
        )

    def update(self, perception: PerceptionResult) -> AlertStatus:
        """Update alert state based on perception results.

        Args:
            perception: Latest PerceptionResult from the perception engine.

        Returns:
            AlertStatus with current alert state and UI commands.
        """
        now = time.monotonic()
        is_drowsy = perception.drowsiness.is_drowsy
        is_distracted = perception.is_holding_distraction and not perception.is_attending_road
        is_attending = perception.is_attending_road and perception.face_detected

        self._prev_state = self._state

        # ── Drowsiness state transitions (HIGHEST PRIORITY) ─────────────
        if self._state == AlertState.NORMAL or self._state == AlertState.DISTRACTION_TRACKING:
            if is_drowsy:
                self._state = AlertState.DROWSINESS_DETECTED
                self._drowsy_start_time = now

        if self._state == AlertState.DROWSINESS_DETECTED:
            if is_drowsy:
                # Confirm drowsiness (immediate — EAR already uses frame-counting)
                self._state = AlertState.ALERTING_DROWSY
                self._last_drowsy_beep_time = 0.0
                self.start_drowsy_beep.emit()
                logger.warning("<<ogohlantirish>> ACTIVATED: Drowsiness alert")
            else:
                self._state = AlertState.NORMAL
                self._drowsy_start_time = None

        if self._state == AlertState.ALERTING_DROWSY:
            if not is_drowsy and is_attending:
                # Begin recovery check
                self._state = AlertState.RECOVERY_CHECK_DROWSY
                self._drowsy_recovery_start = now
                logger.info("Drowsiness recovery check started")
            else:
                # Still drowsy — continue beeping
                self._handle_drowsy_beep(now)

        if self._state == AlertState.RECOVERY_CHECK_DROWSY:
            if is_drowsy:
                # Relapse
                self._state = AlertState.ALERTING_DROWSY
                self._drowsy_recovery_start = None
                logger.warning("Drowsiness relapse during recovery check")
            elif is_attending:
                elapsed = now - (self._drowsy_recovery_start or now)
                if elapsed >= self._drowsy_recovery_s:
                    # Fully recovered
                    self._state = AlertState.NORMAL
                    self._drowsy_recovery_start = None
                    self._drowsy_start_time = None
                    self.stop_drowsy_beep.emit()
                    logger.info("<<ogohlantirish>> CLEARED: Driver recovered")
            else:
                # Not attending but not drowsy — reset recovery timer
                self._drowsy_recovery_start = now

        # ── Distraction state transitions ───────────────────────────────
        # Only process if not in drowsy alert (priority rule)
        if self._state == AlertState.NORMAL:
            if is_distracted:
                self._state = AlertState.DISTRACTION_TRACKING
                self._distraction_start_time = now
                logger.debug("Distraction tracking started")

        if self._state == AlertState.DISTRACTION_TRACKING:
            if not is_distracted:
                # Attention returned before threshold
                self._state = AlertState.NORMAL
                self._distraction_start_time = None
                logger.debug("Distraction tracking cancelled — attention returned")
            elif is_drowsy:
                # Drowsiness takes priority
                self._state = AlertState.DROWSINESS_DETECTED
                self._drowsy_start_time = now
                self._distraction_start_time = None
            else:
                elapsed = now - (self._distraction_start_time or now)
                if elapsed >= self._distraction_onset_s:
                    # Threshold exceeded — activate distraction alert
                    self._state = AlertState.ALERTING_DISTRACTION
                    self._last_distraction_beep_time = 0.0
                    self.play_distraction_beep.emit()
                    logger.warning(
                        "<<ogohlantirish2>> ACTIVATED: Prolonged distraction (%.1fs)", elapsed
                    )

        if self._state == AlertState.ALERTING_DISTRACTION:
            if is_drowsy:
                # Drowsiness takes priority
                self._state = AlertState.DROWSINESS_DETECTED
                self._drowsy_start_time = now
                self.stop_distraction_beep.emit()
            elif not is_distracted and is_attending:
                # Begin recovery check
                self._state = AlertState.RECOVERY_CHECK_DISTRACTION
                self._distraction_recovery_start = now
                logger.info("Distraction recovery check started")
            else:
                # Still distracted — beep every 5 seconds
                self._handle_distraction_beep(now)

        if self._state == AlertState.RECOVERY_CHECK_DISTRACTION:
            if is_distracted:
                # Relapse
                self._state = AlertState.ALERTING_DISTRACTION
                self._distraction_recovery_start = None
                logger.warning("Distraction relapse during recovery check")
            elif is_drowsy:
                # Drowsiness takes priority
                self._state = AlertState.DROWSINESS_DETECTED
                self._drowsy_start_time = now
                self.stop_distraction_beep.emit()
            elif is_attending:
                elapsed = now - (self._distraction_recovery_start or now)
                if elapsed >= self._distraction_recovery_s:
                    # Fully recovered
                    self._state = AlertState.NORMAL
                    self._distraction_recovery_start = None
                    self._distraction_start_time = None
                    self.stop_distraction_beep.emit()
                    logger.info("<<ogohlantirish2>> CLEARED: Attention recovered")
            else:
                self._distraction_recovery_start = now

        # ── Build output status ─────────────────────────────────────────
        status = self._build_status(now)

        if self._state != self._prev_state:
            self.alert_changed.emit(status)

        self._current_status = status
        return status

    def _handle_drowsy_beep(self, now: float) -> None:
        """Handle periodic beeping during drowsiness alert."""
        interval_s = self._drowsy_beep_interval_ms / 1000.0
        if now - self._last_drowsy_beep_time >= interval_s:
            self._last_drowsy_beep_time = now
            # Beep is continuous via audio manager timer, not individual triggers

    def _handle_distraction_beep(self, now: float) -> None:
        """Handle periodic double-beep during distraction alert."""
        interval_s = self._distraction_beep_interval_ms / 1000.0
        if now - self._last_distraction_beep_time >= interval_s:
            self._last_distraction_beep_time = now
            self.play_distraction_beep.emit()

    def _build_status(self, now: float) -> AlertStatus:
        """Build AlertStatus from current internal state."""
        status = AlertStatus(state=self._state)

        if self._state in (AlertState.ALERTING_DROWSY, AlertState.RECOVERY_CHECK_DROWSY):
            status.active_priority = AlertPriority.DROWSINESS
            status.show_border = True
            status.border_color = self._color_drowsy
            status.should_beep = self._state == AlertState.ALERTING_DROWSY
            status.beep_count = 1

        elif self._state in (
            AlertState.ALERTING_DISTRACTION,
            AlertState.RECOVERY_CHECK_DISTRACTION,
        ):
            status.active_priority = AlertPriority.DISTRACTION
            status.show_border = True
            status.border_color = self._color_distraction
            status.should_beep = self._state == AlertState.ALERTING_DISTRACTION
            status.beep_count = self._distraction_beeps

        else:
            status.active_priority = AlertPriority.NONE
            status.show_border = False
            status.should_beep = False

        # Debug info
        if self._distraction_start_time:
            status.distraction_seconds = now - self._distraction_start_time
        if self._drowsy_recovery_start:
            status.recovery_seconds = now - self._drowsy_recovery_start
        elif self._distraction_recovery_start:
            status.recovery_seconds = now - self._distraction_recovery_start

        return status

    @property
    def current_state(self) -> AlertState:
        return self._state

    @property
    def current_status(self) -> AlertStatus:
        return self._current_status

    def reset(self) -> None:
        """Reset to normal state."""
        self._state = AlertState.NORMAL
        self._drowsy_start_time = None
        self._drowsy_recovery_start = None
        self._distraction_start_time = None
        self._distraction_recovery_start = None
        self.stop_drowsy_beep.emit()
        self.stop_distraction_beep.emit()
        logger.info("Alert state machine reset to NORMAL")
