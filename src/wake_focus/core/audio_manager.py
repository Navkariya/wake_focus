"""
Wake Focus - Audio Manager

Low-latency beep playback using Qt QSoundEffect.
Generates WAV beep files programmatically if not present.
Handles beep scheduling for drowsiness (continuous) and distraction (2x every 5s).
"""

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, QUrl, Slot
from PySide6.QtMultimedia import QSoundEffect

from wake_focus.constants import BEEP_DURATION_MS, BEEP_FREQUENCY_HZ, BEEP_SAMPLE_RATE, BEEP_VOLUME

logger = logging.getLogger(__name__)


def generate_beep_wav(
    frequency: int = BEEP_FREQUENCY_HZ,
    duration_ms: int = BEEP_DURATION_MS,
    sample_rate: int = BEEP_SAMPLE_RATE,
    volume: float = BEEP_VOLUME,
    output_path: Path | None = None,
) -> Path:
    """Generate a sine wave beep WAV file.

    Args:
        frequency: Beep frequency in Hz.
        duration_ms: Duration in milliseconds.
        sample_rate: Audio sample rate.
        volume: Volume (0.0 to 1.0).
        output_path: Where to save. Auto-generated if None.

    Returns:
        Path to the generated WAV file.
    """
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / "wake_focus_beep.wav"

    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

    # Generate sine wave with fade-in/out to avoid clicks
    signal = np.sin(2 * np.pi * frequency * t) * volume

    # Fade in/out (5ms each)
    fade_samples = int(sample_rate * 0.005)
    if fade_samples > 0 and num_samples > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out

    # Convert to 16-bit PCM
    pcm_data = (signal * 32767).astype(np.int16)

    # Write WAV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data.tobytes())

    logger.info(
        "Generated beep WAV: %s (freq=%dHz, dur=%dms, vol=%.2f)",
        output_path,
        frequency,
        duration_ms,
        volume,
    )
    return output_path


class AudioManager(QObject):
    """Manages alert beep playback with scheduling."""

    def __init__(
        self,
        beep_frequency: int = BEEP_FREQUENCY_HZ,
        beep_duration_ms: int = BEEP_DURATION_MS,
        beep_volume: float = BEEP_VOLUME,
        drowsy_interval_ms: int = 1000,
        distraction_interval_ms: int = 5000,
        distraction_beep_count: int = 2,
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        self._drowsy_interval = drowsy_interval_ms
        self._distraction_interval = distraction_interval_ms
        self._distraction_count = distraction_beep_count

        # Generate beep WAV
        self._beep_path = generate_beep_wav(
            frequency=beep_frequency,
            duration_ms=beep_duration_ms,
            volume=beep_volume,
        )

        # QSoundEffect for low-latency playback
        self._beep_effect = QSoundEffect(self)
        self._beep_effect.setSource(QUrl.fromLocalFile(str(self._beep_path)))
        self._beep_effect.setVolume(beep_volume)
        self._beep_effect.setLoopCount(1)

        # Drowsiness beep timer (continuous, every 1s)
        self._drowsy_timer = QTimer(self)
        self._drowsy_timer.setInterval(drowsy_interval_ms)
        self._drowsy_timer.timeout.connect(self._play_single_beep)

        # Distraction beep timer (2 beeps every 5s)
        self._distraction_timer = QTimer(self)
        self._distraction_timer.setInterval(distraction_interval_ms)
        self._distraction_timer.timeout.connect(self._play_double_beep)

        # Double-beep delay timer
        self._double_beep_timer = QTimer(self)
        self._double_beep_timer.setInterval(300)  # 300ms between beeps
        self._double_beep_timer.setSingleShot(True)
        self._double_beep_timer.timeout.connect(self._play_single_beep)

        logger.info(
            "AudioManager: beep=%dHz/%dms, drowsy_interval=%dms, distraction_interval=%dms",
            beep_frequency,
            beep_duration_ms,
            drowsy_interval_ms,
            distraction_interval_ms,
        )

    @Slot()
    def start_drowsy_beep(self) -> None:
        """Start continuous drowsiness beeping (<<ogohlantirish>>)."""
        self.stop_all()
        self._play_single_beep()
        self._drowsy_timer.start()
        logger.debug("Drowsy beep started")

    @Slot()
    def stop_drowsy_beep(self) -> None:
        """Stop drowsiness beeping."""
        self._drowsy_timer.stop()
        logger.debug("Drowsy beep stopped")

    @Slot()
    def start_distraction_beep(self) -> None:
        """Start distraction beep cycle (<<ogohlantirish2>>): 2 beeps every 5s."""
        self.stop_all()
        self._play_double_beep()
        self._distraction_timer.start()
        logger.debug("Distraction beep cycle started")

    @Slot()
    def stop_distraction_beep(self) -> None:
        """Stop distraction beep cycle."""
        self._distraction_timer.stop()
        self._double_beep_timer.stop()
        logger.debug("Distraction beep stopped")

    @Slot()
    def play_distraction_beep_once(self) -> None:
        """Play a single distraction event (2 beeps)."""
        self._play_double_beep()

    def stop_all(self) -> None:
        """Stop all beep timers."""
        self._drowsy_timer.stop()
        self._distraction_timer.stop()
        self._double_beep_timer.stop()

    @Slot()
    def _play_single_beep(self) -> None:
        """Play one beep sound."""
        if self._beep_effect.isLoaded():
            self._beep_effect.play()

    @Slot()
    def _play_double_beep(self) -> None:
        """Play two beeps in quick succession."""
        self._play_single_beep()
        self._double_beep_timer.start()  # Second beep after 300ms

    @property
    def beep_wav_path(self) -> Path:
        return self._beep_path
