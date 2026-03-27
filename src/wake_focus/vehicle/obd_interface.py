"""
Wake Focus - OBD-II Interface (Optional)

Connects to an ELM327 OBD-II adapter for real vehicle data.
Falls back gracefully when no adapter is available.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OBDInterface:
    """Optional OBD-II interface via ELM327 adapter."""

    def __init__(self, port: str = "", baud: int = 38400):
        self._port = port
        self._baud = baud
        self._connection = None
        self._available = False

        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt to connect to OBD-II adapter."""
        try:
            import obd

            if self._port:
                self._connection = obd.OBD(self._port, baudrate=self._baud)
            else:
                self._connection = obd.OBD()  # Auto-detect

            if self._connection.is_connected():
                self._available = True
                logger.info("OBD-II connected: %s", self._connection.port_name())
            else:
                logger.info("OBD-II adapter not found, vehicle stats will use GPS estimation")
        except ImportError:
            logger.info("python-obd not installed, OBD-II disabled")
        except Exception as e:
            logger.info("OBD-II connection failed: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    def get_speed(self) -> Optional[float]:
        """Get vehicle speed in km/h from OBD-II."""
        if not self._available or not self._connection:
            return None
        try:
            import obd

            response = self._connection.query(obd.commands.SPEED)
            if response and not response.is_null():
                return response.value.magnitude
        except Exception:
            pass
        return None

    def get_fuel_rate(self) -> Optional[float]:
        """Get fuel rate in L/h from OBD-II (if supported by vehicle)."""
        if not self._available or not self._connection:
            return None
        try:
            import obd

            response = self._connection.query(obd.commands.FUEL_RATE)
            if response and not response.is_null():
                return response.value.magnitude
        except Exception:
            pass
        return None

    def close(self) -> None:
        if self._connection:
            self._connection.close()
