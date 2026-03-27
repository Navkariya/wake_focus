"""
Wake Focus - UI Styles

Global dark theme stylesheet and style constants.
Premium design with modern typography, gradients, and glassmorphism effects.
"""

# ── Color Palette ───────────────────────────────────────────────────────────
BG_PRIMARY = "#0a0e17"       # Deep navy-black
BG_SECONDARY = "#111827"     # Slightly lighter navy
BG_PANEL = "#1a1f2e"         # Panel background
BG_CARD = "#1e2538"          # Card/widget background
BG_HOVER = "#2a3148"         # Hover state
BG_INPUT = "#0d1117"         # Input field background

ACCENT_PRIMARY = "#3b82f6"   # Electric blue
ACCENT_SECONDARY = "#06b6d4" # Cyan
ACCENT_SUCCESS = "#10b981"   # Emerald green
ACCENT_WARNING = "#f59e0b"   # Amber
ACCENT_DANGER = "#ef4444"    # Red
ACCENT_PURPLE = "#8b5cf6"    # Purple

TEXT_PRIMARY = "#f1f5f9"     # Almost white
TEXT_SECONDARY = "#94a3b8"   # Muted silver
TEXT_MUTED = "#64748b"       # Very muted
TEXT_LABEL = "#cbd5e1"       # Label text

BORDER_DEFAULT = "#2d3748"   # Subtle border
BORDER_ACTIVE = "#3b82f6"    # Active element border
BORDER_RADIUS = "8px"        # Standard border radius

# ── Typography ──────────────────────────────────────────────────────────────
FONT_FAMILY = "'Inter', 'Segoe UI', 'Roboto', system-ui, sans-serif"
FONT_SIZE_XS = "10px"
FONT_SIZE_SM = "11px"
FONT_SIZE_BASE = "13px"
FONT_SIZE_LG = "15px"
FONT_SIZE_XL = "18px"
FONT_SIZE_2XL = "22px"

# ── Global Stylesheet ──────────────────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
    /* ── Base ─────────────────────────────────────────────────── */
    QMainWindow {{
        background-color: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BASE};
    }}

    QWidget {{
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
    }}

    /* ── Panels ───────────────────────────────────────────────── */
    QFrame#cameraPanel {{
        background-color: #000000;
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    QFrame#mapPanel {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    QFrame#vehicleStatsPanel {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    QFrame#fleetStatusPanel {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    QFrame#buttonPanel {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    /* ── Buttons ──────────────────────────────────────────────── */
    QPushButton {{
        background-color: {BG_CARD};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 6px;
        padding: 8px 16px;
        font-size: {FONT_SIZE_BASE};
        font-weight: 500;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {BG_HOVER};
        border-color: {ACCENT_PRIMARY};
    }}

    QPushButton:pressed {{
        background-color: {ACCENT_PRIMARY};
        border-color: {ACCENT_PRIMARY};
    }}

    QPushButton#exitButton {{
        background-color: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.3);
        color: {ACCENT_DANGER};
    }}

    QPushButton#exitButton:hover {{
        background-color: rgba(239, 68, 68, 0.3);
        border-color: {ACCENT_DANGER};
    }}

    QPushButton#settingsButton {{
        background-color: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.3);
        color: {ACCENT_PRIMARY};
    }}

    QPushButton#settingsButton:hover {{
        background-color: rgba(59, 130, 246, 0.3);
        border-color: {ACCENT_PRIMARY};
    }}

    QPushButton#profileButton {{
        background-color: rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.3);
        color: {ACCENT_PURPLE};
    }}

    QPushButton#profileButton:hover {{
        background-color: rgba(139, 92, 246, 0.3);
        border-color: {ACCENT_PURPLE};
    }}

    QPushButton#startStopButton {{
        background-color: rgba(16, 185, 129, 0.2);
        border-color: rgba(16, 185, 129, 0.4);
        color: {ACCENT_SUCCESS};
        font-weight: 600;
    }}

    QPushButton#startStopButton:hover {{
        background-color: rgba(16, 185, 129, 0.4);
        border-color: {ACCENT_SUCCESS};
    }}

    /* ── Labels ───────────────────────────────────────────────── */
    QLabel {{
        color: {TEXT_PRIMARY};
        font-size: {FONT_SIZE_BASE};
    }}

    QLabel#panelTitle {{
        color: {TEXT_PRIMARY};
        font-size: {FONT_SIZE_LG};
        font-weight: 600;
        padding: 4px 0px;
        border-bottom: 1px solid {BORDER_DEFAULT};
        margin-bottom: 4px;
    }}

    QLabel#statValue {{
        color: {ACCENT_PRIMARY};
        font-size: {FONT_SIZE_2XL};
        font-weight: 700;
    }}

    QLabel#statLabel {{
        color: {TEXT_SECONDARY};
        font-size: {FONT_SIZE_SM};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    QLabel#statusIndicator {{
        font-size: {FONT_SIZE_SM};
        padding: 2px 8px;
        border-radius: 10px;
    }}

    /* ── Scroll Area ──────────────────────────────────────────── */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}

    QScrollBar:vertical {{
        background-color: {BG_PANEL};
        width: 8px;
        border-radius: 4px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {BORDER_DEFAULT};
        border-radius: 4px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {TEXT_MUTED};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* ── List Widget ──────────────────────────────────────────── */
    QListWidget {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        color: {TEXT_PRIMARY};
        font-size: {FONT_SIZE_SM};
        padding: 2px;
    }}

    QListWidget::item {{
        padding: 4px 6px;
        border-radius: 3px;
    }}

    QListWidget::item:selected {{
        background-color: rgba(59, 130, 246, 0.2);
    }}

    /* ── Dialog ───────────────────────────────────────────────── */
    QDialog {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: {FONT_SIZE_BASE};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {ACCENT_PRIMARY};
    }}

    QCheckBox {{
        color: {TEXT_PRIMARY};
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {BORDER_DEFAULT};
        background-color: {BG_INPUT};
    }}

    QCheckBox::indicator:checked {{
        background-color: {ACCENT_PRIMARY};
        border-color: {ACCENT_PRIMARY};
    }}

    QGroupBox {{
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        color: {TEXT_SECONDARY};
        subcontrol-origin: margin;
        padding: 0 8px;
    }}

    /* ── Tab Widget ───────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        background-color: {BG_PANEL};
    }}

    QTabBar::tab {{
        background-color: {BG_CARD};
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER_DEFAULT};
        padding: 6px 12px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        background-color: {BG_PANEL};
        color: {ACCENT_PRIMARY};
        border-bottom: 2px solid {ACCENT_PRIMARY};
    }}
"""


# ── Panel title helpers ─────────────────────────────────────────────────────
def make_panel_title_html(icon: str, text: str) -> str:
    """Create an HTML formatted panel title with icon."""
    return (
        f'<span style="font-size:14px; font-weight:600; color:{TEXT_PRIMARY};">'
        f'{icon} {text}</span>'
    )
