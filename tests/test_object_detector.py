"""Tests for YOLO26 object-detector class filtering."""

from wake_focus.ml.object_detector import ObjectDetector


def make_detector(target_classes=None):
    detector = ObjectDetector.__new__(ObjectDetector)
    detector._target_classes = target_classes or [
        "cell phone",
        "phone",
        "paper",
        "document",
        "book",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "electronic gadget",
    ]
    detector._normalized_target_classes = detector._expand_target_classes(detector._target_classes)
    return detector


def test_phone_alias_matches_cell_phone():
    detector = make_detector()
    assert detector._is_target_class("cell phone", 67)


def test_paper_alias_maps_to_book():
    detector = make_detector(["paper"])
    assert detector._is_target_class("book", 73)


def test_electronic_gadget_alias_matches_remote():
    detector = make_detector(["electronic gadget"])
    assert detector._is_target_class("remote", 65)


def test_non_target_class_is_filtered_out():
    detector = make_detector(["cell phone"])
    assert detector._is_target_class("banana", 46) is False
