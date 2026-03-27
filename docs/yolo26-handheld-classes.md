# YOLO26 Handheld / Paper Proxy Classes

This project now ships with local YOLO26 model files:

- `models/yolo26n.pt`
- `models/yolo26n.onnx`

The local model was introspected from the copied weights and exposes the standard
80 COCO classes. For Wake Focus distraction monitoring, the runtime filter now
uses this extracted subset:

| YOLO26 class | Wake Focus meaning |
|---|---|
| `cell phone` | phone / mobile phone / smartphone |
| `book` | paper / document / notebook proxy |
| `laptop` | handheld or lap-held computing device |
| `mouse` | small handheld computer accessory |
| `remote` | remote control / handheld controller |
| `keyboard` | portable or handheld keyboard interaction |

Why `paper` maps to `book`:

- The copied YOLO26 COCO model does not expose a native `paper` class.
- For the current baseline, `paper`, `document`, and `notebook` aliases are
  routed to `book` so the detector can still flag reading/document-like behavior.
- If precise paper detection is required, the next step is a fine-tuned custom
  model using the existing `training/` pipeline.
