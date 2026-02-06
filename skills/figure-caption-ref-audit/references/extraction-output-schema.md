# Extraction Output Schema

The extractor writes:
- `figures.json`
- `images/*.png`

## figures.json format
```json
{
  "pdf": "/abs/path/to/paper.pdf",
  "generated_at": "2026-02-04T12:34:56",
  "images_dir": "images",
  "figures": [
    {
      "label": "1",
      "page": 3,
      "caption": "Figure 1: ...",
      "caption_bbox": [x0, y0, x1, y1],
      "figure_bbox": [x0, y0, x1, y1],
      "image_path": "images/figure_1_p3.png",
      "figrefs": [
        {"page": 2, "text": "As shown in Figure 1, ..."}
      ],
      "notes": "any extraction warnings"
    }
  ]
}
```

## Coordinate system
- Bounding boxes are in PDF page coordinates from PyMuPDF (float values).
