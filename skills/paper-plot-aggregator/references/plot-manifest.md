# Plot Manifest Schema

Use a manifest to track provenance and suggested captions for each saved figure.

## Suggested JSON format
```json
{
  "generated_at": "2026-02-04T12:34:56",
  "figures_dir": "figures",
  "figures": [
    {
      "filename": "training loss.png",
      "title": "Training loss vs epoch",
      "caption_suggestion": "Loss decreases monotonically and plateaus after ...",
      "data_sources": [
        "experiment_results/foo.npy",
        "results/metrics.json"
      ],
      "notes": "Any caveats, preprocessing, or filtering used."
    }
  ]
}
```

## Guardrails
- `data_sources` should point to real files.
- Captions must not claim results not supported by the data.
