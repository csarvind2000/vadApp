# Sample videos for testing the application

Nine videos from the **UCF-Crime test set** — none was seen during training, so
what the model does with them is genuine held-out behaviour.

Upload at <http://localhost:5174> → Upload → choose file → Upload and analyze.

Expected outcomes are taken from the offline research pipeline, so the app should
reproduce them closely. Peak score is at threshold 0.50.

## Should detect (use these for a demonstration)

| File | Ground truth | Peak | Expect |
|---|---|---|---|
| `RoadAccidents001_x264.mp4` | RoadAccidents | 1.000 | incidents; category usually correct (this class has the best recall, 0.74) |
| `Shooting002_x264.mp4` | Shooting | 1.000 | incidents; category often wrong — Shooting recall is 0.13 |
| `Explosion004_x264.mp4` | Explosion | 1.000 | incidents; category fair (recall 0.48) |
| `Arson035_x264.mp4` | Arson | 1.000 | incidents; category fair (recall 0.56) |
| `Stealing058_x264.mp4` | Stealing | 1.000 | incidents, and among the best temporal localisation in the test set (IoU 0.64) |

## Should miss (the model's real failure mode)

| File | Ground truth | Peak | Expect |
|---|---|---|---|
| `Explosion025_x264.mp4` | Explosion | 0.049 | **zero incidents** |
| `Shoplifting017_x264.mp4` | Shoplifting | 0.088 | **zero incidents** |

These are not broken uploads. 63 of the 140 anomalous test videos (45%) peak
below 0.50 and produce no incident at the default threshold. Lower the threshold
on the Upload page, or use Re-analyse on the video page, to see the score curve
rise without any interval crossing 0.50.

## Normal videos

| File | Ground truth | Peak | Expect |
|---|---|---|---|
| `Normal_Videos_901_x264.mp4` | Normal | 1.000 | **false alarm** — incidents on footage with no annotated anomaly |
| `Normal_Videos_717_x264.mp4` | Normal | 0.000 | correctly silent, no incidents |

## What to look at

- **Score curve** with the moving playhead and the live score chip during playback
- **Object boxes** overlaid on the player, toggleable, with per-class counts
- **Per-incident categories** in the incident table, shown as a ranked top-2
- **Incident thumbnails** captured at the peak frame
- **Re-analyse** to re-run at a different threshold without re-uploading

## Source

`/media/ranjhaa-local/volume23/LLM/UCF-Crime/Videos`. Any of the 290 test videos
can be used; these were chosen to cover detection, missed detection and false
alarm rather than only the cases that work.
