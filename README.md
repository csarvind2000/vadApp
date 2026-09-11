# vadApp — Video Anomaly Detection

Weakly-supervised video anomaly detection on UCF-Crime, with a deployed
application, a trained model, and the manuscript describing the work.

| | |
|---|---|
| Application | React + Django + Postgres/Redis, with a GPU model service |
| Detector | ATA-HS — adaptive sparse temporal attention over I3D features |
| Object overlay | YOLO11n, person/vehicle boxes drawn on the player |
| Manuscript | [`manuscript/main.pdf`](manuscript/main.pdf) |
| Walkthrough | [YouTube](https://www.youtube.com/watch?v=Hym006TV5zY) |
| Demo and sample videos | [Google Drive](https://drive.google.com/drive/folders/17vvr6t2FfYsheeEf5cHbT_g6Jkda2po3?usp=drive_link) |

## Video

A walkthrough of the application is on
[YouTube](https://www.youtube.com/watch?v=Hym006TV5zY).

The annotated demo video and the sample clips used to test the application are on
[Google Drive](https://drive.google.com/drive/folders/17vvr6t2FfYsheeEf5cHbT_g6Jkda2po3?usp=drive_link).
They are hosted there rather than in this repository because the footage comes
from UCF-Crime, which is distributed for research use by UCF CRCV and does not
grant redistribution rights.

`demo.mp4` shows every detected interval annotated with the anomaly score, the
predicted category and the object boxes. It includes a missed detection, so the
failure mode is visible rather than edited out.

The sample clips cover three cases: videos the detector finds, videos it misses,
and normal footage that triggers a false alarm. Expected outcomes for each are
listed in `sample_videos/README.md`, and `tools/make_input_video.py` in the
research repository rebuilds the merged test clip from the original dataset.

## Results

Controlled comparison on the UCF-Crime test set. All methods use the same I3D
features, the same official split and the same three seeds.

| Method | AUC | *p* | AUC-A | FAR@0.5 |
|---|---|---|---|---|
| **ATA-HS (proposed)** | 0.761 | — | **0.582** | **0.017** |
| MIL ranking (Sultani et al.) | 0.777 | 0.74 | 0.580 | 0.023 |
| Dense self-attention | 0.760 | 0.37 | 0.562 | 0.028 |
| Magnitude selection (RTFM-style) | 0.695 | 0.16 | 0.549 | 0.063 |
| Mean pooling (cannot localise) | 0.775 | 0.96 | 0.537 | 0.304 |

No pair of methods is separated on AUC: every paired video-level bootstrap gives
*p* > 0.16, and mean pooling — which emits one score per video and cannot localise
at all — is indistinguishable from every detector at *p* = 0.96. The measures that
do separate them are AUC restricted to anomalous videos and the false-alarm rate,
where the proposed model leads.

## Quick start

```bash
cd src
cp .env.example .env          # set DJANGO_SECRET_KEY and POSTGRES_PASSWORD
docker compose build
docker compose up -d
```

Frontend <http://localhost:5174> · API <http://localhost:8001/api/>

Note: use `docker compose` (v2 plugin), not the older `docker-compose` binary;
the two name images differently and will start different containers.

Upload one of the sample clips from the
[Google Drive folder](https://drive.google.com/drive/folders/17vvr6t2FfYsheeEf5cHbT_g6Jkda2po3?usp=drive_link);
the expected outcome for each, including the failure cases, is listed alongside
them.


Web application for surveillance video anomaly detection. Users upload a video or
open a live camera feed; the ATA-HS detector scores the footage snippet by
snippet, and stretches scoring above a threshold are saved as **incidents** with
a thumbnail, a score curve, and a reviewer verdict.

The stack is a React (Vite) + MUI frontend, a Django REST backend, Postgres,
Redis/RQ workers, and a separate GPU model service.

## Architecture

| Service | Role |
|---|---|
| `frontend` | React (Vite) + MUI web UI, served by nginx |
| `backend` | Django REST Framework API — videos, jobs, incidents, cameras, streams |
| `postgres` | Relational database |
| `redis` | Queue backing store for RQ |
| `ai_worker` | RQ worker running analysis jobs (CPU only) |
| `vad_model` | Flask API serving I3D + ATA-HS + the category head (**the only GPU container**) |

Only `vad_model` touches the GPU. The backend and workers call it over HTTP, so
the model is loaded once and shared rather than re-instantiated per job. Keeping
inference in a single long-lived process avoids paying the model-load cost on
every job and bounds GPU memory to one copy of the network, regardless of how
many workers are running.

The research package (`src/ucfcrime/`) is **mounted read-only** into the model
container rather than copied. The app and the experiments therefore share one
definition of the architecture; there is no second implementation to drift out
of sync with the published results.

## Data flow

**Offline (upload)**
1. Browser uploads a video → `POST /api/videos/`
2. `POST /api/videos/{id}/analyze/` creates an `AnalysisJob` and enqueues it
3. An `ai_worker` picks it up and calls `vad_model` `/analyze`
4. The model decodes the video, extracts 16-frame I3D snippets, scores them with
   ATA-HS, and returns the score curve plus incident intervals
5. The worker saves incidents and grabs a thumbnail at each peak frame
6. The UI polls the job and renders the score timeline; clicking seeks the player

**Live (camera)**
1. `POST /api/streams/` opens a session on the model service
2. The browser posts JPEG frames at ~10 fps to `POST /api/streams/{id}/frame/`
3. Every 16 frames the model computes a feature, appends it to a rolling context
   window, and scores the window — so the temporal model still sees history
   rather than judging each snippet alone
4. `POST /api/streams/{id}/close/` returns the session summary; incidents are
   persisted into the same table as offline detections

A frame is skipped if the previous one is still in flight, so a slow GPU
produces dropped frames rather than an unbounded request backlog.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health/` | Backend + model service status |
| GET/POST | `/api/videos/` | List / upload videos |
| POST | `/api/videos/{id}/analyze/` | Queue analysis (returns job id) |
| GET | `/api/jobs/{id}/` | Job status, score curve, incidents |
| GET/PATCH | `/api/incidents/` | Incident register; set review verdict |
| GET/POST | `/api/cameras/` | Manage camera records |
| POST | `/api/streams/` | Open a live session |
| POST | `/api/streams/{id}/frame/` | Push one frame |
| POST | `/api/streams/{id}/close/` | Close and persist incidents |

## Getting started

```bash
cd src
cp .env.example .env          # set DJANGO_SECRET_KEY, POSTGRES_*, MODELS_DIR

# Put the trained checkpoint where the model service expects it
mkdir -p models && cp ../../runs/<your_run>/best.pt models/best.pt

docker compose build
docker compose up -d
docker compose exec backend python manage.py createsuperuser
```

- Frontend: <http://localhost:5174>
- API: <http://localhost:8001/api/>
- Django admin: <http://localhost:8001/admin/>
- RQ dashboard: <http://localhost:8001/django-rq/>

Ports are deliberately offset (5174/8001/5434/6381/5001) so this stack can run
alongside other services on the same host without colliding.

### Running without Docker (development)

```bash
# backend
cd src/WebGUI/backend/src
DEBUG=True python manage.py migrate
DEBUG=True python manage.py runserver 8000

# model service (needs the GPU and the cuDNN fix from the research repo)
cd ../../../vad_model
source ../../../../env.sh
RESEARCH_ROOT=$(realpath ../../..) VAD_CHECKPOINT=/path/to/best.pt python app.py

# frontend
cd ../WebGUI/frontend && npm install && npm run dev
```

The backend falls back to SQLite when `POSTGRES_HOST` is unset, so it runs
standalone without the database container.

## Anomaly category classification

Alongside the binary detector, an optional head ranks the 13 UCF-Crime anomaly
categories. It loads from `src/models/category.pt` if present; the detector runs
normally without it, and `/api/health/` reports `category_head: true|false`.

The head is trained on the 810 anomalous training videos, which carry a
video-level category label but no temporal annotation, so its output describes
the video as a whole rather than any single moment.

**Read the output as a ranking, not a label.** Measured on the 140 anomalous
test videos:

| Metric | Value |
|---|---|
| top-1 accuracy | 0.379 |
| top-3 accuracy | 0.629 |
| chance (13 classes) | 0.077 |

Roughly five times chance, and wrong at rank 1 for about three videos in five.
Per-class recall is very uneven — RoadAccidents 0.74 and Arson 0.56, against
Shooting 0.13 and Abuse, Arrest and Assault at 0.00. Two causes: the training
and test category balances differ sharply (Robbery 145/5, Shooting 27/23), and
several categories are near-indistinguishable in surveillance footage because
the label encodes intent rather than appearance (Robbery, Stealing, Burglary).

The UI therefore shows the top three with probabilities and states the accuracy
inline. It is a filter for review, not evidence.

To retrain:

```bash
python -m src.ucfcrime.train_classifier \
    --feature_dir /path/to/features/i3d_r50_c1 --out_dir runs/category
cp runs/category/category.pt vadapp/src/models/category.pt
```

## Debug and development mode

`docker-compose.dev.yaml` bind-mounts the source over the copy baked into the
image, so edits to `settings.py`, `views.py` or the model service take effect on
a restart instead of a rebuild.

```bash
cd src

# start the whole stack in dev mode
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d

# after editing backend or model-service code -- no rebuild needed
docker compose restart backend ai_worker vad_model
```

To run with Django debug output (verbose tracebacks in API responses, SQL echo):

```bash
sed -i 's/^DEBUG=False/DEBUG=True/' .env
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d backend ai_worker
```

Run in the foreground to watch every service log live, which is the quickest way
to see a container fail at startup:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

### Logs and shells

```bash
docker compose logs -f backend          # follow one service
docker compose logs -f vad_model        # GPU service: model load, inference errors
docker compose logs --tail=100 ai_worker
docker compose ps                       # container state and health

docker compose exec backend python manage.py shell    # Django shell
docker compose exec backend bash                      # shell in the container
docker compose exec postgres psql -U vadapp vadapp    # database
```

### Rebuilding

The production compose file copies `src/` into the image at build time, so code
changes require a rebuild unless you are using the dev override above.

```bash
docker compose up -d --build backend ai_worker      # rebuild changed services
docker compose build --no-cache backend             # force a clean rebuild
docker compose down && docker compose up -d --build # full restart
```

### Health checks

```bash
curl -s localhost:8001/api/health/ | python3 -m json.tool   # backend + model status
curl -s localhost:5001/health      | python3 -m json.tool   # model service directly
curl -s localhost:5174/api/health/                          # through the nginx proxy
```

A healthy stack returns `{"backend": "ok", "model_service": {"status": "ok", ...}}`
and the header chip in the UI reads *model ready*.

## Troubleshooting

**`container for service "postgres" is unhealthy`, log says `POSTGRES_PASSWORD` not
specified.** `.env` is missing, so Compose substituted empty values. Copy
`.env.example` to `.env` and fill it in. Postgres has already created a
half-initialised volume at this point, and restarting will not repair it; the
volume must be removed:

```bash
docker compose down -v          # -v is required: it drops the bad volume
docker compose up -d --build
```

**API returns 401, UI header shows "backend unreachable".** The frontend has no
login flow, so the API must be open for it to work. Check `VAD_REQUIRE_AUTH=False`
in `.env`. If it is already `False` and you still get 401, the container is
running a stale image from before this setting existed -- rebuild:

```bash
docker compose build --no-cache backend ai_worker
docker compose up -d backend ai_worker
```

**Code changes have no effect.** The image copies `src/` at build time. Either
rebuild, or use the dev override, which bind-mounts the source.

**`/health` returns `model_unavailable`.** The checkpoint is missing or does not
match the backbone. Confirm `src/models/best.pt` exists and that `VAD_BACKBONE`
matches the backbone the checkpoint was trained with; the engine refuses to load
a checkpoint whose feature dimension differs, rather than producing silently
wrong scores.

**GPU not visible in `vad_model`.** Requires the NVIDIA Container Toolkit. Verify
with `docker compose exec vad_model python -c "import torch; print(torch.cuda.is_available())"`.

**Port already in use.** Ports are offset (5174, 8001, 5001, 5434, 6381) to coexist
with other stacks on the same host. Change the host side of the mapping in
`docker-compose.yaml` if one still collides.

## Notes

**The checkpoint and the backbone must match.** `VADEngine` refuses to load a
checkpoint whose feature dimension differs from the backbone's output, because
silently mismatched features produce confident nonsense rather than an error.

**Thresholds are recorded, not assumed.** Every `AnalysisJob` stores the
threshold and minimum duration used, so an incident can always be traced back to
the settings that produced it, and a video can be re-analysed at a new threshold
without destroying earlier results.

**Incidents merge short dips.** A brief drop below threshold mid-event does not
split one incident into two; runs shorter than the minimum duration are dropped
as noise. Both parameters are per-job.
