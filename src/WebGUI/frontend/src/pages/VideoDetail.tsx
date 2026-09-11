import {
  Alert, Box, Button, Card, CardContent, Chip, FormControlLabel, Grid,
  LinearProgress, Slider, Stack, Switch, Table, TableBody, TableCell, TableHead,
  TableRow, Tooltip, Typography,
} from '@mui/material'
import RefreshIcon from '@mui/icons-material/Refresh'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import BoxOverlay, { type Detections } from '../components/BoxOverlay'
import ScoreTimeline from '../components/ScoreTimeline'
import { api, type AnalysisJob, type Video } from '../services/api'

export default function VideoDetail() {
  const { id } = useParams<{ id: string }>()
  const [video, setVideo] = useState<Video | null>(null)
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const player = useRef<HTMLVideoElement>(null)
  const [now, setNow] = useState(0)      // playback position, seconds
  const [rerunThr, setRerunThr] = useState(0.5)
  const [rerunning, setRerunning] = useState(false)
  const [showBoxes, setShowBoxes] = useState(true)
  const [vsize, setVsize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    if (!id) return
    let stop = false
    const tick = async () => {
      const v = await api.getVideo(id)
      if (stop) return
      setVideo(v)
      if (v.latest_job) {
        const full = await api.getJob(v.latest_job.id)
        if (!stop) setJob(full)
        // Keep polling only while the job is unfinished.
        if (['queued', 'running'].includes(full.status)) setTimeout(tick, 2000)
      }
    }
    tick()
    return () => { stop = true }
  }, [id])

  const rerun = async () => {
    if (!id) return
    setRerunning(true)
    try {
      await api.analyzeVideo(id, rerunThr)
      // Poll until the new job leaves the queue, then refresh the view.
      for (let i = 0; i < 120; i++) {
        const v = await api.getVideo(id)
        const j = v.latest_job ? await api.getJob(v.latest_job.id) : null
        setVideo(v); if (j) setJob(j)
        if (j && !['queued', 'running'].includes(j.status)) break
        await new Promise(r => setTimeout(r, 2000))
      }
    } finally { setRerunning(false) }
  }

  const seek = useCallback((sec: number) => {
    if (player.current) { player.current.currentTime = sec; player.current.play() }
  }, [])

  // Score at the current playback position. Snippet i covers
  // [i*secPerSnippet, (i+1)*secPerSnippet), so the index is a floor.
  const scoreAt = (sec: number, scores?: number[] | null, per?: number) => {
    if (!scores?.length || !per) return null
    const i = Math.min(scores.length - 1, Math.max(0, Math.floor(sec / per)))
    return scores[i]
  }

  if (!video) return <LinearProgress />

  const fps = video.fps || 30
  const secPerSnippet = 16 / fps
  const incidents = job?.incidents ?? []
  const liveScore = scoreAt(now, job?.snippet_scores, secPerSnippet)
  const det = (job?.detections as Detections) ?? null
  const counts = Object.entries(det?.class_counts ?? {}).slice(0, 5)
  // Incident currently under the playhead, used to highlight its row.
  const activeIncident = incidents.find(i => now >= i.start_sec && now <= i.end_sec)

  return (
    <Stack spacing={2}>
      <Card>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <Typography variant="h6">{video.name}</Typography>
            {job && <Chip size="small" label={job.status}
                          color={job.status === 'done' ? 'success'
                            : job.status === 'failed' ? 'error' : 'warning'} />}
            {job?.model_name && <Chip size="small" variant="outlined"
                                      label={`${job.model_name} / ${job.backbone}`} />}
            {job?.max_score != null &&
              <Chip size="small" variant="outlined" label={`peak ${job.max_score.toFixed(3)}`} />}
            <Box flexGrow={1} />
            <Chip size="small" color={incidents.length ? 'error' : 'default'}
                  label={`${incidents.length} incident${incidents.length === 1 ? '' : 's'}`} />
          </Stack>

          {job?.categories?.length ? (
            <Box sx={{ mt: 2 }}>
              <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                <Tooltip title="Ranked over 13 categories. Top-1 0.38, top-3 0.63, chance 0.08; accuracy varies by category.">
                  <Typography variant="body2" color="text.secondary"
                              sx={{ borderBottom: '1px dotted', borderColor: 'divider', cursor: 'help' }}>
                    Likely category
                  </Typography>
                </Tooltip>
                {job.categories.map((c, i) => (
                  <Chip key={c.category} size="small"
                        variant={i === 0 ? 'filled' : 'outlined'}
                        color={i === 0 ? 'warning' : 'default'}
                        label={`${c.category} ${(c.probability * 100).toFixed(0)}%`} />
                ))}
              </Stack>

            </Box>
          ) : job?.status === 'done' ? (
            <Alert severity="info" sx={{ mt: 2 }} icon={false}>
              <Typography variant="body2">
                No category prediction for this run. The run predates the
                category classifier, or found no incident to classify. Re-analyse
                to produce one.
              </Typography>
            </Alert>
          ) : null}

          {job?.status === 'done' && (
            <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }}>
              <Box sx={{ width: 200 }}>
                <Typography variant="caption" color="text.secondary">
                  re-analyse at threshold {rerunThr.toFixed(2)}
                </Typography>
                <Slider size="small" value={rerunThr} min={0.05} max={0.95} step={0.05}
                        disabled={rerunning}
                        onChange={(_, v) => setRerunThr(v as number)} />
              </Box>
              <Tooltip title="Run the current model over this video again. Earlier results are replaced.">
                <span>
                  <Button size="small" variant="outlined" startIcon={<RefreshIcon />}
                          disabled={rerunning} onClick={rerun}>
                    {rerunning ? 'Analysing…' : 'Re-analyse'}
                  </Button>
                </span>
              </Tooltip>
            </Stack>
          )}
          {job?.status === 'failed' && <Alert severity="error" sx={{ mt: 2 }}>{job.error}</Alert>}
          {job && ['queued', 'running'].includes(job.status) &&
            <Box sx={{ mt: 2 }}><LinearProgress /><Typography variant="caption">
              analyzing…</Typography></Box>}
        </CardContent>
      </Card>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card><CardContent>
            {video.file_url ? (
              <>
                <Box sx={{ position: 'relative', lineHeight: 0 }}>
                  <video ref={player} src={video.file_url} controls
                         onTimeUpdate={e => setNow(e.currentTarget.currentTime)}
                         onSeeked={e => setNow(e.currentTarget.currentTime)}
                         onLoadedMetadata={e => setVsize({
                           w: e.currentTarget.clientWidth, h: e.currentTarget.clientHeight })}
                         style={{ width: '100%', borderRadius: 8, background: '#000' }} />
                  {showBoxes && (
                    <BoxOverlay detections={(job?.detections as Detections) ?? null}
                                time={now} width={vsize.w} height={vsize.h} />
                  )}
                </Box>
                <Stack direction="row" alignItems="center" spacing={2} sx={{ mt: 1 }}>
                  <FormControlLabel
                    control={<Switch size="small" checked={showBoxes}
                                     onChange={e => setShowBoxes(e.target.checked)} />}
                    label={<Typography variant="caption">object boxes</Typography>} />
                  {counts.length > 0 && (
                    <Stack direction="row" spacing={0.5} flexWrap="wrap">
                      {counts.map(([k, n]) => (
                        <Chip key={k} size="small" variant="outlined"
                              label={`${k} ${n}`} />
                      ))}
                    </Stack>
                  )}
                </Stack>
              </>
            ) : <Typography color="text.secondary">video unavailable</Typography>}
          </CardContent></Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card><CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Typography variant="subtitle2">Anomaly score over time</Typography>
              <Typography variant="caption" color="text.secondary">click to seek</Typography>
              <Box flexGrow={1} />
              {liveScore != null && (
                <>
                  <Chip size="small" variant="outlined"
                        label={`t ${now.toFixed(1)}s`} />
                  <Chip size="small"
                        color={liveScore > (job?.threshold ?? 0.5) ? 'error' : 'default'}
                        label={liveScore > (job?.threshold ?? 0.5)
                          ? `ANOMALY ${liveScore.toFixed(3)}`
                          : `score ${liveScore.toFixed(3)}`} />
                </>
              )}
            </Stack>
            {job?.snippet_scores?.length
              ? <ScoreTimeline scores={job.snippet_scores} secondsPerPoint={secPerSnippet}
                               threshold={job.threshold} incidents={incidents}
                               onSeek={seek} playhead={now} />
              : <Typography color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
                  no scores yet</Typography>}
          </CardContent></Card>
        </Grid>
      </Grid>

      <Card><CardContent>
        <Typography variant="subtitle2" gutterBottom>Detected incidents</Typography>
        {incidents.length === 0
          ? <Typography color="text.secondary" sx={{ py: 3, textAlign: 'center' }}>
              None above threshold.</Typography>
          : <Table size="small">
              <TableHead><TableRow>
                <TableCell>Preview</TableCell><TableCell>Start</TableCell>
                <TableCell>End</TableCell><TableCell>Duration</TableCell>
                <TableCell>Likely category</TableCell>
                <TableCell align="right">Peak</TableCell><TableCell align="right">Mean</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {incidents.map(inc => (
                  <TableRow key={inc.id} hover sx={{
                              cursor: 'pointer',
                              bgcolor: activeIncident?.id === inc.id
                                ? 'rgba(255,82,82,0.14)' : undefined,
                            }}
                            onClick={() => seek(inc.start_sec)}>
                    <TableCell>{inc.thumbnail_url
                      ? <img src={inc.thumbnail_url} alt="" width={96}
                             style={{ borderRadius: 4, display: 'block' }} />
                      : '—'}</TableCell>
                    <TableCell>{inc.start_sec.toFixed(1)}s</TableCell>
                    <TableCell>{inc.end_sec.toFixed(1)}s</TableCell>
                    <TableCell>{inc.duration_sec.toFixed(1)}s</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap">
                        {(inc.categories ?? []).slice(0, 2).map((c, k) => (
                          <Chip key={c.category} size="small"
                                variant={k === 0 ? 'filled' : 'outlined'}
                                color={k === 0 ? 'warning' : 'default'}
                                label={`${c.category} ${(c.probability * 100).toFixed(0)}%`} />
                        ))}
                        {!inc.categories?.length && (
                          <Typography variant="caption" color="text.secondary">—</Typography>)}
                      </Stack>
                    </TableCell>
                    <TableCell align="right">
                      <Chip size="small" color="error" label={inc.peak_score.toFixed(3)} />
                    </TableCell>
                    <TableCell align="right">{inc.mean_score.toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>}
      </CardContent></Card>
    </Stack>
  )
}
