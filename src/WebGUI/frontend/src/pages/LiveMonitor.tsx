import {
  Alert, Box, Button, Card, CardContent, Chip, FormControl, Grid, InputLabel,
  MenuItem, Select, Slider, Stack, Table, TableBody, TableCell, TableHead,
  TableRow, Typography,
} from '@mui/material'
import { useEffect, useRef, useState } from 'react'
import ScoreTimeline from '../components/ScoreTimeline'
import { DEFAULT_THRESHOLD, LIVE_SEND_FPS } from '../constants'
import { api, type Camera } from '../services/api'

interface LiveIncident {
  start_sec: number; end_sec: number; duration_sec: number
  peak_score: number; mean_score: number
}

export default function LiveMonitor() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sessionRef = useRef<string | null>(null)
  const timerRef = useRef<number | null>(null)
  const inFlight = useRef(false)

  const [cameras, setCameras] = useState<Camera[]>([])
  const [cameraId, setCameraId] = useState<string>('')
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [running, setRunning] = useState(false)
  const [score, setScore] = useState<number | null>(null)
  const [scores, setScores] = useState<number[]>([])
  const [incidents, setIncidents] = useState<LiveIncident[]>([])
  const [err, setErr] = useState('')

  useEffect(() => {
    api.listCameras().then(d => setCameras(d.results)).catch(() => {})
    return () => { stop() }   // always release the webcam on unmount
  }, [])

  const grabFrame = async (): Promise<Blob | null> => {
    const v = videoRef.current, c = canvasRef.current
    if (!v || !c || v.readyState < 2) return null
    // Downscale before sending: the model crops to 224 anyway, so shipping full
    // resolution would only cost bandwidth.
    const w = 320, h = Math.round((v.videoHeight / v.videoWidth) * w) || 240
    c.width = w; c.height = h
    c.getContext('2d')!.drawImage(v, 0, 0, w, h)
    return new Promise(res => c.toBlob(b => res(b), 'image/jpeg', 0.8))
  }

  const tick = async () => {
    // Skip if the previous frame is still in flight, so a slow GPU cannot
    // build an unbounded backlog of requests.
    if (!sessionRef.current || inFlight.current) return
    const blob = await grabFrame()
    if (!blob) return
    inFlight.current = true
    try {
      const r = await api.sendFrame(sessionRef.current, blob)
      if (!r.buffering && typeof r.score === 'number') {
        setScore(r.score)
        setScores(prev => [...prev.slice(-599), r.score])
      }
    } catch (e: any) {
      setErr(e?.response?.data?.error || 'frame upload failed')
    } finally {
      inFlight.current = false
    }
  }

  const start = async () => {
    setErr(''); setScores([]); setIncidents([]); setScore(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      const s = await api.openStream(cameraId || null, threshold, LIVE_SEND_FPS)
      sessionRef.current = s.id
      setRunning(true)
      timerRef.current = window.setInterval(tick, 1000 / LIVE_SEND_FPS)
    } catch (e: any) {
      setErr(e?.response?.data?.error || e.message || 'could not start camera')
    }
  }

  const stop = async () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    const v = videoRef.current
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      v.srcObject = null
    }
    const sid = sessionRef.current
    sessionRef.current = null
    setRunning(false)
    if (sid) {
      try {
        const summary = await api.closeStream(sid)
        setIncidents(summary.incidents ?? [])
      } catch { /* session may already be gone */ }
    }
  }

  const alerting = score != null && score > threshold

  return (
    <Stack spacing={2}>
      <Card><CardContent>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Typography variant="h6">Live monitoring</Typography>
          <Box flexGrow={1} />
          {running && <Chip color={alerting ? 'error' : 'success'}
                            label={alerting ? 'ANOMALY DETECTED' : 'normal'} />}
          {score != null && <Chip variant="outlined" label={`score ${score.toFixed(3)}`} />}
        </Stack>
      </CardContent></Card>

      {err && <Alert severity="error" onClose={() => setErr('')}>{err}</Alert>}

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card><CardContent>
            <Box sx={{
              position: 'relative', borderRadius: 2, overflow: 'hidden',
              outline: alerting ? '3px solid' : 'none', outlineColor: 'error.main',
            }}>
              <video ref={videoRef} muted playsInline
                     style={{ width: '100%', display: 'block', background: '#000' }} />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </Box>

            <Stack spacing={2} sx={{ mt: 2 }}>
              <FormControl size="small" fullWidth disabled={running}>
                <InputLabel>Camera record (optional)</InputLabel>
                <Select label="Camera record (optional)" value={cameraId}
                        onChange={e => setCameraId(e.target.value)}>
                  <MenuItem value="">— none —</MenuItem>
                  {cameras.map(c => (
                    <MenuItem key={c.id} value={c.id}>{c.name} · {c.location}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box>
                <Typography gutterBottom variant="body2">
                  Alert threshold: {threshold.toFixed(2)}
                </Typography>
                <Slider value={threshold} min={0.05} max={0.95} step={0.05} disabled={running}
                        onChange={(_, v) => setThreshold(v as number)} />
              </Box>

              <Stack direction="row" spacing={1}>
                {!running
                  ? <Button variant="contained" fullWidth onClick={start}>Start camera</Button>
                  : <Button variant="outlined" color="error" fullWidth onClick={stop}>
                      Stop and save incidents</Button>}
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Frames are sent at {LIVE_SEND_FPS} fps; the detector scores every 16 frames
                (~{(16 / LIVE_SEND_FPS).toFixed(1)} s) using a rolling window of recent
                context. Incidents are saved when you stop.
              </Typography>
            </Stack>
          </CardContent></Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card><CardContent>
            <Typography variant="subtitle2" gutterBottom>Live score</Typography>
            {scores.length
              ? <ScoreTimeline scores={scores} secondsPerPoint={16 / LIVE_SEND_FPS}
                               threshold={threshold} height={240} />
              : <Typography color="text.secondary" sx={{ py: 8, textAlign: 'center' }}>
                  Start the camera to see live scores.</Typography>}
          </CardContent></Card>
        </Grid>
      </Grid>

      {incidents.length > 0 && (
        <Card><CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Incidents saved from this session
          </Typography>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell>Start</TableCell><TableCell>End</TableCell>
              <TableCell>Duration</TableCell><TableCell align="right">Peak</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {incidents.map((i, k) => (
                <TableRow key={k}>
                  <TableCell>{i.start_sec.toFixed(1)}s</TableCell>
                  <TableCell>{i.end_sec.toFixed(1)}s</TableCell>
                  <TableCell>{i.duration_sec.toFixed(1)}s</TableCell>
                  <TableCell align="right">
                    <Chip size="small" color="error" label={i.peak_score.toFixed(3)} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </Stack>
  )
}
