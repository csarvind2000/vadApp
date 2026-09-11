import {
  Alert, Box, Button, Card, CardContent, LinearProgress, Slider, Stack,
  TextField, Typography,
} from '@mui/material'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { DEFAULT_THRESHOLD } from '../constants'
import { api } from '../services/api'

export default function Upload() {
  const nav = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [pct, setPct] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const pick = (f: File | null) => {
    setFile(f)
    if (f && !name) setName(f.name.replace(/\.[^.]+$/, ''))
  }

  const submit = async () => {
    if (!file) return
    setBusy(true); setErr(''); setPct(0)
    try {
      const video = await api.uploadVideo(file, name || file.name, setPct)
      await api.analyzeVideo(video.id, threshold)
      nav(`/videos/${video.id}`)
    } catch (e: any) {
      setErr(e?.response?.data?.error || e.message || 'upload failed')
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Upload a video for analysis</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          The file is stored, queued, and scored snippet by snippet. Stretches scoring
          above the threshold are saved as incidents you can review.
        </Typography>

        <Stack spacing={3}>
          <Box sx={{
            border: '2px dashed', borderColor: file ? 'primary.main' : 'divider',
            borderRadius: 2, p: 4, textAlign: 'center',
          }}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); pick(e.dataTransfer.files?.[0] ?? null) }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Drag a video here, or
            </Typography>
            <Button variant="outlined" component="label">
              Choose file
              <input hidden type="file" accept="video/*"
                     onChange={e => pick(e.target.files?.[0] ?? null)} />
            </Button>
            {file && (
              <Typography sx={{ mt: 2 }}>
                {file.name} — {(file.size / 1e6).toFixed(1)} MB
              </Typography>
            )}
          </Box>

          <TextField label="Name" value={name} onChange={e => setName(e.target.value)}
                     fullWidth size="small" />

          <Box>
            <Typography gutterBottom>Alert threshold: {threshold.toFixed(2)}</Typography>
            <Box sx={{ px: 1 }}>
              <Slider value={threshold} min={0.05} max={0.95} step={0.05}
                      onChange={(_, v) => setThreshold(v as number)} />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', px: 1 }}>
              <Typography variant="caption" color="text.secondary">more sensitive</Typography>
              <Typography variant="caption" color="text.secondary">stricter</Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              Lower catches more but raises false alarms. This is stored with the job,
              so results always record the threshold that produced them.
            </Typography>
          </Box>

          {busy && <Box><LinearProgress variant="determinate" value={pct} />
            <Typography variant="caption">uploading {pct}%</Typography></Box>}
          {err && <Alert severity="error">{err}</Alert>}

          <Button variant="contained" size="large" disabled={!file || busy} onClick={submit}>
            {busy ? 'Working…' : 'Upload and analyze'}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  )
}
