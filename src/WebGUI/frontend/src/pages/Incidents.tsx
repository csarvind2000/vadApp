import {
  Box, Button, Card, CardContent, Chip, FormControl, InputLabel, LinearProgress,
  MenuItem, Select, Slider, Stack, Table, TableBody, TableCell, TableHead,
  TableRow, TextField, Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { api, type Incident } from '../services/api'

const REVIEW_COLOUR = {
  unreviewed: 'default', confirmed: 'error', false_positive: 'success',
} as const

/** Incident register: every detection from uploads and live sessions, reviewable
 *  and exportable. This is the audit trail the system exists to produce. */
export default function Incidents() {
  const [rows, setRows] = useState<Incident[]>([])
  const [loading, setLoading] = useState(true)
  const [review, setReview] = useState('')
  const [minScore, setMinScore] = useState(0)

  const load = () => {
    setLoading(true)
    const params: Record<string, unknown> = {}
    if (review) params.review = review
    if (minScore > 0) params.min_score = minScore
    api.listIncidents(params).then(d => { setRows(d.results); setLoading(false) })
  }
  useEffect(load, [review, minScore])

  const setVerdict = (id: string, v: Incident['review']) =>
    api.updateIncident(id, { review: v }).then(load)

  const exportCsv = () => {
    const head = ['id', 'source', 'start_sec', 'end_sec', 'duration_sec',
                  'peak_score', 'mean_score', 'review', 'detected_at']
    const body = rows.map(r => [
      r.id, r.video_name || r.camera_name || '', r.start_sec, r.end_sec,
      r.duration_sec, r.peak_score, r.mean_score, r.review, r.detected_at,
    ].join(','))
    const blob = new Blob([[head.join(','), ...body].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `incidents_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <Card>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }} flexWrap="wrap">
          <Typography variant="h6">Incident register</Typography>
          <Box flexGrow={1} />
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>Review state</InputLabel>
            <Select label="Review state" value={review} onChange={e => setReview(e.target.value)}>
              <MenuItem value="">All</MenuItem>
              <MenuItem value="unreviewed">Unreviewed</MenuItem>
              <MenuItem value="confirmed">Confirmed</MenuItem>
              <MenuItem value="false_positive">False positive</MenuItem>
            </Select>
          </FormControl>
          <Box sx={{ width: 180 }}>
            <Typography variant="caption">min peak score {minScore.toFixed(2)}</Typography>
            <Slider size="small" value={minScore} min={0} max={1} step={0.05}
                    onChange={(_, v) => setMinScore(v as number)} />
          </Box>
          <Button variant="outlined" onClick={exportCsv} disabled={!rows.length}>
            Export CSV
          </Button>
        </Stack>

        {loading ? <LinearProgress /> : rows.length === 0 ? (
          <Typography color="text.secondary" sx={{ py: 5, textAlign: 'center' }}>
            No incidents match these filters.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead><TableRow>
              <TableCell>Preview</TableCell><TableCell>Source</TableCell>
              <TableCell>Time</TableCell><TableCell>Duration</TableCell>
              <TableCell align="right">Peak</TableCell><TableCell>Detected</TableCell>
              <TableCell>Review</TableCell><TableCell align="right">Verdict</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {rows.map(r => (
                <TableRow key={r.id} hover>
                  <TableCell>{r.thumbnail_url
                    ? <img src={r.thumbnail_url} alt="" width={88}
                           style={{ borderRadius: 4, display: 'block' }} />
                    : '—'}</TableCell>
                  <TableCell>{r.video_name || r.camera_name || '—'}</TableCell>
                  <TableCell>{r.start_sec.toFixed(1)}–{r.end_sec.toFixed(1)}s</TableCell>
                  <TableCell>{r.duration_sec.toFixed(1)}s</TableCell>
                  <TableCell align="right">
                    <Chip size="small" color="error" label={r.peak_score.toFixed(3)} />
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption">
                      {new Date(r.detected_at).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip size="small" label={r.review.replace('_', ' ')}
                          color={REVIEW_COLOUR[r.review] as any} />
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                      <Button size="small" onClick={() => setVerdict(r.id, 'confirmed')}>
                        Confirm</Button>
                      <Button size="small" color="success"
                              onClick={() => setVerdict(r.id, 'false_positive')}>
                        False</Button>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
