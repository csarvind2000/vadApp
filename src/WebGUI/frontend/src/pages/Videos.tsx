import {
  Box, Card, CardContent, Chip, IconButton, LinearProgress, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, Typography,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Video } from '../services/api'

const fmt = (s: number | null) =>
  s == null ? '—' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

const statusColour = (s?: string) =>
  s === 'done' ? 'success' : s === 'failed' ? 'error' : s === 'running' ? 'warning' : 'default'

export default function Videos() {
  const nav = useNavigate()
  const [videos, setVideos] = useState<Video[]>([])
  const [loading, setLoading] = useState(true)

  const load = () => api.listVideos().then(d => { setVideos(d.results); setLoading(false) })

  useEffect(() => {
    load()
    // Poll while any job is still in flight so progress appears without a refresh.
    const t = setInterval(() => {
      setVideos(prev => {
        if (prev.some(v => ['queued', 'running'].includes(v.latest_job?.status ?? ''))) load()
        return prev
      })
    }, 3000)
    return () => clearInterval(t)
  }, [])

  if (loading) return <LinearProgress />

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Analyzed videos</Typography>
        {videos.length === 0 ? (
          <Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
            Nothing yet — upload a video to get started.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell><TableCell>Duration</TableCell>
                <TableCell>Status</TableCell><TableCell align="right">Max score</TableCell>
                <TableCell align="right">Incidents</TableCell><TableCell />
              </TableRow>
            </TableHead>
            <TableBody>
              {videos.map(v => (
                <TableRow key={v.id} hover sx={{ cursor: 'pointer' }}
                          onClick={() => nav(`/videos/${v.id}`)}>
                  <TableCell>{v.name}</TableCell>
                  <TableCell>{fmt(v.duration_sec)}</TableCell>
                  <TableCell>
                    <Chip size="small" label={v.latest_job?.status ?? 'not analyzed'}
                          color={statusColour(v.latest_job?.status) as any} />
                  </TableCell>
                  <TableCell align="right">
                    {v.latest_job?.max_score != null ? v.latest_job.max_score.toFixed(3) : '—'}
                  </TableCell>
                  <TableCell align="right">
                    {v.n_incidents > 0
                      ? <Chip size="small" color="error" label={v.n_incidents} />
                      : <Typography variant="body2" color="text.secondary">0</Typography>}
                  </TableCell>
                  <TableCell align="right">
                    <IconButton size="small" onClick={e => {
                      e.stopPropagation()
                      api.deleteVideo(v.id).then(load)
                    }}><DeleteIcon fontSize="small" /></IconButton>
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
