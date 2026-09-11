import { AppBar, Box, Chip, Tab, Tabs, Toolbar, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { APP_NAME, APP_VERSION } from './constants'
import { api } from './services/api'
import Incidents from './pages/Incidents'
import LiveMonitor from './pages/LiveMonitor'
import Upload from './pages/Upload'
import VideoDetail from './pages/VideoDetail'
import Videos from './pages/Videos'

const TABS = [
  { label: 'Videos', path: '/videos' },
  { label: 'Upload', path: '/upload' },
  { label: 'Live Monitor', path: '/live' },
  { label: 'Incidents', path: '/incidents' },
]

export default function App() {
  const nav = useNavigate()
  const loc = useLocation()
  const [modelStatus, setModelStatus] = useState<string>('checking')

  useEffect(() => {
    const check = () =>
      api.health()
        .then(d => setModelStatus(d?.model_service?.status ?? 'unknown'))
        .catch(() => setModelStatus('backend unreachable'))
    check()
    const t = setInterval(check, 15000)
    return () => clearInterval(t)
  }, [])

  const active = TABS.findIndex(t => loc.pathname.startsWith(t.path))
  const ok = modelStatus === 'ok'

  return (
    <Box sx={{ minHeight: '100vh' }}>
      <AppBar position="static" color="default" elevation={0}
              sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar sx={{ gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>{APP_NAME}</Typography>
          <Tabs value={active < 0 ? 0 : active}
                onChange={(_, i) => nav(TABS[i].path)}
                sx={{ flexGrow: 1, minHeight: 48 }}>
            {TABS.map(t => <Tab key={t.path} label={t.label} />)}
          </Tabs>
          <Chip size="small" color={ok ? 'success' : 'error'}
                label={ok ? 'model ready' : `model: ${modelStatus}`} />
          <Typography variant="caption" color="text.secondary">v{APP_VERSION}</Typography>
        </Toolbar>
      </AppBar>

      <Box sx={{ p: 3, maxWidth: 1400, mx: 'auto' }}>
        <Routes>
          <Route path="/" element={<Navigate to="/videos" replace />} />
          <Route path="/videos" element={<Videos />} />
          <Route path="/videos/:id" element={<VideoDetail />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/live" element={<LiveMonitor />} />
          <Route path="/incidents" element={<Incidents />} />
        </Routes>
      </Box>
    </Box>
  )
}
