import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import App from './App'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#4f9cf9' },
    error: { main: '#ff5252' },
    warning: { main: '#ffb300' },
    background: { default: '#11151c', paper: '#1a1f29' },
  },
  shape: { borderRadius: 10 },
  typography: { fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif' },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </React.StrictMode>,
)
