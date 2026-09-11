export const API_BASE = import.meta.env.VITE_API_BASE || '/api'
export const APP_NAME = 'Video Anomaly Detection'
export const APP_VERSION = import.meta.env.VITE_APP_VERSION || '1.0.0'

/** Frames per second sent to the detector during live monitoring.
 *  The model consumes 16-frame snippets, so ~10 fps yields a score ~1.6 s.
 *  Sending every browser frame would only add latency, not accuracy. */
export const LIVE_SEND_FPS = 10
export const DEFAULT_THRESHOLD = 0.5
