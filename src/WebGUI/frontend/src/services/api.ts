import axios from 'axios'
import { API_BASE } from '../constants'

export const client = axios.create({ baseURL: API_BASE })

export interface Incident {
  categories: CategoryGuess[] | null
  id: string
  video: string | null
  video_name?: string
  camera: string | null
  camera_name?: string | null
  start_sec: number
  end_sec: number
  duration_sec: number
  peak_score: number
  mean_score: number
  peak_frame: number | null
  thumbnail_url: string | null
  review: 'unreviewed' | 'confirmed' | 'false_positive'
  note: string
  detected_at: string
}

export interface CategoryGuess { category: string; probability: number }

export interface AnalysisJob {
  id: string
  video: string
  status: 'queued' | 'running' | 'done' | 'failed'
  progress: number
  error: string
  threshold: number
  model_name: string
  backbone: string
  max_score: number | null
  mean_score: number | null
  categories: CategoryGuess[] | null
  detections: unknown | null
  n_incidents: number
  created_at: string
  finished_at: string | null
  snippet_scores?: number[]
  attention?: number[] | null
  incidents?: Incident[]
}

export interface Video {
  id: string
  name: string
  file_url: string | null
  source: string
  duration_sec: number | null
  fps: number | null
  n_frames: number | null
  size_bytes: number | null
  created_at: string
  latest_job: AnalysisJob | null
  n_incidents: number
}

export interface Camera {
  id: string
  name: string
  kind: 'browser' | 'rtsp'
  url: string
  location: string
  threshold: number
  enabled: boolean
  n_incidents: number
}

interface Page<T> { count: number; next: string | null; previous: string | null; results: T[] }

export const api = {
  health: () => client.get('/health/').then(r => r.data),

  listVideos: (page = 1) =>
    client.get<Page<Video>>('/videos/', { params: { page } }).then(r => r.data),
  getVideo: (id: string) => client.get<Video>(`/videos/${id}/`).then(r => r.data),
  uploadVideo: (file: File, name: string, onProgress?: (pct: number) => void) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name)
    return client.post<Video>('/videos/', fd, {
      onUploadProgress: e =>
        onProgress && e.total && onProgress(Math.round((e.loaded / e.total) * 100)),
    }).then(r => r.data)
  },
  analyzeVideo: (id: string, threshold: number, minDuration = 1.0) =>
    client.post<AnalysisJob>(`/videos/${id}/analyze/`, {
      threshold, min_duration_sec: minDuration,
    }).then(r => r.data),
  deleteVideo: (id: string) => client.delete(`/videos/${id}/`),

  getJob: (id: string) => client.get<AnalysisJob>(`/jobs/${id}/`).then(r => r.data),

  listIncidents: (params: Record<string, unknown> = {}) =>
    client.get<Page<Incident>>('/incidents/', { params }).then(r => r.data),
  updateIncident: (id: string, data: Partial<Incident>) =>
    client.patch<Incident>(`/incidents/${id}/`, data).then(r => r.data),
  deleteIncident: (id: string) => client.delete(`/incidents/${id}/`),

  listCameras: () => client.get<Page<Camera>>('/cameras/').then(r => r.data),
  createCamera: (data: Partial<Camera>) =>
    client.post<Camera>('/cameras/', data).then(r => r.data),
  deleteCamera: (id: string) => client.delete(`/cameras/${id}/`),

  openStream: (cameraId: string | null, threshold: number, fps: number) =>
    client.post('/streams/', { camera: cameraId, threshold, fps }).then(r => r.data),
  sendFrame: (sid: string, blob: Blob) => {
    const fd = new FormData()
    fd.append('frame', blob, 'f.jpg')
    return client.post(`/streams/${sid}/frame/`, fd).then(r => r.data)
  },
  closeStream: (sid: string) => client.post(`/streams/${sid}/close/`).then(r => r.data),
}
