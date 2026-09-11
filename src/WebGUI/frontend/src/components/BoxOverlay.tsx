import { useEffect, useRef } from 'react'

export interface Box { cls: string; conf: number; x: number; y: number; w: number; h: number }
export interface FrameDetections { frame: number; boxes: Box[] }
export interface Detections {
  fps: number
  sample_step: number
  sampled_frames: number
  detections: FrameDetections[]
  class_counts: Record<string, number>
}

interface Props {
  detections: Detections | null
  /** Current playback position, seconds. */
  time: number
  /** Rendered size of the video element, so boxes scale with it. */
  width: number
  height: number
}

const COLOURS: Record<string, string> = {
  person: '#4f9cf9', car: '#ff9f43', truck: '#ff9f43', bus: '#ff9f43',
  motorcycle: '#ee5a6f', bicycle: '#ee5a6f', knife: '#ff5252',
}
const fallback = '#55a868'

/**
 * Draws detection boxes over the video.
 *
 * Detections exist only for sampled frames, so the nearest sample within half a
 * sampling interval is shown and otherwise nothing is drawn. Holding a stale box
 * indefinitely would place it where the object no longer is.
 */
export default function BoxOverlay({ detections, time, width, height }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    c.width = width; c.height = height
    ctx.clearRect(0, 0, width, height)
    if (!detections?.detections?.length || !width || !height) return

    const fps = detections.fps || 30
    const target = time * fps
    const tol = (detections.sample_step || 1) / 2 + 1

    // nearest sampled frame
    let best: FrameDetections | null = null
    let bestD = Infinity
    for (const d of detections.detections) {
      const dist = Math.abs(d.frame - target)
      if (dist < bestD) { bestD = dist; best = d }
    }
    if (!best || bestD > tol) return

    ctx.lineWidth = 2
    ctx.font = '11px system-ui, sans-serif'
    ctx.textBaseline = 'top'
    for (const b of best.boxes) {
      const col = COLOURS[b.cls] ?? fallback
      const x = b.x * width, y = b.y * height
      const w = b.w * width, h = b.h * height
      ctx.strokeStyle = col
      ctx.strokeRect(x, y, w, h)
      const label = `${b.cls} ${(b.conf * 100).toFixed(0)}%`
      const tw = ctx.measureText(label).width + 6
      ctx.fillStyle = col
      ctx.fillRect(x, Math.max(0, y - 14), tw, 14)
      ctx.fillStyle = '#0b0e13'
      ctx.fillText(label, x + 3, Math.max(0, y - 13))
    }
  }, [detections, time, width, height])

  return (
    <canvas ref={ref}
            style={{ position: 'absolute', top: 0, left: 0, width, height,
                     pointerEvents: 'none' }} />
  )
}
