import {
  Area, AreaChart, CartesianGrid, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { Incident } from '../services/api'

interface Props {
  scores: number[]
  /** Seconds represented by one snippet score (16 frames / fps). */
  secondsPerPoint: number
  threshold: number
  incidents?: Incident[]
  height?: number
  onSeek?: (sec: number) => void
  /** Current playback position, in seconds. Draws a moving playhead. */
  playhead?: number
}

/**
 * Anomaly score over time, with the alert threshold, detected incidents, and a
 * playhead tracking the attached video. Clicking seeks, so a reviewer can move
 * between a peak in the curve and the footage that produced it.
 */
export default function ScoreTimeline({
  scores, secondsPerPoint, threshold, incidents = [], height = 220, onSeek, playhead,
}: Props) {
  const data = scores.map((s, i) => ({ t: +(i * secondsPerPoint).toFixed(2), score: s }))
  const duration = data.length ? data[data.length - 1].t : 0

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data}
                 onClick={(e: any) => onSeek && e?.activeLabel != null && onSeek(Number(e.activeLabel))}
                 margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                 style={{ cursor: onSeek ? 'pointer' : 'default' }}>
        <defs>
          <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#4f9cf9" stopOpacity={0.7} />
            <stop offset="100%" stopColor="#4f9cf9" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a3240" />
        {/* Numeric axis so the playhead can sit at any time, not only at a sample. */}
        <XAxis dataKey="t" type="number" domain={[0, duration]}
               tick={{ fontSize: 11 }} stroke="#8a93a3" allowDataOverflow
               tickFormatter={(v: number) => `${Math.floor(v / 60)}:${String(Math.floor(v % 60)).padStart(2, '0')}`} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#8a93a3" width={38} />
        <Tooltip
          contentStyle={{ background: '#1a1f29', border: '1px solid #2a3240', borderRadius: 8 }}
          formatter={(v: any) => [Number(v).toFixed(3), 'anomaly score']}
          labelFormatter={(l: any) => `t = ${Number(l).toFixed(2)} s`} />
        {incidents.map(inc => (
          <ReferenceArea key={inc.id} x1={inc.start_sec} x2={inc.end_sec}
                         fill="#ff5252" fillOpacity={0.16} />
        ))}
        <ReferenceLine y={threshold} stroke="#ffb300" strokeDasharray="5 4"
                       label={{ value: `threshold ${threshold}`, fontSize: 10, fill: '#ffb300', position: 'insideTopRight' }} />
        <Area type="monotone" dataKey="score" stroke="#4f9cf9" strokeWidth={1.6}
              fill="url(#scoreFill)" isAnimationActive={false} dot={false} />
        {playhead != null && playhead >= 0 && (
          <ReferenceLine x={playhead} stroke="#ffffff" strokeWidth={1.6}
                         isFront ifOverflow="extendDomain" />
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}
