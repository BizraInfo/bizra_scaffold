/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║  BIZRA COGNITIVE CONTROL CENTER                                              ║
 * ║  Version: 1.0.0-PEAK-MASTERPIECE                                             ║
 * ║  Author: BIZRA Cognitive Architecture                                        ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  Peak Masterpiece Edition - Highest SNR Implementation                       ║
 * ║                                                                              ║
 * ║  Components:                                                                 ║
 * ║  1. Live SNR Gauge - Real-time signal-to-noise monitoring                   ║
 * ║  2. Thought Graph - Graph of Thoughts visualization                         ║
 * ║  3. Ihsan Radar - 8-dimensional quality radar                               ║
 * ║  4. Agent Orchestra - PAT/SAT live status                                   ║
 * ║  5. SAPE Patterns - Pattern elevation tracking                              ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

'use client';

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  useSNRMetrics,
  useCognitiveState,
  formatSNR,
  formatSNRdB,
  getSNRColor,
  getSNRBgColor,
  getQualityTierInfo,
  type SNRMetrics,
  type ThoughtNode,
} from '@/lib/snr-engine';
import { useBizraLiveMetrics, type BizraLiveMetrics } from '@/lib/live-data';

// ═══════════════════════════════════════════════════════════════════════════════
// ICON COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const Icons = {
  Activity: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  ),
  Brain: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54z"/>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54z"/>
    </svg>
  ),
  Zap: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  ),
  Target: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    </svg>
  ),
  GitBranch: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>
      <path d="M18 9a9 9 0 0 1-9 9"/>
    </svg>
  ),
  Layers: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/>
      <polyline points="2 12 12 17 22 12"/>
    </svg>
  ),
  Shield: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
  TrendingUp: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
    </svg>
  ),
  TrendingDown: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>
    </svg>
  ),
};

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const GlassCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'highlighted' | 'danger' | 'success' | 'elite';
}> = ({ children, className = '', variant = 'default' }) => {
  const variants = {
    default: 'bg-slate-900/60 border-slate-700/50',
    highlighted: 'bg-emerald-950/30 border-emerald-500/40',
    danger: 'bg-red-950/30 border-red-500/40',
    success: 'bg-emerald-950/40 border-emerald-500/50',
    elite: 'bg-gradient-to-br from-emerald-950/40 via-cyan-950/30 to-violet-950/40 border-emerald-500/50',
  };
  return (
    <div className={`relative overflow-hidden rounded-xl backdrop-blur-md border ${variants[variant]} ${className}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      <div className="relative z-10">{children}</div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// SNR GAUGE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const SNRGauge: React.FC<{ snr: SNRMetrics }> = ({ snr }) => {
  const tierInfo = getQualityTierInfo(snr.qualityTier);
  const circumference = 2 * Math.PI * 45; // radius = 45
  const strokeDashoffset = circumference * (1 - snr.snrScore);

  return (
    <GlassCard variant={snr.qualityTier === 'elite' ? 'elite' : 'default'} className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 text-emerald-400"><Icons.Activity /></div>
          <span className="text-sm font-mono text-slate-400">SNR MONITOR</span>
        </div>
        <span className={`text-xs font-mono px-2 py-1 rounded ${tierInfo.color} bg-current/10`}>
          {tierInfo.icon} {tierInfo.label}
        </span>
      </div>

      <div className="flex items-center gap-8">
        {/* Circular Gauge */}
        <div className="relative w-32 h-32">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle cx="50" cy="50" r="45" fill="none" strokeWidth="8"
              className="stroke-slate-800" />
            {/* Progress circle */}
            <circle cx="50" cy="50" r="45" fill="none" strokeWidth="8"
              strokeLinecap="round"
              className={getSNRBgColor(snr.snrScore).replace('bg-', 'stroke-')}
              style={{
                strokeDasharray: circumference,
                strokeDashoffset,
                transition: 'stroke-dashoffset 0.5s ease',
              }} />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-2xl font-bold font-mono ${getSNRColor(snr.snrScore)}`}>
              {formatSNR(snr.snrScore)}
            </span>
            <span className="text-xs text-slate-500">
              {formatSNRdB(snr.snrDecibels)}
            </span>
          </div>
        </div>

        {/* Signal/Noise Breakdown */}
        <div className="flex-1 space-y-3">
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-emerald-400">SIGNAL</span>
              <span className="text-slate-500 font-mono">{(snr.snrScore * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-500"
                style={{ width: `${snr.snrScore * 100}%` }} />
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-red-400">NOISE</span>
              <span className="text-slate-500 font-mono">
                {((1 - snr.entropyReduction) * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-red-500 to-amber-500 transition-all duration-500"
                style={{ width: `${(1 - snr.entropyReduction) * 100}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Signal Components */}
      <div className="grid grid-cols-5 gap-2 mt-4 pt-4 border-t border-slate-700/50">
        {[
          { label: 'Ihsan', value: snr.signal.ihsanScore, weight: '35%' },
          { label: 'Gates', value: snr.signal.gatePassRate, weight: '25%' },
          { label: 'PAT', value: snr.signal.patEfficiency, weight: '15%' },
          { label: 'SAT', value: snr.signal.satConsensus, weight: '15%' },
          { label: 'SAPE', value: snr.signal.sapeOptimization, weight: '10%' },
        ].map((comp) => (
          <div key={comp.label} className="text-center">
            <div className={`text-lg font-mono font-bold ${comp.value >= 0.95 ? 'text-emerald-400' : comp.value >= 0.85 ? 'text-cyan-400' : 'text-amber-400'}`}>
              {(comp.value * 100).toFixed(0)}
            </div>
            <div className="text-[10px] text-slate-500">{comp.label}</div>
            <div className="text-[9px] text-slate-600">({comp.weight})</div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// IHSAN RADAR COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const IhsanRadar: React.FC<{ metrics: BizraLiveMetrics }> = ({ metrics }) => {
  const dimensions = metrics.ihsanDimensions.length > 0
    ? metrics.ihsanDimensions
    : [
        { dimension: 'correctness', score: 0.95 },
        { dimension: 'safety', score: 0.98 },
        { dimension: 'user_benefit', score: 0.92 },
        { dimension: 'efficiency', score: 0.91 },
        { dimension: 'auditability', score: 0.96 },
        { dimension: 'anti_centralization', score: 0.88 },
        { dimension: 'robustness', score: 0.89 },
        { dimension: 'adl_fairness', score: 0.93 },
      ];

  const avgScore = dimensions.reduce((sum, d) => sum + d.score, 0) / dimensions.length;
  const center = 80;
  const radius = 60;

  // Generate polygon points for radar
  const points = dimensions.map((dim, i) => {
    const angle = (i / dimensions.length) * 2 * Math.PI - Math.PI / 2;
    const x = center + Math.cos(angle) * radius * dim.score;
    const y = center + Math.sin(angle) * radius * dim.score;
    return `${x},${y}`;
  }).join(' ');

  // Generate background polygon (full scale)
  const bgPoints = dimensions.map((_, i) => {
    const angle = (i / dimensions.length) * 2 * Math.PI - Math.PI / 2;
    const x = center + Math.cos(angle) * radius;
    const y = center + Math.sin(angle) * radius;
    return `${x},${y}`;
  }).join(' ');

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 text-violet-400"><Icons.Target /></div>
          <span className="text-sm font-mono text-slate-400">IHSAN RADAR</span>
        </div>
        <span className={`text-lg font-mono font-bold ${avgScore >= 0.95 ? 'text-emerald-400' : avgScore >= 0.90 ? 'text-amber-400' : 'text-red-400'}`}>
          {avgScore.toFixed(2)}
        </span>
      </div>

      <div className="flex items-center gap-6">
        {/* Radar Chart */}
        <svg viewBox="0 0 160 160" className="w-40 h-40">
          {/* Background lines */}
          {[0.25, 0.5, 0.75, 1].map((scale) => (
            <polygon key={scale}
              points={dimensions.map((_, i) => {
                const angle = (i / dimensions.length) * 2 * Math.PI - Math.PI / 2;
                const x = center + Math.cos(angle) * radius * scale;
                const y = center + Math.sin(angle) * radius * scale;
                return `${x},${y}`;
              }).join(' ')}
              fill="none" stroke="rgba(100,116,139,0.2)" strokeWidth="1" />
          ))}
          {/* Axis lines */}
          {dimensions.map((_, i) => {
            const angle = (i / dimensions.length) * 2 * Math.PI - Math.PI / 2;
            const x = center + Math.cos(angle) * radius;
            const y = center + Math.sin(angle) * radius;
            return <line key={i} x1={center} y1={center} x2={x} y2={y} stroke="rgba(100,116,139,0.3)" strokeWidth="1" />;
          })}
          {/* Data polygon */}
          <polygon points={points} fill="rgba(16,185,129,0.2)" stroke="rgb(16,185,129)" strokeWidth="2" />
          {/* Data points */}
          {dimensions.map((dim, i) => {
            const angle = (i / dimensions.length) * 2 * Math.PI - Math.PI / 2;
            const x = center + Math.cos(angle) * radius * dim.score;
            const y = center + Math.sin(angle) * radius * dim.score;
            return <circle key={i} cx={x} cy={y} r="3" fill={dim.score >= 0.95 ? '#10b981' : dim.score >= 0.90 ? '#f59e0b' : '#ef4444'} />;
          })}
        </svg>

        {/* Dimension List */}
        <div className="flex-1 grid grid-cols-2 gap-1">
          {dimensions.map((dim) => (
            <div key={dim.dimension} className="flex items-center gap-2 p-1.5 rounded bg-slate-800/50">
              <div className={`w-1.5 h-1.5 rounded-full ${dim.score >= 0.95 ? 'bg-emerald-500' : dim.score >= 0.90 ? 'bg-amber-500' : 'bg-red-500'}`} />
              <span className="text-[10px] text-slate-400 flex-1 capitalize">{dim.dimension.replace('_', ' ')}</span>
              <span className={`text-xs font-mono ${dim.score >= 0.95 ? 'text-emerald-400' : dim.score >= 0.90 ? 'text-amber-400' : 'text-red-400'}`}>
                {dim.score.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// GRAPH OF THOUGHTS VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

interface ThoughtNodeDisplay {
  id: string;
  label: string;
  type: 'premise' | 'inference' | 'abstraction' | 'synthesis' | 'conclusion';
  x: number;
  y: number;
  confidence: number;
  children: string[];
}

const GraphOfThoughts: React.FC<{ snr: SNRMetrics }> = ({ snr }) => {
  // Generate a sample thought graph based on current cognitive state
  const [activeNode, setActiveNode] = useState<string | null>(null);

  // Sample thought graph structure
  const thoughts: ThoughtNodeDisplay[] = useMemo(() => [
    { id: 'p1', label: 'Request', type: 'premise', x: 20, y: 50, confidence: 1.0, children: ['i1', 'i2'] },
    { id: 'i1', label: 'Context', type: 'inference', x: 35, y: 25, confidence: 0.95, children: ['a1'] },
    { id: 'i2', label: 'Intent', type: 'inference', x: 35, y: 75, confidence: 0.92, children: ['a1'] },
    { id: 'a1', label: 'Pattern', type: 'abstraction', x: 55, y: 50, confidence: snr.signal.sapeOptimization, children: ['s1', 's2'] },
    { id: 's1', label: 'Solution A', type: 'synthesis', x: 75, y: 30, confidence: snr.signal.ihsanScore, children: ['c1'] },
    { id: 's2', label: 'Solution B', type: 'synthesis', x: 75, y: 70, confidence: snr.signal.satConsensus, children: ['c1'] },
    { id: 'c1', label: 'Response', type: 'conclusion', x: 92, y: 50, confidence: snr.snrScore, children: [] },
  ], [snr]);

  const typeColors: Record<string, string> = {
    premise: 'fill-slate-400',
    inference: 'fill-cyan-400',
    abstraction: 'fill-violet-400',
    synthesis: 'fill-amber-400',
    conclusion: 'fill-emerald-400',
  };

  const typeBgColors: Record<string, string> = {
    premise: 'bg-slate-500',
    inference: 'bg-cyan-500',
    abstraction: 'bg-violet-500',
    synthesis: 'bg-amber-500',
    conclusion: 'bg-emerald-500',
  };

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 text-cyan-400"><Icons.GitBranch /></div>
          <span className="text-sm font-mono text-slate-400">GRAPH OF THOUGHTS</span>
        </div>
        <div className="flex gap-2">
          {['premise', 'inference', 'abstraction', 'synthesis', 'conclusion'].map((type) => (
            <span key={type} className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${typeBgColors[type]}`} />
              <span className="text-[9px] text-slate-500 capitalize">{type}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Graph SVG */}
      <svg viewBox="0 0 100 100" className="w-full h-48">
        {/* Edges */}
        {thoughts.flatMap((thought) =>
          thought.children.map((childId) => {
            const child = thoughts.find((t) => t.id === childId);
            if (!child) return null;
            return (
              <line key={`${thought.id}-${childId}`}
                x1={thought.x} y1={thought.y}
                x2={child.x} y2={child.y}
                stroke="rgba(100,116,139,0.4)" strokeWidth="0.5"
                strokeDasharray="2,1" />
            );
          })
        )}
        {/* Nodes */}
        {thoughts.map((thought) => (
          <g key={thought.id}
            onMouseEnter={() => setActiveNode(thought.id)}
            onMouseLeave={() => setActiveNode(null)}
            className="cursor-pointer">
            <circle cx={thought.x} cy={thought.y}
              r={activeNode === thought.id ? 5 : 3.5}
              className={`${typeColors[thought.type]} transition-all duration-200`}
              style={{ opacity: thought.confidence }} />
            {activeNode === thought.id && (
              <>
                <rect x={thought.x - 12} y={thought.y - 18} width="24" height="10" rx="2"
                  fill="rgba(15,23,42,0.9)" stroke="rgba(100,116,139,0.5)" strokeWidth="0.3" />
                <text x={thought.x} y={thought.y - 11} textAnchor="middle"
                  className="text-[4px] fill-white font-mono">{thought.label}</text>
              </>
            )}
          </g>
        ))}
      </svg>

      {/* Active Node Details */}
      {activeNode && (
        <div className="mt-2 p-2 rounded bg-slate-800/50 text-xs">
          <span className="text-slate-400">Active: </span>
          <span className="text-white font-mono">
            {thoughts.find((t) => t.id === activeNode)?.label}
          </span>
          <span className="text-slate-500 ml-2">
            (Confidence: {thoughts.find((t) => t.id === activeNode)?.confidence.toFixed(2)})
          </span>
        </div>
      )}
    </GlassCard>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT ORCHESTRA COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const AgentOrchestra: React.FC<{ metrics: BizraLiveMetrics }> = ({ metrics }) => {
  const patCount = metrics.health?.agents?.pat_count || 7;
  const satCount = metrics.health?.agents?.sat_count || 5;
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setPulse((p) => (p + 1) % 12), 500);
    return () => clearInterval(interval);
  }, []);

  const patAgents = [
    { name: 'Scribe', icon: 'S' },
    { name: 'Analyst', icon: 'A' },
    { name: 'Arbiter', icon: 'R' },
    { name: 'Synthesizer', icon: 'Y' },
    { name: 'Designer', icon: 'D' },
    { name: 'Executor', icon: 'E' },
    { name: 'Guardian', icon: 'G' },
  ].slice(0, patCount);

  const satAgents = [
    { name: 'Security', icon: 'Sec', veto: true },
    { name: 'Formal', icon: 'Frm', veto: true },
    { name: 'Ethics', icon: 'Eth', veto: true },
    { name: 'Resource', icon: 'Res', veto: false },
    { name: 'Context', icon: 'Ctx', veto: false },
  ].slice(0, satCount);

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 text-violet-400"><Icons.Layers /></div>
          <span className="text-sm font-mono text-slate-400">AGENT ORCHESTRA</span>
        </div>
        <span className={`text-xs font-mono ${metrics.health?.status === 'healthy' ? 'text-emerald-400' : 'text-amber-400'}`}>
          {metrics.health?.status === 'healthy' ? '● HEALTHY' : '○ DEGRADED'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* PAT Agents */}
        <div>
          <div className="text-[10px] text-violet-400 font-mono mb-2">PAT ({patCount})</div>
          <div className="flex flex-wrap gap-1">
            {patAgents.map((agent, i) => (
              <div key={agent.name}
                className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                  pulse % patCount === i
                    ? 'bg-violet-500 text-white scale-110'
                    : 'bg-violet-500/20 text-violet-400'
                }`}
                title={agent.name}>
                {agent.icon}
              </div>
            ))}
          </div>
        </div>

        {/* SAT Agents */}
        <div>
          <div className="text-[10px] text-cyan-400 font-mono mb-2">SAT ({satCount})</div>
          <div className="flex flex-wrap gap-1">
            {satAgents.map((agent, i) => (
              <div key={agent.name}
                className={`h-8 px-2 rounded-lg flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                  agent.veto ? 'border border-amber-500/50' : ''
                } ${
                  pulse % satCount === i
                    ? 'bg-cyan-500 text-white scale-105'
                    : 'bg-cyan-500/20 text-cyan-400'
                }`}
                title={`${agent.name}${agent.veto ? ' (VETO)' : ''}`}>
                {agent.icon}
                {agent.veto && <span className="ml-1 text-[8px] text-amber-400">V</span>}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Gate Status */}
      <div className="grid grid-cols-4 gap-2 mt-4 pt-4 border-t border-slate-700/50">
        {[
          { label: 'Ihsan', status: metrics.health?.gates?.ihsan || 'unknown' },
          { label: 'Performance', status: metrics.health?.gates?.performance || 'unknown' },
          { label: 'Quality', status: metrics.health?.gates?.quality || 'unknown' },
          { label: 'Security', status: metrics.health?.gates?.security || 'unknown' },
        ].map((gate) => (
          <div key={gate.label} className="text-center p-2 rounded bg-slate-800/50">
            <div className={`text-xs font-mono ${gate.status === 'active' ? 'text-emerald-400' : 'text-slate-500'}`}>
              {gate.status === 'active' ? '●' : '○'} {gate.label}
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// SAPE PATTERN MONITOR
// ═══════════════════════════════════════════════════════════════════════════════

const SAPEPatternMonitor: React.FC<{ metrics: BizraLiveMetrics }> = ({ metrics }) => {
  const sape = metrics.health?.sape;

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 text-amber-400"><Icons.Zap /></div>
          <span className="text-sm font-mono text-slate-400">SAPE PATTERNS</span>
        </div>
        <span className="text-xs font-mono text-emerald-400">
          {sape?.total_snr_improvement?.toFixed(1) || '0.0'}x SNR
        </span>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <div className="text-center">
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {sape?.patterns_active || 0}
          </div>
          <div className="text-[10px] text-slate-500">Active</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold font-mono text-cyan-400">
            {sape?.patterns_registered || 0}
          </div>
          <div className="text-[10px] text-slate-500">Registered</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold font-mono text-amber-400">
            {sape?.pending_elevations || 0}
          </div>
          <div className="text-[10px] text-slate-500">Pending</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold font-mono text-violet-400">
            {sape?.total_latency_saved_ms?.toFixed(0) || 0}ms
          </div>
          <div className="text-[10px] text-slate-500">Saved</div>
        </div>
      </div>

      {/* Pattern Efficiency Bar */}
      <div className="mt-4 pt-4 border-t border-slate-700/50">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-slate-500">Pattern Utilization</span>
          <span className="text-emerald-400 font-mono">
            {sape?.patterns_active && sape?.patterns_registered
              ? ((sape.patterns_active / sape.patterns_registered) * 100).toFixed(0)
              : 0}%
          </span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-amber-500 via-emerald-500 to-cyan-500 transition-all duration-500"
            style={{
              width: `${sape?.patterns_active && sape?.patterns_registered
                ? (sape.patterns_active / sape.patterns_registered) * 100
                : 0}%`
            }} />
        </div>
      </div>
    </GlassCard>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COGNITIVE CONTROL CENTER
// ═══════════════════════════════════════════════════════════════════════════════

export const CognitiveControlCenter: React.FC = () => {
  const snrMetrics = useSNRMetrics();
  const liveMetrics = useBizraLiveMetrics();
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setLastUpdate(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const tierInfo = getQualityTierInfo(snrMetrics.qualityTier);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-4 md:p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 text-emerald-400"><Icons.Brain /></div>
            <div>
              <h1 className="text-xl font-bold">Cognitive Control Center</h1>
              <p className="text-slate-500 text-sm">BIZRA Peak Masterpiece Edition</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full animate-pulse ${liveMetrics.health?.status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              <span className="text-xs font-mono text-slate-400">
                {liveMetrics.health?.status?.toUpperCase() || 'CONNECTING'}
              </span>
            </div>
            <span className="text-xs font-mono text-slate-500">
              {lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>

      {/* Tier Banner */}
      {snrMetrics.qualityTier === 'elite' && (
        <div className="max-w-7xl mx-auto mb-6">
          <div className="p-3 rounded-lg bg-gradient-to-r from-emerald-950/50 via-cyan-950/50 to-violet-950/50 border border-emerald-500/30 text-center">
            <span className="text-emerald-400 font-mono text-sm">
              {tierInfo.icon} ELITE COGNITIVE STATE ACHIEVED {tierInfo.icon}
            </span>
            <span className="text-slate-400 text-xs ml-2">
              SNR {formatSNR(snrMetrics.snrScore)} | {formatSNRdB(snrMetrics.snrDecibels)}
            </span>
          </div>
        </div>
      )}

      {/* Main Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* SNR Gauge - Full Width */}
        <div className="lg:col-span-2">
          <SNRGauge snr={snrMetrics} />
        </div>

        {/* Ihsan Radar */}
        <IhsanRadar metrics={liveMetrics} />

        {/* Graph of Thoughts */}
        <GraphOfThoughts snr={snrMetrics} />

        {/* Agent Orchestra */}
        <AgentOrchestra metrics={liveMetrics} />

        {/* SAPE Pattern Monitor */}
        <SAPEPatternMonitor metrics={liveMetrics} />
      </div>

      {/* Footer */}
      <div className="max-w-7xl mx-auto mt-6 text-center text-slate-600 text-xs font-mono">
        BIZRA Cognitive Control Center v1.0.0 | Peak Masterpiece Edition | {new Date().toISOString().split('T')[0]}
      </div>
    </div>
  );
};

export default CognitiveControlCenter;
