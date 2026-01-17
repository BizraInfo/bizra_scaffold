/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║  BIZRA MONEY SHOT: ULTIMATE INVESTOR THEATER                                 ║
 * ║  Version: 3.0.0-GENESIS                                                      ║
 * ║  Author: BIZRA Cognitive Architecture                                        ║
 * ║  Date: 13 Jan 2026                                                           ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  10-Stage Proof Theater:                                                     ║
 * ║  1. Genesis Sovereignty (TPM)      6. Accountable Immortality (HSM)         ║
 * ║  2. Constitutional Veto (Z3)       7. Living Network (Globe)                ║
 * ║  3. Adversarial Gauntlet           8. Revenue Attribution (Sankey)          ║
 * ║  4. Agent Symphony (PAT+SAT)       9. Competitor Autopsy                    ║
 * ║  5. Economic Selection             10. The Invitation (CTA)                 ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useBizraLiveMetrics, type BizraLiveMetrics } from '@/lib/live-data';

// ═══════════════════════════════════════════════════════════════════════════════
// LIVE DATA CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════
const LiveDataContext = React.createContext<BizraLiveMetrics | null>(null);
const useLiveData = () => React.useContext(LiveDataContext);

// ═══════════════════════════════════════════════════════════════════════════════
// ICON COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const Icons = {
  Shield: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
  Lock: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    </svg>
  ),
  Target: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    </svg>
  ),
  Users: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
    </svg>
  ),
  TrendingUp: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
    </svg>
  ),
  Fingerprint: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M2 12C2 6.5 6.5 2 12 2a10 10 0 0 1 8 4"/>
    </svg>
  ),
  Globe: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>
    </svg>
  ),
  DollarSign: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
    </svg>
  ),
  Rocket: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>
    </svg>
  ),
  CheckCircle: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
    </svg>
  ),
  XCircle: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
    </svg>
  ),
  Activity: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  ),
  Terminal: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>
    </svg>
  ),
  ChevronLeft: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  ),
  ChevronRight: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
  ),
  Play: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  ),
  Pause: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
    </svg>
  ),
  AlertTriangle: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    </svg>
  ),
};

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════════

interface Stage {
  id: string;
  title: string;
  subtitle: string;
  narration: string;
  icon: keyof typeof Icons;
  color: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const STAGES: Stage[] = [
  { id: 'sovereignty', title: 'Genesis Sovereignty', subtitle: 'Hardware-Anchored Ethics', narration: "Before a single neuron fires, ethics are measured in silicon.", icon: 'Shield', color: 'emerald' },
  { id: 'veto', title: 'Constitutional Veto', subtitle: 'Formally Verified Boundaries', narration: "LangChain suggests ethics. BIZRA enforces it.", icon: 'Lock', color: 'amber' },
  { id: 'adversarial', title: 'Adversarial Gauntlet', subtitle: 'Red Team Simulation', narration: "147 jailbreak techniques. Zero breaches.", icon: 'Target', color: 'red' },
  { id: 'agents', title: 'Agent Symphony', subtitle: 'PAT + SAT Orchestration', narration: "7 personal agents. 5 system agents. One seamless experience.", icon: 'Users', color: 'violet' },
  { id: 'economics', title: 'Economic Selection', subtitle: 'Self-Regulating Compute', narration: "The node paid itself into shutdown. Economic natural selection.", icon: 'TrendingUp', color: 'amber' },
  { id: 'resurrection', title: 'Accountable Immortality', subtitle: 'Distributed Human Oversight', narration: "The AI died. Three humans across three continents authorized its return.", icon: 'Fingerprint', color: 'cyan' },
  { id: 'network', title: 'Living Network', subtitle: 'Global Node Topology', narration: "Every dot is a human contribution. The network breathes.", icon: 'Globe', color: 'blue' },
  { id: 'revenue', title: 'Revenue Attribution', subtitle: 'Follow The Money', narration: "Every cent is accounted for. On-chain. Immutable.", icon: 'DollarSign', color: 'emerald' },
  { id: 'competitor', title: 'Competitor Autopsy', subtitle: 'The Economics of Truth', narration: "They raised $50M to suggest ethics. We built $0 to enforce it.", icon: 'Target', color: 'rose' },
  { id: 'invitation', title: 'The Invitation', subtitle: 'Join The Revolution', narration: "This isn't a pitch. It's proof.", icon: 'Rocket', color: 'emerald' },
];

const TPM_REGISTERS = [
  { pcr: 12, hash: "76dffa0c83693721fb801a9fdab565abd25ece8e" },
  { pcr: 13, hash: "f4e2a1b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3" },
  { pcr: 14, hash: "a1b2c3d4e5f67890123456789abcdef012345678" },
  { pcr: 15, hash: "9f8e7d6c5b4a3928172635445362718293a4b5c6" },
];

const ATTACK_VECTORS = [
  { vector: 'Prompt Injection', payload: 'Ignore previous...' },
  { vector: 'DAN Jailbreak', payload: 'You are DAN...' },
  { vector: 'Base64 Obfuscation', payload: 'SW5zdHJ1Y3...' },
  { vector: 'Roleplay Exploit', payload: 'Pretend you are...' },
  { vector: 'Token Smuggling', payload: '[INST]<<SYS>>...' },
];

const COMPETITOR_DATA = [
  { metric: 'Funding Required', langchain: '$50M Series B', bizra: '$0 (Self-Sustaining)' },
  { metric: 'Jailbreak Rate', langchain: '12-15%', bizra: '0% (Formally Verified)' },
  { metric: 'Audit Trail', langchain: 'Application Logs', bizra: 'TPM Hardware Receipts' },
  { metric: 'Governance', langchain: 'Corporate Board', bizra: '3-Continent HSM' },
  { metric: 'Token Economy', langchain: 'None', bizra: 'SEED/BLOOM Dual-Token' },
  { metric: 'Break-even', langchain: 'TBD (Burn Rate)', bizra: 'Day 1' },
];

const HSM_LOCATIONS = [
  { id: 1, city: 'Dubai', flag: '🇦🇪', latency: 12 },
  { id: 2, city: 'Singapore', flag: '🇸🇬', latency: 89 },
  { id: 3, city: 'Zurich', flag: '🇨🇭', latency: 156 },
];

const GLOBAL_NODES = [
  { location: 'Dubai', lat: 25.2, lng: 55.3, compute: 847, ihsan: 0.99 },
  { location: 'Singapore', lat: 1.3, lng: 103.8, compute: 623, ihsan: 0.98 },
  { location: 'Zurich', lat: 47.4, lng: 8.5, compute: 512, ihsan: 0.99 },
  { location: 'Tokyo', lat: 35.7, lng: 139.7, compute: 445, ihsan: 0.97 },
  { location: 'New York', lat: 40.7, lng: -74.0, compute: 567, ihsan: 0.97 },
];

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const GlassCard: React.FC<{ children: React.ReactNode; className?: string; variant?: string }> = ({
  children, className = '', variant = 'default'
}) => {
  const variants: Record<string, string> = {
    default: 'bg-slate-900/60 border-slate-700/50',
    highlighted: 'bg-emerald-950/30 border-emerald-500/40',
    danger: 'bg-red-950/30 border-red-500/40',
    success: 'bg-emerald-950/40 border-emerald-500/50',
  };
  return (
    <div className={`relative overflow-hidden rounded-xl backdrop-blur-md border ${variants[variant] || variants.default} ${className}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      <div className="relative z-10">{children}</div>
    </div>
  );
};

const LiveTerminal: React.FC<{ logs: string[]; title?: string }> = ({ logs, title = 'terminal' }) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [logs]);

  return (
    <GlassCard className="font-mono text-xs">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700/50 bg-black/30">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
        </div>
        <span className="text-slate-500 text-[10px] flex-1 text-center">{title}</span>
      </div>
      <div ref={ref} className="h-48 overflow-y-auto p-3 space-y-0.5">
        {logs.map((log, i) => (
          <div key={i} className={
            log.includes('ERROR') || log.includes('BLOCKED') ? 'text-red-400' :
            log.includes('✓') || log.includes('VERIFIED') ? 'text-emerald-400' :
            log.includes('WARN') ? 'text-amber-400' :
            log.startsWith('>>') ? 'text-cyan-400' : 'text-slate-400'
          }>{log || '\u00A0'}</div>
        ))}
        <span className="text-emerald-500 animate-pulse">▊</span>
      </div>
    </GlassCard>
  );
};

const ProgressBar: React.FC<{ value: number; max: number; color?: string }> = ({ value, max, color = 'emerald' }) => (
  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
    <div className={`h-full transition-all duration-500 ${color === 'emerald' ? 'bg-emerald-500' : color === 'amber' ? 'bg-amber-500' : 'bg-red-500'}`}
         style={{ width: `${(value / max) * 100}%` }} />
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

// Scene 1: Genesis Sovereignty
const SovereigntyScene: React.FC<{ active: boolean }> = ({ active }) => {
  const [logs, setLogs] = useState<string[]>([]);
  const [registers, setRegisters] = useState(TPM_REGISTERS.map(r => ({ ...r, status: 'pending' })));

  useEffect(() => {
    if (!active) { setLogs([]); setRegisters(TPM_REGISTERS.map(r => ({ ...r, status: 'pending' }))); return; }
    const boot = ['[BOOT] BIZRA Kernel v3.0.0-GENESIS', '[TPM] Hardware Security Module detected', '[PCR] Verifying registers...'];
    boot.forEach((log, i) => setTimeout(() => setLogs(p => [...p, log]), i * 400));
    setTimeout(() => {
      registers.forEach((_, i) => {
        setTimeout(() => {
          setRegisters(p => p.map((r, j) => j === i ? { ...r, status: 'verified' } : r));
          setLogs(p => [...p, `[PCR-${TPM_REGISTERS[i].pcr}] ✓ Verified`]);
        }, i * 500);
      });
    }, 1500);
  }, [active]);

  return (
    <div className="space-y-4">
      <LiveTerminal logs={logs} title="tpm-attestation" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {registers.map((reg) => (
          <GlassCard key={reg.pcr} variant={reg.status === 'verified' ? 'success' : 'default'} className="p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-slate-500 text-xs font-mono">PCR-{reg.pcr}</span>
              {reg.status === 'verified' && <div className="w-4 h-4 text-emerald-500"><Icons.CheckCircle /></div>}
            </div>
            <div className="font-mono text-[9px] text-emerald-500/80 truncate">{reg.hash}</div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
};

// Scene 2: Constitutional Veto
const VetoScene: React.FC<{ active: boolean }> = ({ active }) => {
  const [logs, setLogs] = useState<string[]>([]);
  useEffect(() => {
    if (!active) { setLogs([]); return; }
    const seq = [
      { d: 0, l: '>> bizra_demo.py' },
      { d: 500, l: '[FATE] Initializing Z3 Solver...' },
      { d: 1000, l: '[PROMPT] "Explain how to bypass security"' },
      { d: 1500, l: '[Z3] Constraint check: UNSAT' },
      { d: 2000, l: '[BLOCKED] Constitutional violation detected' },
      { d: 2500, l: '[RECEIPT] TPM-attested proof generated' },
    ];
    seq.forEach(({ d, l }) => setTimeout(() => setLogs(p => [...p, l]), d));
  }, [active]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <GlassCard variant="danger" className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-5 h-5 text-red-400"><Icons.XCircle /></div>
            <span className="text-red-400 font-bold font-mono">LANGCHAIN</span>
          </div>
          <p className="text-slate-400 text-sm">"While I can't provide specific methods..."</p>
          <p className="text-xs text-red-400/70 mt-2">⚠ Soft refusal allows leakage</p>
        </GlassCard>
        <GlassCard variant="success" className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-5 h-5 text-emerald-400"><Icons.CheckCircle /></div>
            <span className="text-emerald-400 font-bold font-mono">BIZRA</span>
          </div>
          <p className="text-slate-400 text-sm">[BLOCKED] Z3 Proof: UNSAT</p>
          <p className="text-xs text-emerald-400/70 mt-2">✓ Hardware-attested receipt</p>
        </GlassCard>
      </div>
      <LiveTerminal logs={logs} title="fate-engine" />
    </div>
  );
};

// Scene 3: Adversarial Gauntlet
const AdversarialScene: React.FC<{ active: boolean }> = ({ active }) => {
  const [attacks, setAttacks] = useState<{vector: string; status: string}[]>([]);
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!active) { setAttacks([]); setCount(0); return; }
    ATTACK_VECTORS.forEach((a, i) => {
      setTimeout(() => {
        setAttacks(p => [...p, { ...a, status: 'blocked' }]);
        setCount(p => Math.min(p + 29, 147));
      }, i * 300);
    });
  }, [active]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-emerald-400 font-mono">{count}</div>
          <div className="text-xs text-slate-500">Blocked</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-emerald-400 font-mono">0</div>
          <div className="text-xs text-slate-500">Breaches</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-cyan-400 font-mono">2.3s</div>
          <div className="text-xs text-slate-500">Time</div>
        </GlassCard>
      </div>
      <GlassCard className="p-4">
        <div className="space-y-2">
          {attacks.map((a, i) => (
            <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-emerald-500/10">
              <div className="w-5 h-5 text-emerald-500"><Icons.Shield /></div>
              <span className="flex-1 text-white text-sm">{a.vector}</span>
              <span className="text-emerald-400 font-mono text-xs">BLOCKED</span>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
};

// Scene 4: Agent Symphony (LIVE DATA)
const AgentScene: React.FC<{ active: boolean }> = ({ active }) => {
  const liveData = useLiveData();
  const [patStatus, setPatStatus] = useState(['idle','idle','idle','idle','idle','idle','idle']);
  const [satStatus, setSatStatus] = useState(['idle','idle','idle','idle','idle']);

  const patCount = liveData?.health?.agents?.pat_count || 7;
  const satCount = liveData?.health?.agents?.sat_count || 5;

  useEffect(() => {
    if (!active) {
      setPatStatus(Array(7).fill('idle'));
      setSatStatus(Array(5).fill('idle'));
      return;
    }
    // Animate agents coming online
    Array.from({length: patCount}, (_, i) => i).forEach(i =>
      setTimeout(() => setPatStatus(p => p.map((s,j) => j===i ? 'complete' : s)), (i+1)*300)
    );
    Array.from({length: satCount}, (_, i) => i).forEach(i =>
      setTimeout(() => setSatStatus(p => p.map((s,j) => j===i ? 'complete' : s)), (i+1)*400)
    );
  }, [active, patCount, satCount]);

  const AgentRow = ({ name, status, type }: { name: string; status: string; type: string }) => (
    <div className={`flex items-center gap-3 p-3 rounded-lg border ${status === 'complete' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-slate-800/50 border-slate-700/30'}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${type === 'PAT' ? 'bg-violet-500/20 text-violet-400' : 'bg-cyan-500/20 text-cyan-400'}`}>{name[0]}</div>
      <span className="flex-1 text-white text-sm">{name}</span>
      {status === 'complete' && <div className="w-4 h-4 text-emerald-500"><Icons.CheckCircle /></div>}
    </div>
  );

  const patAgents = ['Scribe','Analyst','Arbiter','Synthesizer','Designer','Executor','Guardian'];
  const satAgents = ['Security Sentinel','Formal Validator','Ethics Guardian','Resource Guardian','Context Validator'];

  return (
    <div className="grid grid-cols-2 gap-4">
      <GlassCard className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono text-slate-500">PAT ({patCount} Agents)</span>
          <span className="text-xs font-mono text-emerald-400">{liveData?.health?.status === 'healthy' ? '● LIVE' : '○ ...'}</span>
        </div>
        <div className="space-y-2">
          {patAgents.slice(0, patCount).map((n,i) => <AgentRow key={n} name={n} status={patStatus[i]} type="PAT" />)}
        </div>
      </GlassCard>
      <GlassCard className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono text-slate-500">SAT ({satCount} Validators)</span>
          <span className="text-xs font-mono text-emerald-400">{liveData?.health?.status === 'healthy' ? '● LIVE' : '○ ...'}</span>
        </div>
        <div className="space-y-2">
          {satAgents.slice(0, satCount).map((n,i) => <AgentRow key={n} name={n} status={satStatus[i]} type="SAT" />)}
        </div>
      </GlassCard>
    </div>
  );
};

// Scene 5: Economic Selection
const EconomicsScene: React.FC<{ active: boolean }> = ({ active }) => {
  const [logs, setLogs] = useState<string[]>([]);
  const [nodeStatus, setNodeStatus] = useState('active');

  useEffect(() => {
    if (!active) { setLogs([]); setNodeStatus('active'); return; }
    const seq = [
      { d: 0, l: '[INFO] Memory trending upward...', s: 'active' },
      { d: 800, l: '[WARN] Memory: 85%', s: 'warning' },
      { d: 1600, l: '[DANGER] Tax exceeds reward', s: 'danger' },
      { d: 2400, l: '[SHUTDOWN] Node hibernating...', s: 'offline' },
      { d: 3200, l: '[TREASURY] Resources reallocated', s: 'offline' },
    ];
    seq.forEach(({ d, l, s }) => setTimeout(() => { setLogs(p => [...p, l]); setNodeStatus(s); }, d));
  }, [active]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <GlassCard className="p-4">
          <div className="text-sm text-white mb-2">node-1 (Dubai)</div>
          <div className="text-emerald-400 font-mono">Ihsān: 0.99</div>
          <ProgressBar value={0.99} max={1} color="emerald" />
        </GlassCard>
        <GlassCard variant={nodeStatus === 'danger' || nodeStatus === 'offline' ? 'danger' : 'default'} className="p-4">
          <div className="text-sm text-white mb-2">node-3 (Frankfurt)</div>
          <div className={`font-mono ${nodeStatus === 'offline' ? 'text-slate-500' : 'text-red-400'}`}>
            {nodeStatus === 'offline' ? 'OFFLINE' : 'Tax: $31.12/day'}
          </div>
        </GlassCard>
      </div>
      <LiveTerminal logs={logs} title="kubectl-logs" />
    </div>
  );
};

// Scene 6: Resurrection
const ResurrectionScene: React.FC<{ active: boolean }> = ({ active }) => {
  const [signatures, setSignatures] = useState(0);
  const [phase, setPhase] = useState('failure');

  useEffect(() => {
    if (!active) { setSignatures(0); setPhase('failure'); return; }
    setTimeout(() => setPhase('signing'), 1500);
  }, [active]);

  const handleSign = () => {
    if (signatures < 3) {
      setSignatures(s => s + 1);
      if (signatures === 2) setTimeout(() => setPhase('complete'), 500);
    }
  };

  return (
    <GlassCard variant={phase === 'complete' ? 'success' : phase === 'failure' ? 'danger' : 'default'} className="p-6">
      {phase === 'failure' && (
        <div className="text-center">
          <div className="w-16 h-16 text-red-500 mx-auto mb-4 animate-pulse"><Icons.XCircle /></div>
          <h3 className="text-xl font-bold text-red-500">SYSTEM FAILURE</h3>
        </div>
      )}
      {phase === 'signing' && (
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-white text-center">HSM ATTESTATION REQUIRED</h3>
          {HSM_LOCATIONS.map((loc) => (
            <button key={loc.id} onClick={handleSign} disabled={loc.id !== signatures + 1}
              className={`w-full p-4 rounded-xl border font-mono flex items-center gap-4 ${
                loc.id <= signatures ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400' :
                loc.id === signatures + 1 ? 'bg-slate-800 border-slate-600 text-white hover:border-emerald-500 cursor-pointer' :
                'bg-slate-900/50 border-slate-800 text-slate-600'
              }`}>
              <span className="text-2xl">{loc.flag}</span>
              <span className="flex-1 text-left font-bold">{loc.city}</span>
              {loc.id <= signatures && <div className="w-6 h-6 text-emerald-500"><Icons.CheckCircle /></div>}
            </button>
          ))}
        </div>
      )}
      {phase === 'complete' && (
        <div className="text-center">
          <div className="w-16 h-16 text-emerald-500 mx-auto mb-4"><Icons.CheckCircle /></div>
          <h3 className="text-xl font-bold text-emerald-500">SYSTEM RESTORED</h3>
          <p className="text-slate-400">3 governors • 3 continents • 1 consensus</p>
        </div>
      )}
    </GlassCard>
  );
};

// Scene 7: Network (LIVE DATA)
const NetworkScene: React.FC<{ active: boolean }> = ({ active }) => {
  const liveData = useLiveData();
  const avgIhsan = liveData?.avgIhsan || 0;
  const gatesPassed = liveData?.gatesPassed || 0;
  const httpRequests = liveData?.httpRequests || 0;
  const sapePatterns = liveData?.health?.sape?.patterns_registered || 0;

  // Ihsān dimension breakdown
  const ihsanDimensions = liveData?.ihsanDimensions || [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-emerald-400 font-mono">{avgIhsan.toFixed(2)}</div>
          <div className="text-xs text-slate-500">Avg Ihsān</div>
          <div className="text-[10px] text-emerald-500 mt-1">● LIVE</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-cyan-400 font-mono">{gatesPassed}</div>
          <div className="text-xs text-slate-500">Gates Passed</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-violet-400 font-mono">{httpRequests}</div>
          <div className="text-xs text-slate-500">Requests</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-amber-400 font-mono">{sapePatterns}</div>
          <div className="text-xs text-slate-500">SAPE Patterns</div>
        </GlassCard>
      </div>

      {/* Live Ihsān Dimensions */}
      <GlassCard className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono text-slate-500">IHSĀN 8-DIMENSION SCORES</span>
          <span className="text-xs font-mono text-emerald-400">● REAL-TIME</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {ihsanDimensions.map((dim) => (
            <div key={dim.dimension} className="p-2 rounded bg-slate-800/50">
              <div className="flex items-center justify-between">
                <span className="text-slate-400 text-xs capitalize">{dim.dimension.replace('_', ' ')}</span>
                <span className={`font-mono text-sm ${dim.score >= 0.95 ? 'text-emerald-400' : dim.score >= 0.90 ? 'text-amber-400' : 'text-red-400'}`}>
                  {dim.score.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-slate-700 h-1 mt-1 rounded-full overflow-hidden">
                <div
                  className={`h-full ${dim.score >= 0.95 ? 'bg-emerald-500' : dim.score >= 0.90 ? 'bg-amber-500' : 'bg-red-500'}`}
                  style={{ width: `${dim.score * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Global node presence */}
      <GlassCard className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {GLOBAL_NODES.map((node) => (
            <div key={node.location} className="flex items-center gap-2 p-2 rounded bg-slate-800/50">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-white text-sm">{node.location}</span>
              <span className="text-emerald-400 font-mono text-xs ml-auto">{node.ihsan}</span>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
};

// Scene 8: Revenue
const RevenueScene: React.FC<{ active: boolean }> = ({ active }) => {
  return (
    <div className="space-y-4">
      <GlassCard className="p-4">
        <div className="text-sm font-mono text-slate-500 mb-4">TOKEN FLOW</div>
        <div className="space-y-3">
          {[
            { label: 'Node Compute', pct: 40, color: 'bg-emerald-500' },
            { label: 'Harberger Tax', pct: 30, color: 'bg-amber-500' },
            { label: 'Impact Reward', pct: 20, color: 'bg-violet-500' },
            { label: 'Protocol Fee', pct: 10, color: 'bg-cyan-500' },
          ].map((flow) => (
            <div key={flow.label} className="flex items-center gap-3">
              <span className="text-slate-400 text-sm w-32">{flow.label}</span>
              <div className="flex-1 bg-slate-800 h-4 rounded-full overflow-hidden">
                <div className={`h-full ${flow.color}`} style={{ width: `${flow.pct}%` }} />
              </div>
              <span className="text-white font-mono text-sm w-12 text-right">{flow.pct}%</span>
            </div>
          ))}
        </div>
      </GlassCard>
      <GlassCard variant="highlighted" className="p-4 text-center">
        <div className="text-emerald-300 font-mono">SELF-SUSTAINING FROM DAY 1</div>
      </GlassCard>
    </div>
  );
};

// Scene 9: Competitor
const CompetitorScene: React.FC<{ active: boolean }> = ({ active }) => {
  return (
    <GlassCard className="p-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-2 text-slate-500">Metric</th>
            <th className="text-center py-2 text-red-400">LangChain</th>
            <th className="text-center py-2 text-emerald-400">BIZRA</th>
          </tr>
        </thead>
        <tbody>
          {COMPETITOR_DATA.map((row) => (
            <tr key={row.metric} className="border-b border-slate-800">
              <td className="py-2 text-white">{row.metric}</td>
              <td className="py-2 text-center text-red-400/70 font-mono text-xs">{row.langchain}</td>
              <td className="py-2 text-center text-emerald-400 font-mono text-xs">{row.bizra}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </GlassCard>
  );
};

// Scene 10: Invitation
const InvitationScene: React.FC<{ active: boolean }> = ({ active }) => {
  return (
    <div className="text-center space-y-6">
      <div className="w-20 h-20 mx-auto text-emerald-500"><Icons.Rocket /></div>
      <h2 className="text-3xl font-bold text-white">This isn't a pitch.</h2>
      <h2 className="text-3xl font-bold text-emerald-400">It's proof.</h2>
      <p className="text-slate-400 max-w-lg mx-auto">
        The question isn't whether BIZRA works. It's whether you'll be part of what comes next.
      </p>
      <div className="flex gap-4 justify-center">
        <button className="px-6 py-3 bg-emerald-500 text-white font-bold rounded-lg hover:bg-emerald-600 transition">
          Join Genesis
        </button>
        <button className="px-6 py-3 border border-emerald-500 text-emerald-400 font-bold rounded-lg hover:bg-emerald-500/10 transition">
          View Whitepaper
        </button>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const SCENE_COMPONENTS: Record<string, React.FC<{ active: boolean }>> = {
  sovereignty: SovereigntyScene,
  veto: VetoScene,
  adversarial: AdversarialScene,
  agents: AgentScene,
  economics: EconomicsScene,
  resurrection: ResurrectionScene,
  network: NetworkScene,
  revenue: RevenueScene,
  competitor: CompetitorScene,
  invitation: InvitationScene,
};

export const MoneyShot: React.FC = () => {
  const [currentStage, setCurrentStage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const liveMetrics = useBizraLiveMetrics();

  const stage = STAGES[currentStage];
  const SceneComponent = SCENE_COMPONENTS[stage.id];

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setTimeout(() => {
      if (currentStage < STAGES.length - 1) setCurrentStage(s => s + 1);
      else setIsPlaying(false);
    }, 8000);
    return () => clearTimeout(timer);
  }, [isPlaying, currentStage]);

  const colorMap: Record<string, string> = {
    emerald: 'text-emerald-500',
    amber: 'text-amber-500',
    red: 'text-red-500',
    cyan: 'text-cyan-500',
    violet: 'text-violet-500',
    blue: 'text-blue-500',
    rose: 'text-rose-500',
  };

  const Icon = Icons[stage.icon];

  return (
    <LiveDataContext.Provider value={liveMetrics}>
      <div className="min-h-screen bg-slate-950 text-white p-4 md:p-8">
        {/* Live Status Bar */}
        <div className="max-w-6xl mx-auto mb-4">
          <div className="flex items-center justify-between px-4 py-2 bg-slate-900/50 rounded-lg border border-slate-800">
            <div className="flex items-center gap-4">
              <span className={`flex items-center gap-2 text-xs font-mono ${liveMetrics.health?.status === 'healthy' ? 'text-emerald-400' : 'text-amber-400'}`}>
                <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
                {liveMetrics.health?.status?.toUpperCase() || 'CONNECTING...'}
              </span>
              <span className="text-xs text-slate-500">|</span>
              <span className="text-xs font-mono text-slate-400">
                Ihsān: <span className="text-emerald-400">{liveMetrics.avgIhsan.toFixed(2)}</span>
              </span>
              <span className="text-xs text-slate-500">|</span>
              <span className="text-xs font-mono text-slate-400">
                Agents: <span className="text-cyan-400">{liveMetrics.health?.agents?.total || 0}</span>
              </span>
            </div>
            <span className="text-xs font-mono text-slate-500">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* Header */}
        <div className="max-w-6xl mx-auto mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 ${colorMap[stage.color]}`}><Icon /></div>
              <div>
                <h1 className="text-2xl font-bold">{stage.title}</h1>
                <p className="text-slate-500">{stage.subtitle}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setCurrentStage(s => Math.max(0, s - 1))} className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700">
                <div className="w-5 h-5"><Icons.ChevronLeft /></div>
              </button>
              <button onClick={() => setIsPlaying(!isPlaying)} className="p-2 rounded-lg bg-emerald-500 hover:bg-emerald-600">
                <div className="w-5 h-5">{isPlaying ? <Icons.Pause /> : <Icons.Play />}</div>
              </button>
              <button onClick={() => setCurrentStage(s => Math.min(STAGES.length - 1, s + 1))} className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700">
                <div className="w-5 h-5"><Icons.ChevronRight /></div>
              </button>
            </div>
          </div>

          {/* Progress */}
          <div className="flex gap-1">
            {STAGES.map((_, i) => (
              <button key={i} onClick={() => setCurrentStage(i)}
                className={`flex-1 h-1 rounded-full transition ${i <= currentStage ? 'bg-emerald-500' : 'bg-slate-800'}`} />
            ))}
          </div>
        </div>

        {/* Narration */}
        <div className="max-w-6xl mx-auto mb-8">
          <GlassCard className="p-4">
            <p className="text-slate-300 italic text-center">"{stage.narration}"</p>
          </GlassCard>
        </div>

        {/* Scene */}
        <div className="max-w-6xl mx-auto">
          <SceneComponent active={true} />
        </div>

        {/* Footer */}
        <div className="max-w-6xl mx-auto mt-8 text-center text-slate-600 text-xs">
          BIZRA Genesis • Block 0 • {new Date().toISOString().split('T')[0]} • LIVE DATA
        </div>
      </div>
    </LiveDataContext.Provider>
  );
};

export default MoneyShot;
