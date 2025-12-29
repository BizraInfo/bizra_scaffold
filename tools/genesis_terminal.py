#!/usr/bin/env python3
"""
BIZRA GENESIS TERMINAL
════════════════════════════════════════════════════════════════════════════════
The "Glass Cockpit" for the Genesis Orchestrator.

Real-time TUI (Terminal User Interface) that visualizes the autonomous
reasoning process. This is the observability layer that transforms the
abstract "thinking" into visible, comprehensible data flows.

"A peak masterpiece system is not a black box.
 It requires a Glass Cockpit—a high-density, real-time interface
 that allows you to watch the Graph of Thoughts expand,
 monitor the SNR fluctuations, and verify Ihsān compliance as it happens."

FEATURES:
────────────────────────────────────────────────────────────────────────────────
1. THOUGHT STREAM: Watch the Graph of Thoughts expand in real-time
2. SNR MONITOR: Live sparklines tracking signal-to-noise ratios
3. LENS ARRAY: Visualizing active interdisciplinary domains
4. WISDOM VAULT: Access to crystallized insights from Giants Protocol

DESIGN PHILOSOPHY:
────────────────────────────────────────────────────────────────────────────────
- High Data Density: Maximal information, minimal chrome
- Reactive: Updates instantaneously with the Orchestrator's cycle
- Cybernetic Aesthetic: "Matrix-style" visibility into the machine mind
- Rosé Pine Theme: Elegant, low-strain color palette

KEYBINDINGS:
────────────────────────────────────────────────────────────────────────────────
- q: Quit the terminal
- space: Trigger a new processing cycle
- t: Toggle between dark/light themes
- w: Focus wisdom vault
- c: Clear thought stream

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Optional

# Add repo root to path for imports
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Check Textual availability
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Grid, Horizontal, Vertical
    from textual.widgets import (
        Header, Footer, Static, Label, Button, 
        Tree, RichLog, ProgressBar,
    )
    from textual.reactive import reactive
    from textual.binding import Binding
    from textual.message import Message
    from rich.text import Text
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    print("Textual not installed. Install with: pip install textual>=0.47.0")

# Check Sparkline availability (added in Textual 0.40+)
SPARKLINE_AVAILABLE = False
if TEXTUAL_AVAILABLE:
    try:
        from textual.widgets import Sparkline
        SPARKLINE_AVAILABLE = True
    except ImportError:
        pass

# Import Genesis modules
try:
    from core.genesis.genesis_events import (
        GenesisEvent,
        GenesisEventType,
        GenesisEventListener,
    )
    from core.genesis.genesis_orchestrator_streaming import StreamingGenesisOrchestrator
    from core.genesis.genesis_orchestrator import WisdomRepository
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    print(f"Genesis modules not available: {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default problem for demo
DEFAULT_PROBLEM = (
    "Design a self-correcting governance mechanism for the Genesis Node "
    "that maintains Ihsān compliance under adversarial conditions while "
    "optimizing for network consensus efficiency"
)

# SNR thresholds for color coding
SNR_HIGH = 0.80
SNR_MEDIUM = 0.50


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

if TEXTUAL_AVAILABLE:

    class SnrMonitor(Static):
        """
        Real-time SNR visualization widget.
        
        Displays a 60-second rolling window of SNR readings as a sparkline
        (or text-based bar chart if Sparkline not available).
        """
        
        # Reactive data for the sparkline
        data: reactive[list] = reactive(lambda: [0.0] * 60)
        current_snr: reactive[float] = reactive(0.0)
        
        def compose(self) -> ComposeResult:
            yield Label("📊 SNR SIGNAL (60s Window)", classes="panel-header")
            if SPARKLINE_AVAILABLE:
                yield Sparkline(self.data, summary_function=max, id="snr-sparkline")
            else:
                yield Label("▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁", id="snr-bars")
            yield Label("Current: 0.00 | Level: ---", id="snr-current")
        
        def add_reading(self, value: float) -> None:
            """Add a new SNR reading to the monitor."""
            self.current_snr = value
            new_data = list(self.data[1:]) + [value]
            self.data = new_data
            
            # Update sparkline if available
            if SPARKLINE_AVAILABLE:
                sparkline = self.query_one("#snr-sparkline", Sparkline)
                sparkline.data = new_data
            else:
                # Text-based visualization
                bars = self._values_to_bars(new_data[-20:])
                self.query_one("#snr-bars", Label).update(bars)
            
            # Update current value with color
            level, color = self._get_level_color(value)
            self.query_one("#snr-current", Label).update(
                f"Current: [{color}]{value:.2f}[/] | Level: [{color}]{level}[/]"
            )
        
        def _get_level_color(self, value: float) -> tuple[str, str]:
            """Get level name and color for SNR value."""
            if value >= SNR_HIGH:
                return "HIGH", "green"
            elif value >= SNR_MEDIUM:
                return "MEDIUM", "yellow"
            else:
                return "LOW", "red"
        
        def _values_to_bars(self, values: list) -> str:
            """Convert values to Unicode bar chart."""
            bars = "▁▂▃▄▅▆▇█"
            result = []
            for v in values:
                idx = int(v * (len(bars) - 1))
                idx = max(0, min(len(bars) - 1, idx))
                result.append(bars[idx])
            return "".join(result)

    class LensArray(Static):
        """
        Active Domain Lenses visualization.
        
        Shows the 6 interdisciplinary lenses with highlighting
        for the currently active lens.
        """
        
        LENSES = ["CRYPTO", "ECON", "PHIL", "GOV", "SYS", "COG"]
        active_lens: reactive[str] = reactive("")
        
        def compose(self) -> ComposeResult:
            yield Label("🔬 DOMAIN LENSES", classes="panel-header")
            with Grid(classes="lens-grid"):
                for lens in self.LENSES:
                    yield Label(lens, id=f"lens-{lens}", classes="lens-item")
        
        def highlight(self, lens_name: str) -> None:
            """Highlight the specified lens."""
            self.active_lens = lens_name.upper()[:3]  # Match abbreviation
            
            # Reset all lenses
            for lens in self.LENSES:
                widget = self.query_one(f"#lens-{lens}", Label)
                widget.remove_class("active")
            
            # Activate specific lens
            if self.active_lens in self.LENSES:
                self.query_one(f"#lens-{self.active_lens}", Label).add_class("active")
            
            # Also check full names
            lens_map = {
                "CRYPTO": "CRYPTO", "CRYPTOGRAPHY": "CRYPTO",
                "ECON": "ECON", "ECONOMICS": "ECON",
                "PHIL": "PHIL", "PHILOSOPHY": "PHIL",
                "GOV": "GOV", "GOVERNANCE": "GOV",
                "SYS": "SYS", "SYSTEMS": "SYS",
                "COG": "COG", "COGNITIVE": "COG",
            }
            mapped = lens_map.get(lens_name.upper())
            if mapped:
                self.query_one(f"#lens-{mapped}", Label).add_class("active")
        
        def clear_highlight(self) -> None:
            """Clear all highlights."""
            for lens in self.LENSES:
                self.query_one(f"#lens-{lens}", Label).remove_class("active")

    class ThoughtStream(RichLog):
        """
        Real-time thought stream visualization.
        
        Displays reasoning steps as they occur with depth indentation
        and SNR-based coloring.
        """
        
        def log_thought(
            self,
            content: str,
            snr: float,
            depth: int,
            node_id: str = "",
        ) -> None:
            """Log a thought with formatting."""
            indent = "  " * depth
            icon = "🧠" if depth == 0 else "├─" if depth == 1 else "│ └─"
            
            # Color based on SNR
            if snr >= SNR_HIGH:
                color = "green"
                indicator = "🟢"
            elif snr >= SNR_MEDIUM:
                color = "yellow"
                indicator = "🟡"
            else:
                color = "red"
                indicator = "🔴"
            
            # Truncate content
            display_content = content[:60] + "..." if len(content) > 60 else content
            
            self.write(
                f"{indent}{icon} {indicator} [{color}]{display_content}[/] "
                f"[dim](SNR: {snr:.2f})[/]"
            )
        
        def log_phase(self, phase: str, progress: float) -> None:
            """Log a phase transition."""
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            self.write(f"\n[bold cyan]▶ {phase}[/] [{progress_bar}] {progress*100:.0f}%")
        
        def log_event(self, event_type: str, data: dict) -> None:
            """Log a generic event."""
            if "COMPLETE" in event_type:
                self.write(f"  [green]✓[/] {event_type.replace('_', ' ').title()}")
            elif "START" in event_type:
                self.write(f"  [cyan]→[/] {event_type.replace('_', ' ').title()}")
            elif "PRUNED" in event_type:
                self.write(f"  [red]✗[/] Pruned: SNR too low")

    class WisdomVault(Tree):
        """
        Hierarchical view of crystallized wisdom.
        
        Displays the wisdom repository as an expandable tree.
        """
        
        def __init__(self, *args, **kwargs):
            super().__init__("📚 Wisdom Vault", *args, **kwargs)
            self.show_root = True
        
        def add_cycle(self, cycle_id: str, results: dict) -> None:
            """Add a completed cycle to the tree."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            node = self.root.add(f"Cycle {cycle_id} @ {timestamp}", expand=True)
            
            # Add attestation hash
            if "hash" in results:
                node.add_leaf(f"Hash: {results['hash'][:24]}...")
            
            # Add node binding
            if "node_id" in results and results["node_id"]:
                node.add_leaf(f"Node: {results['node_id']}")
            
            # Add crystallized count
            if "crystallized" in results:
                node.add_leaf(f"Crystallized: {results['crystallized']} insights")


    # =========================================================================
    # MAIN APPLICATION
    # =========================================================================

    class GenesisTerminal(App):
        """
        The BIZRA Genesis Terminal - Glass Cockpit for the Autonomous Engine.
        
        Provides real-time observability into:
        - Graph of Thoughts expansion
        - SNR fluctuations
        - Ihsān compliance
        - Wisdom crystallization
        """
        
        # Rosé Pine inspired theme
        CSS = """
        Screen {
            layout: grid;
            grid-size: 3 3;
            grid-rows: 1fr 4fr 1fr;
            grid-columns: 1fr 2fr 1fr;
            background: #191724;
            color: #e0def4;
        }

        .panel-header {
            background: #26233a;
            color: #9ccfd8;
            padding: 0 1;
            text-style: bold;
            width: 100%;
        }

        #snr-monitor {
            border: solid #31748f;
            row-span: 1;
            column-span: 1;
            padding: 1;
        }

        #snr-sparkline {
            color: #9ccfd8;
            height: 3;
        }

        #snr-bars {
            color: #9ccfd8;
        }

        #lens-array {
            border: solid #c4a7e7;
            row-span: 1;
            column-span: 1;
            padding: 1;
        }

        .lens-grid {
            grid-size: 3 2;
            grid-gutter: 1;
            margin: 1;
            height: auto;
        }

        .lens-item {
            content-align: center middle;
            background: #26233a;
            color: #6e6a86;
            padding: 0 1;
            height: 1;
        }

        .lens-item.active {
            background: #ebbcba;
            color: #191724;
            text-style: bold;
        }

        #status-panel {
            border: solid #f6c177;
            row-span: 1;
            column-span: 1;
            padding: 1;
        }

        #thought-stream {
            border: solid #eb6f92;
            row-span: 2;
            column-span: 2;
            background: #1f1d2e;
        }

        #wisdom-vault {
            border: solid #f6c177;
            row-span: 2;
            column-span: 1;
            background: #1f1d2e;
        }

        #controls {
            column-span: 3;
            align: center middle;
            border-top: solid #ebbcba;
            height: 100%;
            layout: horizontal;
        }

        Button {
            margin: 0 1;
        }

        #btn-ignite {
            background: #31748f;
            color: #e0def4;
        }

        #btn-ignite:hover {
            background: #9ccfd8;
            color: #191724;
        }

        #btn-clear {
            background: #26233a;
        }

        #status-line {
            margin: 0 2;
            color: #908caa;
        }

        Header {
            background: #26233a;
            color: #e0def4;
        }

        Footer {
            background: #26233a;
        }
        """
        
        BINDINGS = [
            Binding("q", "quit", "Quit", show=True),
            Binding("space", "ignite", "Ignite", show=True),
            Binding("c", "clear", "Clear", show=True),
            Binding("w", "focus_wisdom", "Wisdom", show=True),
        ]
        
        # State
        cycle_count: reactive[int] = reactive(0)
        is_running: reactive[bool] = reactive(False)
        
        def compose(self) -> ComposeResult:
            """Compose the application layout."""
            yield Header(show_clock=True)
            
            # Row 1: Monitors
            yield SnrMonitor(id="snr-monitor")
            yield LensArray(id="lens-array")
            yield Static(
                "[bold]Status[/]\n\nReady for input...",
                id="status-panel",
                classes="",
            )
            
            # Row 2: Main content
            yield ThoughtStream(id="thought-stream", highlight=True, markup=True)
            yield WisdomVault(id="wisdom-vault")
            
            # Row 3: Controls
            with Container(id="controls"):
                yield Button("🔥 IGNITE GENESIS ENGINE", id="btn-ignite", variant="primary")
                yield Button("🗑 Clear", id="btn-clear", variant="default")
                yield Label("Awaiting Architect input...", id="status-line")
            
            yield Footer()
        
        def on_mount(self) -> None:
            """Initialize on mount."""
            self.title = "BIZRA AEON OMEGA // GENESIS TERMINAL"
            self.sub_title = "Glass Cockpit v1.0.0"
            
            # Initialize orchestrator
            self.orchestrator: Optional[StreamingGenesisOrchestrator] = None
            
            if ORCHESTRATOR_AVAILABLE:
                try:
                    self.orchestrator = StreamingGenesisOrchestrator(
                        beam_width=4,
                        max_depth=3,
                        fail_closed=False,
                        emit_delay=0.05,
                    )
                    self._update_status("Orchestrator Online. Ready to Ignite.")
                except Exception as e:
                    self._update_status(f"[red]Orchestrator Error: {e}[/]")
            else:
                self._update_status("[yellow]Mock Mode (Orchestrator not loaded)[/]")
        
        def _update_status(self, message: str) -> None:
            """Update status line."""
            self.query_one("#status-line", Label).update(message)
        
        def _update_status_panel(self, content: str) -> None:
            """Update status panel."""
            self.query_one("#status-panel", Static).update(content)
        
        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            if event.button.id == "btn-ignite":
                if not self.is_running:
                    self.run_worker(self.run_genesis_cycle())
            elif event.button.id == "btn-clear":
                self.query_one("#thought-stream", ThoughtStream).clear()
        
        def action_ignite(self) -> None:
            """Keybinding action for ignite."""
            if not self.is_running:
                self.run_worker(self.run_genesis_cycle())
        
        def action_clear(self) -> None:
            """Keybinding action for clear."""
            self.query_one("#thought-stream", ThoughtStream).clear()
        
        def action_focus_wisdom(self) -> None:
            """Focus the wisdom vault."""
            self.query_one("#wisdom-vault").focus()
        
        async def run_genesis_cycle(self) -> None:
            """Run a complete genesis processing cycle."""
            self.is_running = True
            self.cycle_count += 1
            
            btn = self.query_one("#btn-ignite", Button)
            btn.disabled = True
            
            stream = self.query_one("#thought-stream", ThoughtStream)
            snr_mon = self.query_one("#snr-monitor", SnrMonitor)
            lens_arr = self.query_one("#lens-array", LensArray)
            vault = self.query_one("#wisdom-vault", WisdomVault)
            
            problem = DEFAULT_PROBLEM
            
            stream.write(f"\n{'═' * 60}")
            stream.write(f"[bold white]CYCLE #{self.cycle_count}[/]")
            stream.write(f"[dim]Problem:[/] {problem[:50]}...")
            stream.write(f"{'─' * 60}\n")
            
            self._update_status("[yellow]Running Autonomous Cycle...[/]")
            
            last_phase = ""
            attestation_data = {}
            crystallized_count = 0
            
            if self.orchestrator:
                # Real orchestrator mode
                try:
                    async for event in self.orchestrator.process_streaming(problem):
                        # Update progress
                        self._update_status(
                            f"[cyan]{event.phase}[/] ({event.progress*100:.0f}%)"
                        )
                        
                        # Log phase transitions
                        if event.phase != last_phase:
                            stream.log_phase(event.phase, event.progress)
                            last_phase = event.phase
                        
                        # Handle specific event types
                        if event.type == GenesisEventType.LENS_ACTIVATED:
                            lens_arr.highlight(event.data.get("lens", ""))
                        
                        elif event.type == GenesisEventType.THOUGHT_NODE_CREATED:
                            stream.log_thought(
                                content=event.data.get("content", ""),
                                snr=event.data.get("snr", 0),
                                depth=event.data.get("depth", 0),
                                node_id=event.data.get("node_id", ""),
                            )
                            snr_mon.add_reading(event.data.get("snr", 0))
                        
                        elif event.type == GenesisEventType.SNR_SCORE_COMPUTED:
                            snr_mon.add_reading(event.data.get("snr", 0))
                        
                        elif event.type == GenesisEventType.THOUGHT_NODE_PRUNED:
                            stream.log_event("PRUNED", event.data)
                        
                        elif event.type == GenesisEventType.CRYSTAL_INSIGHT_ADDED:
                            crystallized_count += 1
                        
                        elif event.type == GenesisEventType.ATTEST_COMPLETE:
                            attestation_data = {
                                "hash": event.data.get("hash", ""),
                                "node_id": event.data.get("node_id"),
                                "crystallized": crystallized_count,
                            }
                        
                        # Small delay for visual pacing
                        await asyncio.sleep(0.02)
                    
                except Exception as e:
                    stream.write(f"\n[red]Error: {e}[/]")
                    self._update_status(f"[red]Error: {e}[/]")
            else:
                # Mock mode - simulate events
                await self._run_mock_cycle(stream, snr_mon, lens_arr)
                attestation_data = {
                    "hash": "mock_" + hex(hash(problem))[-12:],
                    "node_id": "mock_node0",
                    "crystallized": 2,
                }
            
            # Finalize
            stream.write(f"\n{'─' * 60}")
            stream.write(f"[bold green]✓ CYCLE #{self.cycle_count} COMPLETE[/]")
            
            if attestation_data:
                stream.write(f"[dim]Attestation:[/] {attestation_data.get('hash', 'N/A')}")
                vault.add_cycle(str(self.cycle_count), attestation_data)
            
            lens_arr.clear_highlight()
            self._update_status("Cycle Complete. Ready.")
            self._update_status_panel(
                f"[bold]Status[/]\n\n"
                f"Cycles: {self.cycle_count}\n"
                f"Last SNR: {snr_mon.current_snr:.2f}\n"
                f"Crystallized: {crystallized_count}"
            )
            
            btn.disabled = False
            self.is_running = False
        
        async def _run_mock_cycle(
            self,
            stream: ThoughtStream,
            snr_mon: SnrMonitor,
            lens_arr: LensArray,
        ) -> None:
            """Run a mock cycle for demo purposes."""
            import random
            
            phases = [
                ("Ignition", 0.05),
                ("Lens Analysis", 0.20),
                ("Giants Protocol", 0.30),
                ("Expansion", 0.70),
                ("SNR Gating", 0.85),
                ("Crystallization", 0.95),
                ("Binding", 1.0),
            ]
            
            lenses = ["CRYPTO", "ECON", "PHIL", "GOV", "SYS", "COG"]
            
            for phase, progress in phases:
                stream.log_phase(phase, progress)
                self._update_status(f"[cyan]{phase}[/] ({progress*100:.0f}%)")
                
                if phase == "Lens Analysis":
                    for lens in lenses:
                        lens_arr.highlight(lens)
                        await asyncio.sleep(0.1)
                
                elif phase == "Expansion":
                    for depth in range(3):
                        for _ in range(3):
                            snr = random.uniform(0.4, 0.95)
                            snr_mon.add_reading(snr)
                            stream.log_thought(
                                content=f"Analyzing implications at depth {depth}...",
                                snr=snr,
                                depth=depth,
                            )
                            await asyncio.sleep(0.1)
                
                else:
                    await asyncio.sleep(0.3)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point for the Genesis Terminal."""
    if not TEXTUAL_AVAILABLE:
        print("Error: Textual is required for the Genesis Terminal.")
        print("Install with: pip install textual>=0.47.0")
        return 1
    
    app = GenesisTerminal()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
