#!/usr/bin/env python3
"""
ATIS v5.0 — Agent Monitor (Phase 2: Real-Time Monitoring)
Displays live dashboard of active agent status and development state.
Auto-refreshes every 2 seconds to show agent transitions.
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Try to import rich for nice formatting, fallback to basic if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class AgentMonitor:
    """Real-time monitor for agent activity"""
    
    def __init__(self):
        self.state_file = Path("dev_state.json")
        self.refresh_interval = 2  # seconds
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def load_state(self) -> dict:
        """Load current dev_state.json"""
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def display_with_rich(self, data: dict):
        """Display with rich formatting (fancy output)"""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Title
        self.console.print(Panel(
            "[bold cyan]ATIS v5.0 — Multi-Agent Development Monitor[/bold cyan]",
            style="bold blue"
        ))
        
        # Agent Status Table
        table = Table(title="Agent Status", show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan", width=15)
        table.add_column("Status", style="green", width=12)
        table.add_column("Progress", style="yellow", width=10)
        table.add_column("Current Task", style="white", width=40)
        
        for agent_name, agent_data in data.get('agents', {}).items():
            status = agent_data.get('status', 'idle').upper()
            progress = agent_data.get('progress', 0)
            task = agent_data.get('last_task', 'N/A')[:35]
            
            # Color status
            if status == 'ACTIVE':
                status_colored = "[bold green]🟢 ACTIVE[/bold green]"
            elif status == 'IDLE':
                status_colored = "[yellow]⚪ IDLE[/yellow]"
            else:
                status_colored = f"[white]{status}[/white]"
            
            progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
            table.add_row(agent_name, status_colored, f"[{progress_bar}]", task)
        
        self.console.print(table)
        
        # Metrics Panel
        metrics = data.get('metrics', {})
        vix = metrics.get('global_vix', 'N/A')
        vix_str = f"[red]{vix} 🛑[/red]" if isinstance(vix, (int, float)) and vix > 25 else f"[green]{vix} ✅[/green]"
        
        metrics_text = f"""
[bold cyan]📊 Real-Time Metrics:[/bold cyan]
  • Active Agent: [bold]{data.get('active_agent', 'None')}[/bold]
  • Current Task: {data.get('current_task', 'N/A')}
  • Phase: {data.get('phase', 1)}
  
[bold cyan]🎯 Performance:[/bold cyan]
  • Baseline F1: {metrics.get('baseline_f1', 0.65):.4f}
  • Current F1: {metrics.get('current_f1', 0.0):.4f}
  
[bold cyan]🌍 Global Signals:[/bold cyan]
  • Data Alignment: {metrics.get('data_alignment', 'pending')}
  • Global VIX: {vix_str}
  • US Gap: {metrics.get('us_gap', 'N/A')}
  • Forex Strength: {metrics.get('forex_strength', 'N/A')}
  
[bold cyan]🛡️ Safety:[/bold cyan]
  • Circuit Breaker: {"[bold green]ENABLED ✅[/bold green]" if data.get('circuit_breakers', {}).get('enabled') else "[bold red]DISABLED ❌[/bold red]"}
  • VIX Threshold: {data.get('circuit_breakers', {}).get('vix_threshold', 25)}
  • SL Max: {data.get('circuit_breakers', {}).get('sl_percent', 0.30)*100:.1f}%
"""
        self.console.print(Panel(metrics_text, style="bold blue"))
        
        # Footer
        timestamp = data.get('timestamp', datetime.now().isoformat())
        self.console.print(f"[dim]Last updated: {timestamp} | Refresh: {self.refresh_interval}s[/dim]")
    
    def display_basic(self, data: dict):
        """Display with basic formatting (fallback for no rich)"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("="*70)
        print("ATIS v5.0 — MULTI-AGENT DEVELOPMENT MONITOR")
        print("="*70)
        print()
        
        print("AGENT STATUS:")
        print("-" * 70)
        for agent_name, agent_data in data.get('agents', {}).items():
            status = agent_data.get('status', 'idle').upper()
            progress = agent_data.get('progress', 0)
            task = agent_data.get('last_task', 'N/A')[:40]
            print(f"  {agent_name:15s} [{status:6s}] [{progress:3d}%] {task}")
        
        print()
        print("REAL-TIME METRICS:")
        print("-" * 70)
        print(f"  Active Agent: {data.get('active_agent', 'None')}")
        print(f"  Current Task: {data.get('current_task', 'N/A')}")
        print(f"  Phase: {data.get('phase', 1)}")
        print()
        
        metrics = data.get('metrics', {})
        print("  PERFORMANCE:")
        print(f"    Baseline F1: {metrics.get('baseline_f1', 0.65):.4f}")
        print(f"    Current F1:  {metrics.get('current_f1', 0.0):.4f}")
        print()
        
        print("  GLOBAL SIGNALS:")
        print(f"    Data Alignment: {metrics.get('data_alignment', 'pending')}")
        print(f"    Global VIX:     {metrics.get('global_vix', 'N/A')}")
        print(f"    US Gap:         {metrics.get('us_gap', 'N/A')}")
        print(f"    Forex Strength: {metrics.get('forex_strength', 'N/A')}")
        print()
        
        print("  SAFETY:")
        print(f"    Circuit Breaker: {'ENABLED' if data.get('circuit_breakers', {}).get('enabled') else 'DISABLED'}")
        print(f"    VIX Threshold:   {data.get('circuit_breakers', {}).get('vix_threshold', 25)}")
        print(f"    SL Max:          {data.get('circuit_breakers', {}).get('sl_percent', 0.30)*100:.1f}%")
        
        print()
        print("="*70)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        print(f"Updated: {timestamp} | Refresh: {self.refresh_interval}s | Press Ctrl+C to exit")
    
    def run(self):
        """Main monitoring loop"""
        print("🚀 Starting ATIS Agent Monitor...")
        time.sleep(1)
        
        try:
            while True:
                data = self.load_state()
                
                if not data:
                    print("[!] dev_state.json not found. Waiting for Orchestrator to initialize...")
                    time.sleep(self.refresh_interval)
                    continue
                
                # Display based on rich availability
                if RICH_AVAILABLE:
                    self.display_with_rich(data)
                else:
                    self.display_basic(data)
                
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\n\n✅ Monitor stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(self.refresh_interval)


if __name__ == "__main__":
    monitor = AgentMonitor()
    monitor.run()
