"""Live diagnostics and self-test for Gemma-Quest inference engine.

Run with: python3 -m src.engine
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from .factory import LLMFactory
from .local import LocalProvider
from .cloud import CloudProvider


def main():
    console = Console()
    console.print(
        Panel.fit(
            "[bold bright_green]Gemma-Quest: Inference Engine Diagnostics[/bold bright_green]\n"
            "[dim]Testing Local Ollama, Cloud API, and Hybrid Routing[/dim]",
            border_style="bright_green",
        )
    )

    cfg = load_config()

    # 1. Inspect Local Ollama
    local_prov = LocalProvider(
        model_name=cfg.llm.local_model,
        base_url=cfg.llm.ollama_base_url,
    )
    local_online = local_prov.is_available()

    # 2. Inspect Cloud API
    cloud_key = cfg.llm.get_api_key()
    cloud_prov = CloudProvider(
        api_key=cloud_key or "",
        endpoint=cfg.llm.cloud_endpoint,
        model_id=cfg.llm.cloud_model_id,
        provider_name=cfg.llm.cloud_provider,
    )
    cloud_configured = cloud_prov.is_available()

    # 3. Active Factory Provider
    active_prov = LLMFactory.get_provider(cfg)
    active_status = active_prov.get_status()

    # Diagnostics Table
    table = Table(
        title="Inference Backend Readiness",
        border_style="bright_green",
    )
    table.add_column("Component", style="bold yellow", width=22)
    table.add_column("Configuration", style="white", width=34)
    table.add_column("Readiness / Status", style="bold", width=24)

    table.add_row(
        "Local Ollama Server",
        f"Host: {cfg.llm.ollama_base_url}\nModel: {cfg.llm.local_model}",
        "[bright_green]ONLINE[/bright_green]" if local_online else "[dim yellow]OFFLINE (Daemon not running)[/dim yellow]",
    )

    cloud_status_text = (
        "[bright_green]READY[/bright_green]"
        if cloud_configured
        else "[dim yellow]UNCONFIGURED (Missing Key/URL)[/dim yellow]"
    )
    table.add_row(
        "Cloud API Client",
        f"Provider: {cfg.llm.cloud_provider}\nEndpoint: {cfg.llm.cloud_endpoint or 'None'}\nKey Env: {cfg.llm.api_key_env_var} ({'Set' if cloud_key else 'Missing'})",
        cloud_status_text,
    )

    table.add_row(
        "Active Runtime Provider",
        f"Mode: [cyan]{cfg.llm.mode}[/cyan]\nStatus: {active_status}",
        "[bright_green]CONFIGURED[/bright_green]",
    )

    console.print(table)

    # 4. Optional Live Test Execution
    console.print(f"\n[bold yellow]Active Provider Status:[/bold yellow] [bold bright_green]{active_status}[/bold bright_green]")

    if local_online or cloud_configured:
        console.print("[dim]Attempting test generation ping across active provider...[/dim]")
        test_messages = [
            {"role": "system", "content": "You are a dungeon master. Respond in 1 brief sentence."},
            {"role": "user", "content": "The adventurer enters the tavern."},
        ]

        start_time = time.time()
        try:
            response_text = active_prov.chat(test_messages, {"max_tokens": 50, "temperature": 0.7})
            duration = time.time() - start_time
            console.print(
                Panel(
                    f"[bold]Response ({duration:.2f}s):[/bold]\n{response_text.strip()}",
                    title="Live Test Completion",
                    border_style="bright_green",
                )
            )
        except Exception as e:
            console.print(f"[bold red]Live generation failed:[/bold red] {e}")
    else:
        console.print(
            "\n[dim yellow]Note: Local Ollama is currently offline and Cloud credentials are unset.[/dim yellow]\n"
            "[dim]Start Ollama via 'ollama serve' or configure GEMMA_API_KEY in .env for live generations.[/dim]"
        )


if __name__ == "__main__":
    main()
