import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.ERROR,  # Only show errors from libraries
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
