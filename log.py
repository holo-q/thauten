import logging

from rich.console import Console
from rich.logging import RichHandler

cl = Console()
logging.basicConfig(
    level=logging.ERROR,  # Only show errors from libraries
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=cl, rich_tracebacks=True, show_path=False)]
)
