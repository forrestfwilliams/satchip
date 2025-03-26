"""satchip processing."""

import logging
from pathlib import Path


log = logging.getLogger(__name__)


def process_satchip(greeting: str = 'Hello world!') -> Path:
    """Create a greeting product.

    Args:
        greeting: Write this greeting to a product file (Default: "Hello world!" )
    """
    log.debug(f'Greeting: {greeting}')
    product_file = Path('greeting.txt')
    product_file.write_text(greeting)
    return product_file
