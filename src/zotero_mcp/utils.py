from typing import List, Dict
import os

def format_creators(creators: List[Dict[str, str]]) -> str:
    """
    Format creator names into a string.
    
    Args:
        creators: List of creator objects from Zotero.
        
    Returns:
        Formatted string with creator names.
    """
    names = []
    for creator in creators:
        if "firstName" in creator and "lastName" in creator:
            names.append(f"{creator['lastName']}, {creator['firstName']}")
        elif "name" in creator:
            names.append(creator["name"])
    return "; ".join(names) if names else "No authors listed"


def is_local_mode() -> bool:
    """Return True if running in local mode.

    Local mode is enabled when environment variable `ZOTERO_LOCAL` is set to a
    truthy value ("true", "yes", or "1", case-insensitive).
    """
    value = os.getenv("ZOTERO_LOCAL", "")
    return value.lower() in {"true", "yes", "1"}

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import threading
            result = [TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]

        return wrapper
    return decorator
