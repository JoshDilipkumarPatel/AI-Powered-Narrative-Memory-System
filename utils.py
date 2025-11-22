import json
import time
import re
from typing import Any, Optional

def safe_parse_json(text: str) -> Optional[Any]:
    """
    Try to extract a JSON object from a model response, trying a few repair heuristics.
    Returns a Python object or None.
    """
    txt = text.strip()

    try:
        return json.loads(txt)
    except Exception:
        pass

    m = re.search(r'(\{.*\})', txt, re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    repaired = txt.replace("'", '"')
    try:
        return json.loads(repaired)
    except Exception:
        pass

    return None

def retry_with_backoff(fn, retries=3, base_delay=1, *args, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
