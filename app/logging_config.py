import logging, sys, uuid, json, hashlib
import structlog

def setup_logging(level=logging.INFO):
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            timestamper,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

logger = structlog.get_logger()

def new_request_id() -> str:
    return str(uuid.uuid4())

def safe_hash_bytes(b: bytes) -> str | None:
    return hashlib.sha256(b).hexdigest()[:16] if b else None

def safe_hash_obj(obj) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
