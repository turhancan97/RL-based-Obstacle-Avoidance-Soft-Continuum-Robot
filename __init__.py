try:
    from . import kinematics  # type: ignore
except ImportError:  # pragma: no cover - direct script/import fallback
    import kinematics  # type: ignore
