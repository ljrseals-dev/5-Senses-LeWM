"""Sensory modules (Eyes, Ears, ...). Currently only world_model (Eyes). Imports are lazy so the
numpy-only parts (toy_env, physics) work without torch."""

__all__ = ["WorldModelEye", "LeWMStyle"]


def __getattr__(name):
    if name in __all__:
        from . import world_model
        return getattr(world_model, name)
    raise AttributeError(f"module 'senses' has no attribute {name!r}")
