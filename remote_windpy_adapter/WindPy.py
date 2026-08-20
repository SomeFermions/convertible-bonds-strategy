"""Ubuntu compatibility shim: from WindPy import w"""
try:
    from .RemoteWindPy import RemoteWind, WindData, w
except ImportError:  # Supports PYTHONPATH=remote_windpy_adapter.
    from RemoteWindPy import RemoteWind, WindData, w

__all__ = ["RemoteWind", "WindData", "w"]
