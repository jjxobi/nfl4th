import torch  # noqa: F401

# Importing torch first guarantees its DLLs load before pandas's on Windows,
# where loading pandas first causes torch's DLL load to fail. See
# src/nfl4th/__init__.py for the same fix on the product side.
