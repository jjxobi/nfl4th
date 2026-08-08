import torch  # noqa: F401

# torch must be imported before pandas anywhere in this process. On Windows,
# once pandas 1.5.3 has loaded its DLLs, torch's own DLL load fails with
# OSError: [WinError 1114]. Importing torch here first, before any submodule
# gets a chance to import pandas, guarantees the safe load order on every
# path into this package.
