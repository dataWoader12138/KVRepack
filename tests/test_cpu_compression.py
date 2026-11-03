"""Real CPU tensor invariants and explicit reproductions of known source findings.

Use the pinned hybridkv Python to execute; a Python without torch skips this suite.
The shared implementation also has a CLI that emits a detailed JSON report:
  python scripts/check_cpu_compression.py --source-root ..
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / 'scripts/check_cpu_compression.py'
_spec = spec_from_file_location('kvrepack_cpu_compression_checks', _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
CPUCompressionChecks = _module.CPUCompressionChecks
