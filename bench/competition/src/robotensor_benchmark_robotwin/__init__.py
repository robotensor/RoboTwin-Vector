"""The RoboTwin-Vector benchmark as a plugin of the Robotensor Robotensor Vector orchestrator.

The orchestrator finds it through the `robotensor.benchmarks` entry point `robotwin`, which loads
`BENCHMARK`. It never imports `vector_orchestrator`: the orchestrator checks it structurally.
"""

from .plugin import BENCHMARK, VERSION, RoboTwinBenchmark

__version__ = VERSION

__all__ = ["BENCHMARK", "RoboTwinBenchmark", "__version__"]
