"""Optional generation contract. The evaluator never imports a model library."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedOutput:
    output: Path
    model_revision: str
    recipe: dict[str, Any]
    seed: int | None


class Editor(Protocol):
    def generate(self, source: Path, instruction: str, destination: Path, *, seed: int | None) -> GeneratedOutput: ...


class IdentityExample:
    """A no-change control for integration tests; never evidence of edit success."""
    def generate(self, source, instruction, destination, *, seed=None):
        import shutil
        destination = Path(destination)
        if destination.exists():
            raise ValueError("Refusing to overwrite an existing output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return GeneratedOutput(destination, "identity-example-v1", {"operation": "copy", "instruction": instruction}, seed)
