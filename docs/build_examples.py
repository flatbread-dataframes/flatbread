from pathlib import Path
import importlib.util

from flatbread.output.html.display import PitaDisplayMixin


EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "assets" / "examples"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build():
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        if path.name.startswith("_"):
            continue

        module = load_module(path)
        result = getattr(module, "result", None)
        if result is None:
            print(f"  skipping {path.name}: no 'result' found")
            continue

        # mirror subfolder structure in output
        relative = path.relative_to(EXAMPLES_DIR)
        out_dir = OUTPUT_DIR / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out = out_dir / f"{path.stem}.json"
        if isinstance(result, PitaDisplayMixin):
            out.write_text(result.get_json())
        else:
            out.write_text(result.pita.get_json())
        print(f"  {relative} -> {out.name}")


if __name__ == "__main__":
    build()
