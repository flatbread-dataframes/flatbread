from pathlib import Path
import importlib.util
import sys


EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "assets" / "examples"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        module = load_module(path)
        examples = getattr(module, "examples", None)
        if examples is None:
            print(f"  skipping {path.name}: no 'examples' dict found")
            continue

        for name, df in examples.items():
            out = OUTPUT_DIR / f"{name}.json"
            out.write_text(df.pita.get_json())
            print(f"  {path.name} -> {out.name}")


if __name__ == "__main__":
    build()
