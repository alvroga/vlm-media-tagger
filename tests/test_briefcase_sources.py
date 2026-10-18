"""Every first-party module the app imports is in briefcase's `sources`.
`briefcase dev` runs from src/ where everything is importable, so a missing
entry only shows up as a crash of the built app; this catches it here."""
import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _import_closure() -> set[str]:
    local = {p.stem for p in SRC.glob("*.py")} | {"translation_models"}
    seen: set[str] = set()
    todo = list((SRC / "vmt_app").glob("*.py"))
    while todo:
        tree = ast.parse(todo.pop().read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in local and name not in seen:
                    seen.add(name)
                    todo.append(SRC / "translation_models" / "__init__.py"
                                if name == "translation_models" else SRC / f"{name}.py")
    return seen


def _sources() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return config["tool"]["briefcase"]["app"]["vmt_app"]["sources"]


def test_every_imported_module_is_bundled():
    bundled = {Path(s).stem if s.endswith(".py") else Path(s).name for s in _sources()}
    assert _import_closure() - bundled == set()


def test_every_source_entry_exists():
    assert [s for s in _sources() if not (ROOT / s).exists()] == []


def test_vendored_dirs_are_bundled():
    # llama_manager.find_binary and profanity._VENDOR_DIR look for these
    # under Resources/app/ in the built app.
    assert {"vendor/llama_cpp", "vendor/ldnoobw"} <= set(_sources())
