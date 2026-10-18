"""The JS ↔ Python bridge of the app stays in lockstep: every api() call in
app.js is an Api method, every event type the backend pushes is handled by
UI.dispatch, and every element id app.js looks up exists in index.html."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "vmt_app"
WEBUI = ROOT / "resources" / "webui"
JS = (WEBUI / "app.js").read_text()
HTML = (WEBUI / "index.html").read_text()
APP_PY = (ROOT / "app.py").read_text()


def _api_methods() -> set[str]:
    tree = ast.parse(APP_PY)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Api")
    return {f.name for f in cls.body if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")}


def _pushed_types() -> set[str]:
    return set(re.findall(r'push\(type="([a-z_]+)"', APP_PY)) | set(
        re.findall(r'self\.push\(\s*type="([a-z_]+)"', APP_PY))


def test_every_js_api_call_exists_in_python():
    called = set(re.findall(r'api\("([a-z_]+)"', JS))
    assert called, "no api() calls found — regex out of date?"
    assert called - _api_methods() == set()


def test_every_api_method_is_used_by_the_front_end():
    called = set(re.findall(r'api\("([a-z_]+)"', JS))
    assert _api_methods() - called == set()


def test_every_pushed_event_is_dispatched():
    handled = set(re.findall(r'case "([a-z_]+)":', JS))
    pushed = _pushed_types()
    assert pushed, "no push(type=...) found — regex out of date?"
    assert pushed - handled == set()


def test_every_element_id_used_in_js_exists():
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', JS))
    defined = set(re.findall(r'id="([A-Za-z0-9_-]+)"', HTML))
    assert used - defined == set()


def test_no_inner_html_in_front_end():
    # External text (paths, share names, errors) must never be parsed as HTML.
    assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write", JS)
