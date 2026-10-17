"""CLI entry point for the translation-models package. It is a package (so
manifest.json travels with it), so the invocation is `-m`:

    PYTHONPATH=src python -m translation_models status es
    PYTHONPATH=src python -m translation_models download es
"""
from translation_models import _main

_main()
