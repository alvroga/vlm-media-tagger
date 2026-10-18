import os


def _main() -> None:
    if os.environ.get("VMT_PROBE_PATH"):
        # Decode-probe subprocess (media_probe.probe_video): skip the GUI.
        # A frozen briefcase app ignores argv and always boots this entry
        # point again, so the env var is how probe_video tells this
        # re-invocation to run the probe instead of a second window.
        from media_probe import _run_as_child

        _run_as_child()
    else:
        from vmt_app.app import main

        main()  # blocks until the window closes, then exits the process


if __name__ == "__main__":
    _main()
