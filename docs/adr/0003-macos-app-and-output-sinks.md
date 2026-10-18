# ADR-0003: A macOS app and pluggable output sinks (JSON sidecars by default)

Status: Accepted — implemented (`src/vmt_app/`, `src/sinks.py`, `src/library.py`, [`docs/plugins.md`](../plugins.md))
Date: 2026-10-17
Supersedes / related: ADR-0001, ADR-0002

## Context
The CLI suits batch jobs and experiments, but trying the pipeline on a real library (and seeing where it fails)
should not need a terminal. Results also need to land somewhere useful: a file next to the media, a database,
a web service.

## Decision
1. A free, open-source macOS app (`src/vmt_app/`, pywebview UI, packaged with briefcase) over the same
   pipeline: a sources list (folders, scanned recursively, and single files), batch runs with progress, live
   log and Stop, scheduled re-runs, a skipped-files view, settings, and first-launch model setup through
   `llama_manager`.
2. Results go through an output-sink interface (`src/sinks.py`): `is_done(media, tasks)`,
   `write(media, record)`, `close()`.
   - `SidecarSink` (default) writes `<file>.<ext>.json` beside the original (full name + `.json`, so
     `clip.mp4` and `clip.mov` never collide), atomically, merging tasks and recording source size/mtime,
     model, prompt set, tool version and timestamp. A read-only folder falls back to a mirror under the app's
     support directory.
   - `JsonlSink` (optional) appends every record to one file.
   - Third-party sinks load from `OUTPUT_SINKS="sidecar,mypkg.module:MySink"`; `docs/plugins.md` shows a
     ~30-line example.
3. Resume: a file is skipped when its sidecar already holds the requested tasks for the file's current size
   and mtime; a changed file is redone.
4. The CLI uses the same sinks (`--sidecar`, `--sinks`, `--skip-done`).
5. Sources can live on SMB/AFP/NFS shares. macOS mounts them (Finder's Connect to Server, or `open smb://…`
   from the app), with credentials in the Keychain, so the app never handles a share password. The app
   remembers which share a source lives on (from `mount`), reconnects it at launch, and optionally unmounts it
   on quit (`src/vmt_app/volumes.py`). To the scanner a mounted share is just another folder. (A
   `smbutil`/`mount_smbfs` design that took the password itself was rejected: the password would ride in a
   process argument visible to `ps`.)
6. For the first release the sidecars are the only processed-state record (no database). `is_done` stays behind
   the sink interface so a local index can replace or cache it later without touching the batch code.

## Consequences
+ Anyone can run it on their own library for free and send the metadata anywhere by writing one class.
+ One pipeline and one record format behind both the CLI and the app.
− Packaging work: a vendored `llama-server` in the bundle, code signing, and RAM guidance (4.7 needs a
  32 GB+ Mac). The release is signed but not notarized, so a downloaded build needs one "Open Anyway" in
  System Settings → Privacy & Security.
