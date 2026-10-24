# Third-party notices

Unmix's own code is MIT licensed (see `LICENSE`). It uses and downloads third-party software and models that
keep their own licences:

| Component | Used for | Licence |
|---|---|---|
| `vendor/apollo` (Apollo) | Optional "Heal" repair of a stem | CC BY-SA 4.0 (`vendor/apollo/LICENSE`) |
| `vendor/rubberband/unmix-stretch` (built from Rubber Band 4.0.0 plus a small wrapper, `unmix_stretch.cpp`) | Best-quality key and tempo change. Run as a separate program (like ffmpeg), not linked into Unmix | GPL-2.0-or-later (`vendor/rubberband/COPYING.rubberband`). Source: https://breakfastquay.com/rubberband/ and https://breakfastquay.com/files/releases/rubberband-4.0.0.tar.bz2; rebuild with `vendor/rubberband/build.sh` |
| python-audio-separator | Runs the separation models | MIT |
| Demucs v4 (`htdemucs_ft`, `htdemucs_6s`) | Instrument separation | MIT |
| BS-Roformer / Mel-Band Roformer vocal checkpoints, Mel-Roformer denoise and de-reverb checkpoints | Vocal split and voice clean up | Downloaded on first use from their authors' releases; each keeps its author's licence |
| FastAPI, uvicorn, pywebview, yt-dlp, soundfile, NumPy, SciPy, librosa, PyTorch | The app itself | MIT, BSD or Apache-2.0 |
| Space Grotesk (embedded in `index.html`) | Interface font | SIL Open Font License 1.1 |

Model files are not part of this repository or of the installer. They are downloaded to
`~/Library/Application Support/Unmix` when you first use a feature that needs them. Check each model's licence
before using stems commercially.
