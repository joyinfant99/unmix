# Third-party notices

Unmix's own code is MIT licensed (see `LICENSE`). It uses and downloads third-party software and models that
keep their own licences:

| Component | Used for | Licence |
|---|---|---|
| `vendor/apollo` (Apollo) | Optional "Heal" repair of a stem | CC BY-SA 4.0 (`vendor/apollo/LICENSE`) |
| python-audio-separator | Runs the separation models | MIT |
| Demucs v4 (`htdemucs_ft`, `htdemucs_6s`) | Instrument separation | MIT |
| BS-Roformer / Mel-Band Roformer vocal checkpoints, Mel-Roformer denoise and de-reverb checkpoints | Vocal split and voice clean up | Downloaded on first use from their authors' releases; each keeps its author's licence |
| FastAPI, uvicorn, pywebview, yt-dlp, soundfile, NumPy, PyTorch | The app itself | MIT, BSD or Apache-2.0 |
| Space Grotesk (embedded in `index.html`) | Interface font | SIL Open Font License 1.1 |

Model files are not part of this repository or of the installer. They are downloaded to
`~/Library/Application Support/Unmix` when you first use a feature that needs them. Check each model's licence
before using stems commercially.
