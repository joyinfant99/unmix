# Unmix

A local web app that separates a song into stems (vocals, drums, bass, guitar, piano, other), can clean up noise and reverb from speech, and lets you mix the results like a small DAW. Everything runs on this Mac (Apple M3 Pro, 18 GB). Nothing is uploaded anywhere.

Styled to match bandosapp.com (same palette, Space Grotesk font, radii and thin progress bars).

Formerly called StemSep, then Prism. Now **Unmix** (shown as Band OS Unmix on bandosapp.com). The project folder is `Unmix`.

## Download, license and updates

- **Download:** https://bandosapp.com/unmix (Mac only, Apple Silicon). The installer is free; the page asks for an email address and a 6 digit code sent to it before the download starts.
- **Open source:** the code is MIT licensed (`LICENSE`); third-party components keep their own licences (`NOTICE.md`). Issues and pull requests are welcome. To build it yourself, see "Developing and testing" below.
- **Install stats:** installed copies send an anonymous install id, their version, macOS version, chip, Mac model and memory with the update check (nothing else), and on first launch the app offers to confirm the user's email with a 6 digit code (skippable). They show on the Unmix page of the Band OS admin portal, which also lists release versions and dates.
- **Updates:** the installed app asks the Band OS server for a newer version when it starts and offers "Update and restart". Publishing a version: `./release.sh <version> "notes"` builds the installer, uploads it to the download storage and tags the source (needs the BandOS backend checkout; see `release.sh`).

## Using it

- **Open:** click **Unmix** in the Dock (installed in `/Applications` by the .pkg). It opens Unmix in a native window (pywebview) with the server running inside the same process. Clicking it again just brings the existing window to the front. The top-right of the page shows a green "Server connected" dot (red if the server stops responding).
- **Quit:** right-click the Dock icon, Quit (or Cmd+Q). This stops the server.
- **Command line alternative:** `./run.sh` runs `desktop.py` (same native window).
- **Development:** see "Developing and testing" below. `./dev.sh` runs from source with auto-restart and needs no install.
- **Projects are files:** every song is one `Song Name.umx` file (a macOS package: Finder shows one file, inside are the source audio, stems and settings). Double-click a `.umx` to open it in Unmix, use **Open a project** on the start screen, or pick it in the sidebar list of recent unmixes (newest first; search, rename, Trash, Show in Finder). New projects are saved in the folder shown under "Where new projects are saved" (default `~/Library/Application Support/Unmix/data`, changeable). A project opened from anywhere else is edited in place. **Save a copy…** (toolbar) writes a duplicate wherever you choose. Old-style libraries (separate source and result folders) are converted automatically on first launch.
- **Start:** after an upload or link, a pop-up asks what to do: Song instrument separator (all stems), Vocal and instrument separator (2 tracks) or Voice clean up (noise and reverb, whole recording). Models are chosen automatically; audio over 5 minutes uses the lighter single vocal model. It then runs to the end and opens the mixer.
- **Mixer:** one track per stem with mute, solo, volume (dB readout, click it to type a value) and a live level meter. Toolbar: play/stop, shrink/expand tracks (tracks open shrunk by default), Save a copy, Split again, Bounce. Tooltips always stay inside the window. The sidebar collapses to an icon rail.
- **Logs:** `~/Library/Logs/Unmix.log`.
- On first launch macOS may ask to let Unmix access the Desktop folder. Click Allow.

### The flow
1. **Upload:** a file, or a YouTube/link (best audio YouTube offers, about Opus 160 kbps, never lossless). This creates the `.umx` project.
2. **Choose:** the pop-up (see Start above) with a time estimate for each choice. No other options; Heal is per stem in the mixer.
3. **Process:** live per-step progress.
4. **Mix:** all stems are rendered first and time-aligned, then loaded together. Per-stem Heal with a Split/Healed A/B switch.
5. **Bounce:** the only place a file format (WAV, FLAC, MP3 320) is chosen. Either the mix as heard (mutes, solos, volumes applied) or each stem as its own file, saved wherever you pick. The project is never modified by a bounce.
**Split again** (toolbar) re-runs the split on the project's own audio with the previous settings; there is no re-upload.

## How it works

`app.py` (FastAPI) + `index.html` (single page, Web Audio for the mixer). One background worker runs jobs one at a time.

| Step | Model | Notes |
|---|---|---|
| Vocal split (default) | BS-Roformer ep_368 + Mel-Band Roformer (becruily), averaged | overlap 2. Instrumental = mix minus vocals |
| Vocal split (audio over 5 minutes) | BS-Roformer ep_368 alone | chosen automatically; about 1.6x quicker, about 0.4 dB less clean |
| 4 stems | Demucs v4 `htdemucs_ft` | run on the instrumental |
| 6 stems (adds guitar, piano) | Demucs v4 `htdemucs_6s` | run on the instrumental |
| Denoise | Mel-Roformer denoise (aufr33) | |
| De-reverb | Mel-Roformer de-reverb (anvuew) | |
| Heal | Apollo (vendored in `vendor/apollo`) | repairs artifacts, subtle effect |

Key design decisions:
- **Cleanup runs on the whole recording only for the Voice clean up goal (speech).** The noise/de-reverb models are built for isolated vocals and strip instruments from a full mix (measured: de-reverb dropped a full mix by about 13 dB), so song goals never run them.
- **Models are picked by weight, not by the user.** Audio over 5 minutes uses the single vocal model; shorter audio uses the two-model ensemble. There are no option checkboxes.
- **Stems are conformed** to stereo, 44.1 kHz and the exact source length so they stay in sync in the mixer.
- Each stem can carry variants (split, cleaned, healed, original). Heal is per stem, from the mixer.
- Library: `python-audio-separator` (runs on the Mac GPU via MPS/CoreML).

### Time estimates
`timings.json` holds measured speed per model (seconds of compute per second of audio) and is refined automatically after each real run. Measured here: an 86 s song, all 6 stems, default mode: estimate 195 s, actual 170 s. A 4-minute song takes about 8 minutes (about 5 with the lighter model).

## Quality findings (see `eval/`)

Scored against ground-truth stems from MUSDB18 (6 test tracks, 25 s clips, clip-mean SDR in dB):

| Setup | Vocals | Instrumental |
|---|---|---|
| Single BS-Roformer (before) | 15.95 | 20.55 |
| + instrumental as mix minus vocals | 15.95 | 20.85 |
| Ensemble BS-Roformer + Mel-Band | 16.36 | 21.23 |

- Ensemble gain is real but small (+0.4 dB on 6 tracks, only about +0.1 dB on a separate 3-track check). Do not expect a dramatic change.
- The newest community vocal models (unwa) were 8-10x slower. The best (`revive_v3e`, 16.31 dB) added only 0.14 dB when ensembled, so it was left out. Others scored worse. 3 of 9 candidate models were not scored (benchmark stopped).
- Never compared against LALAL.AI or Moises. Do a blind A/B on 3-4 songs to find out.
- **Weakest area: guitar and piano** (from `htdemucs_6s`). No better model found in the library yet.

## Cloud cost study (`cloud/`)

Benchmarked on Colab's free T4 (180 s song). The T4 was only about 1.3-1.6x faster than this Mac for the heavy models. Estimated cost for 50 songs/month on pay-per-second serverless GPUs: about $2.50-10 compute (about $0.60/hr T4 assumed from memory, verify), plus $0-20 hosting. Results in `cloud/results_colab_t4.md`. Not built: a cloud version. It would need direct-to-storage uploads, background job queue, per-user storage, and the YouTube feature dropped (YouTube blocks cloud IPs). Also check licenses before charging anyone: Apollo is CC BY-SA 4.0, MUSDB18 is research-only, other checkpoints unchecked.

## Files

```
app.py            backend: planning, pipelines, API
index.html        the whole UI (upload, goal pop-up, progress, mixer); Band OS palette and Space Grotesk font embedded
run.sh            start the window from a terminal
dev.sh            development mode: run from source, auto-restart on .py changes, port 8799
timings.json      measured model speeds (auto-updated)
benchmark.py      re-measure local model speeds
vendor/apollo/    Apollo heal model code (CC BY-SA 4.0)
models/           downloaded model files (about 6 GB)
data/             default library: one .umx package per song (deleting one deletes that song)
assets/           icon artwork (make_icon.py), Unmix.icns
eval/             model comparison: bench_models.py, ensemble.py, clips/ (MUSDB, research use only), out/
cloud/            Colab benchmark notebook and results
.venv/            Python 3.11 environment
```

### Developing and testing
- `./dev.sh` runs the app from source in a native window. Saving a `.py` file restarts it; press Cmd+R in the window for `index.html` changes. It uses port 8799 and `./data`, so it never touches an installed Unmix (port 8765, `~/Library/Application Support/Unmix`). `./dev.sh plain` runs once without the watcher.
- `APP_ONLY=1 ./build_app.sh` builds `dist/Unmix.app` without the installer pkg, for checking packaging-only things (bundled files, the `.umx` file association, the updater). Delete the build afterwards so Spotlight does not list it as an installed app.
- `./build_app.sh` (no flag) also builds the `.pkg`; `./release.sh <version> "notes"` builds, tags and publishes it.

### How the app is packaged
`desktop.py` is the app (pywebview window + uvicorn in one process, single-instance lock in `data/.unmix.lock`). `build_app.sh` bundles it with PyInstaller into `Unmix.app` and wraps it in an installer `.pkg` that installs to `/Applications`. Models are not bundled; they download on first use. Redraw the icon with `.venv/bin/python assets/make_icon.py`.

## Known issues and open items

- Not tested: the clean-up-only goal, noisy or echoey recordings (cleanup thresholds were tuned on two songs), and quitting from the Dock or re-clicking the icon.
- One benchmark run took 979 s instead of about 46 s (probably the Mac stalled; not confirmed).
- A background login item cannot run this project: macOS blocks launch agents from reading `~/Desktop`. (A silent login item was built, then removed at the owner's request.)
- Python's own SSL certificates are not set up on this Mac (urllib fails). Use `curl`/requests.
- Disk is tight (about 38 GB free at last check). `data/` and `eval/` (about 1.3 GB) can be deleted.
- An unused notebook `Untitled0.ipynb` was created in the owner's Google Drive during the Colab test.
- Ideas not done: better guitar/piano model, blind A/B listening page, drum-kit splitter (kick/snare/hats), "residual" track, a trained healer for separation artifacts, a cloud version.

## How Unmix gets from the website to your Mac

Unmix is not a website. It is a normal Mac app that you download from bandosapp.com as a `.pkg`; the page only hosts the download and the instructions. What happens:

1. **Website:** `bandosapp.com/free-tools/unmix` links to `Unmix-<version>.pkg` (a GitHub release; the link is `NEXT_PUBLIC_UNMIX_DOWNLOAD_URL`).
2. **Download and open the .pkg.** It is not notarized (no paid Apple account), so macOS blocks a double-click: right-click the file, choose **Open**, then **Open** again. (Or System Settings > Privacy & Security > Open Anyway.)
3. **System check, then install.** The installer first checks the Mac (Apple Silicon, macOS version, free disk space, memory) and only then copies `Unmix.app` into `/Applications`.
4. **First launch.** Open Unmix from Applications. The app starts a small server inside itself (`127.0.0.1:8765`, only reachable from this Mac) and shows the page in its own window, so it looks like a browser tab but nothing is served from the internet and nothing is uploaded. The first split downloads the models (about 6 GB) into `~/Library/Application Support/Unmix`.
5. **Updates.** Unmix checks the GitHub releases and offers a new `.pkg`; installing it replaces the app and keeps your models, settings and projects.

If the app does not open after installing, see Troubleshooting.

### Troubleshooting

- **Nothing happens when opening the .pkg:** use right-click > Open (see step 2).
- **App bounces in the Dock and quits, or no window:** read `~/Library/Logs/Unmix.log`. Version 1.0.0 had a start-up crash after a reinstall (a leftover link to `ffmpeg` pointing at the deleted app); it is fixed, the link is now rebuilt on every launch. If you hit it on an old build, delete `~/Library/Application Support/Unmix/bin` and open the app again.
- **"Unmix is already running" or port 8765 busy:** quit Unmix from the Dock (or `pkill -x Unmix`) and reopen.
- **Fully fresh start:** drag `Unmix.app` to the Trash, delete `~/Library/Application Support/Unmix` (models and settings; your `.umx` projects live where you saved them and are not touched), and install again.

## Building a downloadable app (.dmg)

`./build_app.sh [version]` produces `dist/Unmix.app` and `dist/Unmix-<version>.pkg` (PyInstaller, ad-hoc signed). Models are not bundled; they download on first use into `~/Library/Application Support/Unmix` (which also holds the library). Without a paid Apple Developer account the app is not notarized, so first launch needs right-click > Open (or `xattr -dr com.apple.quarantine /Applications/Unmix.app`). Apple Silicon only (built on arm64).
