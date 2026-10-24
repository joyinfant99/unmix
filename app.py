"""Unmix: local web app. Upload -> plan -> process -> mixer -> bounce. Each song is one .umx project package."""
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import yt_dlp
from audio_separator.separator import Separator
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import analysis
import openq

ROOT = Path(__file__).parent  # read-only resources (index.html, vendor/); inside the bundle when frozen
# Writable state: next to the code in dev, in Application Support when running as the packaged app
STATE = Path.home() / "Library" / "Application Support" / "Unmix" if getattr(sys, "frozen", False) else ROOT
MODELS = STATE / "models"
DEFAULT_DATA = STATE / "data"
CONFIG_FILE = STATE / "config.json"
STATE.mkdir(parents=True, exist_ok=True)


def _cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_cfg(**updates) -> None:
    cfg = _cfg()
    for k, v in updates.items():
        if v in (None, [], ""):
            cfg.pop(k, None)
        else:
            cfg[k] = v
    CONFIG_FILE.write_text(json.dumps(cfg))


def _configured_library() -> Path | None:
    raw = _cfg().get("library_dir")
    return Path(raw) if raw else None


# Where projects (stems and source audio) live. The user can point this anywhere, e.g. an external drive.
_custom = _configured_library()
LIBRARY_MISSING: str | None = None  # set when the chosen folder is unavailable (drive unplugged) and we fell back
def _usable(folder: Path) -> bool:
    """Exists, can be written and can be listed (macOS may deny a folder such as ~/Desktop until the user allows it)."""
    try:
        if not (folder.is_dir() and os.access(folder, os.W_OK)):
            return False
        next(folder.iterdir(), None)
        return True
    except OSError:
        return False


if _custom and _usable(_custom):
    DATA = _custom
else:
    DATA = DEFAULT_DATA
    if _custom:
        LIBRARY_MISSING = str(_custom)
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# audio-separator shells out to a bare `ffmpeg`; expose the bundled binary under that name
_BIN = STATE / "bin"
_BIN.mkdir(exist_ok=True)
_link = _BIN / "ffmpeg"
# re-point every launch: after a reinstall, update or move the old link dangles (exists() is False)
if not (_link.is_symlink() and os.path.realpath(_link) == os.path.realpath(FFMPEG)):
    _link.unlink(missing_ok=True)
    _link.symlink_to(FFMPEG)
os.environ["PATH"] = f"{_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
SR = 44100

# --- model catalogue -------------------------------------------------------
VOCAL_A = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"      # BS-Roformer
VOCAL_B = "mel_band_roformer_vocals_becruily.ckpt"          # Mel-Band Roformer (different architecture)
M = {
    "vocal_split": (VOCAL_A, "BS-Roformer + Mel-Band Roformer ensemble", "Two different models averaged, which measured about +0.4 dB cleaner vocals than either alone"),
    "vocal_split_fast": (VOCAL_A, "BS-Roformer (single model)", "One model only: about 1.6x quicker, slightly less clean"),
    "stems4": ("htdemucs_ft.yaml", "Demucs v4 fine-tuned", "Highest-scoring drums / bass / other splitter"),
    "stems6": ("htdemucs_6s.yaml", "Demucs v4 6-stem", "Only model that also isolates guitar and piano"),
    "denoise": ("denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "Mel-Roformer denoise", "Removes hiss, hum and background noise"),
    "dereverb": ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "Mel-Roformer de-reverb", "Dries up roomy / echoey vocals"),
    "heal": ("apollo", "Apollo", "Repairs holes and artifacts left by separation"),
}
LABELS = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "guitar": "Guitar", "piano": "Piano",
          "other": "Other", "instrumental": "Instrumental", "cleaned": "Cleaned audio"}

# Seconds of compute per second of audio, plus fixed load overhead. Calibrated by
# benchmark.py and refined after every real run.
TIMINGS_FILE = STATE / "timings.json"
DEFAULT_TIMINGS = {
    "vocal_split": {"rtf": 2.05, "base": 4}, "vocal_split_fast": {"rtf": 1.25, "base": 2}, "stems4": {"rtf": 1.2, "base": 10}, "stems6": {"rtf": 0.7, "base": 8},
    "denoise": {"rtf": 0.7, "base": 6}, "dereverb": {"rtf": 0.7, "base": 6}, "heal": {"rtf": 0.6, "base": 6},
}

app = FastAPI(title="Unmix")
jobs: dict[str, dict] = {}
sources: dict[str, dict] = {}  # one entry per project; the project id is also its job id
PATHS: dict[str, Path] = {}  # project id -> its .umx package on disk (library folder or anywhere the user opened it from)
EXT = ".umx"
tasks: "queue.Queue[tuple]" = queue.Queue()


def load_timings() -> dict:
    t = json.loads(json.dumps(DEFAULT_TIMINGS))
    if TIMINGS_FILE.exists():
        for k, v in json.loads(TIMINGS_FILE.read_text()).items():
            t.setdefault(k, {}).update(v)
    return t


def record_timing(key: str, duration: float, actual: float) -> None:
    """Fold a measured run into the estimate (keeps the fixed overhead, adjusts the rate)."""
    if duration < 20:
        return
    t = load_timings()
    rtf_new = max((actual - t[key]["base"]) / duration, 0.05)
    t[key]["rtf"] = round(0.6 * t[key]["rtf"] + 0.4 * rtf_new, 3)
    TIMINGS_FILE.write_text(json.dumps(t, indent=1))


def est(key: str, duration: float) -> float:
    t = load_timings()[key]
    return round(t["base"] + t["rtf"] * duration)


def model_ready(key: str) -> bool:
    fn = M[key][0]
    if key == "heal":
        return (MODELS / "apollo" / "pytorch_model.bin").exists()
    return (MODELS / fn).exists() or any(MODELS.glob(fn.split(".")[0] + "*"))


# --- audio helpers ---------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def stem_of(path: str) -> str:
    m = re.findall(r"\(([^)]+)\)", Path(path).name)
    return m[-1] if m else Path(path).stem


# --- keeping the Mac responsive ---------------------------------------------
# Separation saturates the GPU, every core and a lot of memory. By default the worker runs at background
# priority (Apple's "utility" class: favours efficiency cores), leaves some cores free, uses smaller GPU
# batches, and frees GPU memory between models.
GPU_BATCH = 2
_QOS_UTILITY = 0x11


def set_priority() -> None:
    """Applies to the calling (worker) thread, so the window and the rest of the Mac stay responsive."""
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(_QOS_UTILITY, 0)
        import torch  # noqa: PLC0415

        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
    except Exception:  # noqa: BLE001  (a tuning failure must never stop a job)
        pass


def free_memory() -> None:
    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_model(model: str, src: Path, out_dir: Path, overlap: int | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"mdxc_params": {"segment_size": 256, "override_model_segment_size": False, "batch_size": GPU_BATCH,
                             "overlap": overlap, "pitch_shift": 0}} if overlap else {}
    sep = Separator(output_dir=str(out_dir), model_file_dir=str(MODELS), output_format="WAV", log_level=30, **extra)
    try:
        sep.load_model(model_filename=model)
        return {norm(stem_of(f)): out_dir / Path(f).name for f in sep.separate(str(src))}
    finally:
        del sep
        free_memory()


def pick(stems: dict[str, Path], *wanted: str) -> Path:
    for w in wanted:
        for k, p in stems.items():
            if w in k and not (w == "reverb" and k.startswith("no")):
                return p
    return next(iter(stems.values()))


def conform(src: Path, dst: Path, frames: int, floaty: bool = False) -> None:
    """Write `src` as stereo 44.1 kHz of exactly `frames` samples so every stem lines up."""
    x, sr = sf.read(src, always_2d=True)
    if sr != SR:
        raise RuntimeError(f"unexpected sample rate {sr}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    x = x[:frames] if len(x) >= frames else np.pad(x, ((0, frames - len(x)), (0, 0)))
    sf.write(dst, x if floaty else np.clip(x, -1, 1), SR, subtype="FLOAT" if floaty else "PCM_16")


def split_vocals(src: Path, work: Path, fast: bool, frames: int) -> tuple[Path, Path]:
    """Vocals via one model (fast) or an average of two different architectures (best).
    The instrumental is mix - vocals, which scored higher than the models' own instrumental output."""
    outs = []
    for i, model in enumerate([VOCAL_A] if fast else [VOCAL_A, VOCAL_B]):
        parts = run_model(model, src, work / f"v{i}", 2)
        outs.append(pick(parts, "vocal") if any("vocal" in k for k in parts) else next(iter(parts.values())))
    mix, _ = sf.read(src, always_2d=True)
    voc = np.zeros((frames, 2), dtype=np.float32)
    for p in outs:
        x, _ = sf.read(p, always_2d=True)
        n = min(frames, len(x))
        voc[:n] += x[:n, :2] / len(outs)
    inst = np.zeros((frames, 2), dtype=np.float32)
    n = min(frames, len(mix))
    inst[:n] = mix[:n, :2] - voc[:n]
    sf.write(work / "vocals_mix.wav", voc, SR, subtype="FLOAT")
    sf.write(work / "instrumental_mix.wav", inst, SR, subtype="FLOAT")
    return work / "vocals_mix.wav", work / "instrumental_mix.wav"


def download_audio(url: str, out: Path) -> tuple[Path, str]:
    opts = {
        "format": "bestaudio/best", "outtmpl": str(out / "dl.%(ext)s"), "noplaylist": True,
        "quiet": True, "no_warnings": True, "ffmpeg_location": FFMPEG,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return out / "dl.wav", info.get("title", "audio")


def to_source(raw: Path, dst: Path) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-vn", "-ar", str(SR), "-ac", "2",
                    "-c:a", "pcm_s16le", str(dst)], check=True)


def heal_file(src: Path, dst: Path) -> None:
    sys.path.insert(0, str(ROOT / "vendor" / "apollo"))
    import inference  # noqa: PLC0415  (vendored Apollo)

    ck = str(MODELS / "apollo" / "pytorch_model.bin")
    if not Path(ck).exists():
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        hf_hub_download("JusperLee/Apollo", "pytorch_model.bin", local_dir=str(MODELS / "apollo"))
    kw = dict(chunk_seconds=6, overlap_seconds=1, chunk_batch_size=2, chunk_pad_seconds=1)
    try:
        inference.run_inference(str(src), str(dst), ck, "mps", **kw)
    except Exception:  # noqa: BLE001  (fall back to CPU if the GPU path fails)
        inference.run_inference(str(src), str(dst), ck, "cpu", **kw)


# --- transpose and tempo -----------------------------------------------------
# The mixer asks for every stem at the project's key shift and tempo. Each stem is rendered once by ffmpeg
# (asetrate + atempo, no extra libraries) into a cache folder and reused; the .umx file only stores the two numbers.
SHIFT_CACHE = STATE / "cache" / "shift"
SHIFT_CACHE.mkdir(parents=True, exist_ok=True)
SEMI_MAX = 12                # one octave either way
TEMPO_MIN, TEMPO_MAX = 0.5, 2.0  # speed relative to the original (ffmpeg's atempo range)
UNPITCHED = {"drums"}        # drums have no pitch: they follow the tempo but are never transposed
_shift_locks: dict[str, threading.Lock] = {}
_shift_guard = threading.Lock()


def _atempo_chain(semi: int, tempo: float) -> str:
    """Pitch up by resampling (which also speeds up), then correct the speed to the wanted tempo."""
    ratio = 2 ** (semi / 12)
    parts = [f"asetrate={SR * ratio:.4f}", f"aresample={SR}"] if semi else []
    f = tempo / ratio
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2
    while f < 0.5:
        parts.append("atempo=0.5")
        f *= 2
    if abs(f - 1) > 1e-6:
        parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def _env_1khz(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).max(axis=1)
    n = len(m) // 44
    return m[: n * 44].reshape(n, 44).max(axis=1)


def _early_samples(orig: np.ndarray, out: np.ndarray, tempo: float, search_ms: int = 80) -> int:
    """ffmpeg's tempo change starts its output 10-40 ms early, by a different amount for each chain. Compare the
    loudness outline with the original (time-scaled) and return how many samples `out` is early, so every stem
    can be put back on the same grid. 0 when the stem has no outline to compare."""
    from scipy.signal import fftconvolve  # noqa: PLC0415

    eo, ey = _env_1khz(orig), _env_1khz(out)
    if len(eo) < 500 or len(ey) < 500 or ey.max() < 1e-4:
        return 0
    ref = np.interp(np.arange(len(ey)) * tempo, np.arange(len(eo)), eo, right=0)
    smooth = lambda e: np.sqrt(np.convolve(e, np.ones(8) / 8, "same"))  # noqa: E731
    a, b = smooth(ref), smooth(ey)
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0
    c = fftconvolve(a, b[::-1], mode="full") / den
    lags = np.arange(-len(b) + 1, len(a))
    c = np.where(np.abs(lags) <= search_ms, c, -np.inf)
    i = int(np.argmax(c))
    return int(lags[i] * SR / 1000) if c[i] >= 0.3 else 0  # a stem with no clear outline is left alone


# Best quality: Rubber Band (the "finer" engine), run as a separate program like ffmpeg. About 6 s per minute of audio.
# Fast preview: ffmpeg, a few seconds per song. If the Rubber Band helper is missing or fails, ffmpeg is used instead.
RUBBERBAND = ROOT / "vendor" / "rubberband" / "unmix-stretch"


def _render_rubberband(path: Path, semi: int, tempo: float, formant: bool) -> np.ndarray:
    x, _ = sf.read(path, always_2d=True, dtype="float32")
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "in.f32", Path(td) / "out.f32"
        a.write_bytes(np.ascontiguousarray(x[:, :2], dtype="<f4").tobytes())
        subprocess.run([str(RUBBERBAND), str(a), str(b), "2", str(SR), f"{tempo:.6f}", str(semi), "1" if formant else "0"],
                       check=True, timeout=1800)
        return np.frombuffer(b.read_bytes(), dtype="<f4").reshape(-1, 2).copy()


def _render_ffmpeg(path: Path, semi: int, tempo: float, raw: Path) -> np.ndarray:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-af", _atempo_chain(semi, tempo),
                    "-ar", str(SR), "-ac", "2", "-c:a", "pcm_f32le", str(raw)], check=True)
    orig, _ = sf.read(path, always_2d=True, dtype="float32")
    y, _ = sf.read(raw, always_2d=True, dtype="float32")
    early = _early_samples(orig, y, tempo)  # ffmpeg starts early by an amount that differs per stem
    if early > 0:
        y = np.pad(y, ((early, 0), (0, 0)))
    elif early < 0:
        y = y[-early:]
    return y


def shifted_file(path: Path, semi: int, tempo: float, hq: bool = True) -> Path:
    """The stem at `semi` semitones and `tempo` times the speed (cached). `path` itself when nothing changes."""
    if semi == 0 and abs(tempo - 1) < 1e-4:
        return path
    st = path.stat()
    engine = "rb" if hq and RUBBERBAND.exists() else "ff"
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{semi}|{tempo:.4f}|{engine}".encode()).hexdigest()[:24]
    out = SHIFT_CACHE / f"{key}.wav"
    with _shift_guard:
        lock = _shift_locks.setdefault(key, threading.Lock())
    with lock:  # the mixer asks for all stems at once; two requests for the same file render it only once
        if out.exists():
            os.utime(out)
            return out
        raw, tmp = SHIFT_CACHE / f"{key}.raw.wav", SHIFT_CACHE / f"{key}.tmp.wav"
        try:
            orig_frames = sf.info(path).frames
            y = None
            if engine == "rb":
                try:
                    y = _render_rubberband(path, semi, tempo, formant=path.name.startswith("vocals__"))
                except (OSError, subprocess.SubprocessError, ValueError):
                    y = None  # the helper could not run on this Mac: fall back to ffmpeg
            if y is None:
                y = _render_ffmpeg(path, semi, tempo, raw)
            frames = int(round(orig_frames / tempo))
            y = y[:frames] if len(y) >= frames else np.pad(y, ((0, frames - len(y)), (0, 0)))
            floaty = sf.info(path).subtype == "FLOAT"
            sf.write(tmp, y if floaty else np.clip(y, -1, 1), SR, subtype="FLOAT" if floaty else "PCM_16")
            tmp.replace(out)
        finally:
            raw.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    _prune_shift_cache()
    return out


def _prune_shift_cache(limit_gb: float = 3.0) -> None:
    """Rendered copies are disposable: keep the cache under a size limit, dropping the least recently used."""
    try:
        files = sorted((f for f in SHIFT_CACHE.glob("*.wav") if f.name.count(".") == 1), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while files and total > limit_gb * 1e9:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink(missing_ok=True)
    except OSError:
        pass


def with_mix(path: Path, semi: int, tempo: float, hq: bool = True) -> Path:
    """Apply the project's key shift and tempo to one stem file (drums only follow the tempo)."""
    if abs(semi) > SEMI_MAX or not TEMPO_MIN <= tempo <= TEMPO_MAX:
        raise HTTPException(400, "Key shift is limited to one octave and tempo to 50% to 200%")
    if path.name.split("__")[0] in UNPITCHED:
        semi = 0
    try:
        return shifted_file(path, semi, tempo, hq)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        raise HTTPException(500, f"Could not change the key or tempo: {e}") from e


def mix_of(job: dict) -> tuple[int, float]:
    m = job.get("mix") or {}
    return int(m.get("semitones", 0)), float(m.get("tempo", 1.0))


_prune_shift_cache()


# --- planning --------------------------------------------------------------
class PlanReq(BaseModel):
    source_id: str
    goal: str  # all (every instrument) | split (vocals + instrumental) | clean (voice clean up)
    denoise: bool = False  # set automatically for the clean goal
    dereverb: bool = False
    fast: bool = False  # set automatically for long songs


def make_plan(req: PlanReq, duration: float) -> dict:
    g = req.goal
    steps, stems, notes = [], [], []

    def step(sid, mkey, label):
        steps.append({"id": sid, "model": mkey, "label": label, "engine": M[mkey][1], "why": M[mkey][2],
                      "est": est(mkey, duration), "status": "pending", "ready": model_ready(mkey)})

    if g == "clean":
        step("denoise", "denoise", "Remove noise")
        step("dereverb", "dereverb", "Remove reverb")
        stems = ["cleaned"]
        notes.append("Cleans the whole recording. Best for speech, podcasts and calls.")
    elif g in ("all", "split"):
        step("split", "vocal_split_fast" if req.fast else "vocal_split", "Separate vocals from the music")
        if g == "all":
            step("instr", "stems6", "Separate the instruments")
            stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        else:
            stems = ["vocals", "instrumental"]
    else:
        raise HTTPException(400, "Unknown goal")
    return {"steps": steps, "stems": stems, "stem_labels": [LABELS.get(s, s.title()) for s in stems],
            "total_est": sum(s["est"] for s in steps), "notes": notes, "duration": round(duration, 1),
            "needs_download": [s["engine"] for s in steps if not s["ready"]]}


def pdir(pid: str) -> Path:
    try:
        return PATHS[pid]
    except KeyError:
        raise HTTPException(404, "Project not found") from None


# --- execution -------------------------------------------------------------
def save_job(job: dict) -> None:
    (pdir(job["id"]) / "job.json").write_text(json.dumps(job))


def add_variant(job: dict, name: str, variant: str, src: Path, frames: int, floaty: bool = False) -> None:
    fn = f"{name}__{variant}.wav"
    conform(src, pdir(job["id"]) / fn, frames, floaty)
    st = next((s for s in job["stems"] if s["name"] == name), None)
    if not st:
        st = {"name": name, "label": LABELS.get(name, name.title()), "variants": {}, "default": variant}
        job["stems"].append(st)
    st["variants"][variant] = fn
    st["default"] = variant


def _heal_to(src: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    dst = work / f"healed_{src.stem}.wav"
    heal_file(src, dst)
    return dst


def run_job(job_id: str) -> None:
    job = jobs[job_id]
    out = pdir(job_id)
    work = out / "work"
    src = out / "source.wav"
    dur = sources[job["source_id"]]["duration"]
    frames = sf.info(src).frames
    plan, req = job["plan"], PlanReq(**job["req"])
    job["status"] = "running"
    idx = {s["id"]: i for i, s in enumerate(job["steps"])}

    def timed(sid, fn):
        s = job["steps"][idx[sid]]
        s["status"], s["t0"] = "running", time.time()
        res = fn()
        s["status"], s["actual"] = "done", round(time.time() - s["t0"], 1)
        if sid in ("split", "instr", "denoise", "dereverb"):
            record_timing(s["model"], dur, s["actual"])
        return res

    try:
        if req.goal == "clean":
            cur = src
            for sid in ("denoise", "dereverb"):
                if sid in idx:
                    cur = timed(sid, lambda sid=sid, cur=cur: pick(run_model(M[sid][0], cur, work / sid), "dry", "noreverb"))
            add_variant(job, "cleaned", "original", src, frames)
            add_variant(job, "cleaned", "cleaned", cur, frames)
        else:
            raw: dict[str, Path] = {}
            vp, inst_path = timed("split", lambda: split_vocals(src, work / "split", req.fast, frames))
            raw["vocals"] = vp
            if "instr" in idx:
                key = job["steps"][idx["instr"]]["model"]
                sub = timed("instr", lambda: run_model(M[key][0], inst_path, work / "instr"))
                raw.update({k: p for k, p in sub.items() if k != "vocals"})
            else:
                raw["instrumental"] = inst_path
            for name in plan["stems"]:
                add_variant(job, name, "split", raw[name], frames)
            if "clean_vocals" in idx and "vocals" in plan["stems"]:
                def clean():
                    cur = raw["vocals"]
                    if req.denoise:
                        cur = pick(run_model(M["denoise"][0], cur, work / "dn"), "dry")
                    if req.dereverb:
                        cur = pick(run_model(M["dereverb"][0], cur, work / "dr"), "noreverb")
                    return cur
                add_variant(job, "vocals", "cleaned", timed("clean_vocals", clean), frames)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        job["elapsed"] = round(time.time() - job["started"], 1)
        save_job(job)


def run_heal_one(job_id: str, stem: str) -> None:
    job = jobs[job_id]
    st = next(s for s in job["stems"] if s["name"] == stem)
    cur = pdir(job_id) / st["variants"][st["default"]]
    frames = sf.info(cur).frames
    try:
        t0 = time.time()
        add_variant(job, stem, "healed", _heal_to(cur, pdir(job_id) / "work"), frames, floaty=True)
        record_timing("heal", frames / SR, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        job["heal_error"] = str(e)
    finally:
        shutil.rmtree(pdir(job_id) / "work", ignore_errors=True)
        job["healing"] = None
        save_job(job)


def run_analysis(job_id: str) -> None:
    """Find the key, tempo, beat map and time signature. Anything unclear is stored as 'not found' with the reason."""
    job = jobs.get(job_id)
    if not job or job_id not in PATHS:
        return
    try:
        out = pdir(job_id)
        files = {st["name"]: out / st["variants"][st["default"]] for st in job["stems"]}
        job["analysis"] = analysis.analyze_project({k: p for k, p in files.items() if p.exists()}, out / "source.wav")
    except Exception as e:  # noqa: BLE001  (analysis is a bonus: it must never get in the way of the project)
        job["analysis"] = {"version": analysis.VERSION, "key": {"found": False, "reason": f"Analysis failed: {e}"},
                           "tempo": {"found": False, "reason": f"Analysis failed: {e}"}}
    finally:
        job.pop("analyzing", None)
        try:
            save_job(job)
        except HTTPException:  # the project was removed while it ran
            pass


def worker() -> None:
    while True:
        kind, arg = tasks.get()
        try:
            set_priority()
            if kind == "job":
                jobs[arg]["started"] = time.time()
                run_job(arg)
                if jobs[arg].get("status") == "done":
                    jobs[arg]["analyzing"] = True
                    run_analysis(arg)
            elif kind == "analyze":
                run_analysis(arg)
            elif kind == "heal":
                run_heal_one(*arg)
        except Exception as e:  # noqa: BLE001
            print("task failed:", kind, e)


threading.Thread(target=worker, daemon=True).start()


# --- projects: one .umx package per song ------------------------------------
def _read(f: Path):
    try:
        return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return None


def _unique_pkg(root: Path, title: str) -> Path:
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title).strip(". ")[:80] or "Untitled"
    cand, n = root / (base + EXT), 2
    while cand.exists():
        cand = root / f"{base} {n}{EXT}"
        n += 1
    return cand


def _reid(pkg: Path, old: str, new: str) -> None:
    meta = _read(pkg / "meta.json") or {}
    meta["id"] = new
    (pkg / "meta.json").write_text(json.dumps(meta))
    job = _read(pkg / "job.json")
    if job:
        job.update(id=new, source_id=new)
        (pkg / "job.json").write_text(json.dumps(job))


def register(pkg: Path) -> str | None:
    """Load a package into memory and return its project id (None if it isn't a valid project)."""
    meta = _read(pkg / "meta.json")
    if not meta or not meta.get("id") or not (pkg / "source.wav").exists():
        return None
    pid = meta["id"]
    if pid in PATHS and PATHS[pid].resolve() != pkg.resolve():  # a copy of a project that is already open
        new = "p" + uuid.uuid4().hex[:9]
        _reid(pkg, pid, new)
        pid = new
        meta["id"] = new
    PATHS[pid] = pkg
    sources[pid] = meta
    job = _read(pkg / "job.json")
    if job:
        job["source_id"] = pid
        job.pop("analyzing", None)  # an analysis that was cut short by closing the app starts again when the project opens
        if job.get("status") in ("queued", "running"):
            job.update(status="error", error="Interrupted: Unmix was closed while this was running.")
        jobs[pid] = job
    return pid


def _migrate_legacy(root: Path) -> None:
    """Older versions kept a source folder and a result folder per song. Wrap each song into one package."""
    dirs = [d for d in root.iterdir() if d.is_dir() and d.suffix != EXT and not d.name.startswith(".")]
    used: set[str] = set()
    for jd in dirs:
        job = _read(jd / "job.json")
        sd = root / str((job or {}).get("source_id", "."))
        if not job or not job.get("id") or not (sd / "source.wav").is_file():
            continue
        pkg = _unique_pkg(root, job.get("title") or "Untitled")
        try:
            pkg.mkdir()
            shutil.copy2(sd / "source.wav", pkg / "source.wav")
            for f in jd.iterdir():
                if f.is_file() and f.name != "job.json":
                    shutil.move(str(f), pkg / f.name)
            meta = _read(sd / "meta.json") or {}
            meta.update(id=job["id"], title=job.get("title") or meta.get("title", "Untitled"))
            job["source_id"] = job["id"]
            (pkg / "meta.json").write_text(json.dumps(meta))
            (pkg / "job.json").write_text(json.dumps(job))
        except Exception:  # noqa: BLE001  (leave the old folders untouched if anything fails)
            shutil.rmtree(pkg, ignore_errors=True)
            continue
        used.add(sd.name)
        shutil.rmtree(jd, ignore_errors=True)
    for sd in dirs:  # songs that were uploaded but never split
        meta = _read(sd / "meta.json")
        if sd.exists() and sd.name not in used and meta and meta.get("id") and (sd / "source.wav").is_file() and not (sd / "job.json").exists():
            pkg = _unique_pkg(root, meta.get("title") or "Untitled")
            try:
                pkg.mkdir()
                shutil.move(str(sd / "source.wav"), pkg / "source.wav")
                (pkg / "meta.json").write_text(json.dumps(meta))
                used.add(sd.name)
            except Exception:  # noqa: BLE001
                shutil.rmtree(pkg, ignore_errors=True)
    for name in used:
        shutil.rmtree(root / name, ignore_errors=True)


def restore() -> None:
    try:
        _migrate_legacy(DATA)
    except Exception as e:  # noqa: BLE001
        print("library migration failed:", e)
    for d in sorted(DATA.iterdir()):
        if d.is_dir() and d.suffix == EXT:
            register(d)
        elif d.is_dir() and d.name.startswith(".new-"):
            shutil.rmtree(d, ignore_errors=True)  # an upload that was interrupted
    opened = [p for p in _cfg().get("opened", []) if Path(p).is_dir()]  # projects opened from outside the library
    for p in opened:
        register(Path(p))
    _save_cfg(opened=opened)


restore()


def _open_path(raw: str) -> str:
    p = Path(raw).expanduser()
    if p.suffix != EXT or not p.is_dir():
        raise HTTPException(400, "That isn't an Unmix project (.umx).")
    pid = register(p)
    if not pid:
        raise HTTPException(400, "That project looks damaged: its audio is missing.")
    if not p.resolve().is_relative_to(DATA.resolve()):
        opened = _cfg().get("opened", [])
        if str(p) not in opened:
            _save_cfg(opened=[*opened, str(p)])
    return pid


def _forget(pid: str) -> None:
    p = PATHS.pop(pid, None)
    jobs.pop(pid, None)
    sources.pop(pid, None)
    if p:
        _save_cfg(opened=[x for x in _cfg().get("opened", []) if x != str(p)])


def _trash(p: Path) -> None:
    """Move to the Trash (Finder's 'Put Back' works), so a misclick never deletes a song for good."""
    try:
        from Foundation import NSFileManager, NSURL  # noqa: PLC0415

        ok, _url, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
            NSURL.fileURLWithPath_(str(p)), None, None)
        if not ok:
            raise OSError(str(err))
    except ImportError:
        shutil.rmtree(p, ignore_errors=True)  # not on a Mac with PyObjC (development only)


def _save_meta(pid: str) -> None:
    (pdir(pid) / "meta.json").write_text(json.dumps(sources[pid]))


def _dialog(kind, **kw):
    try:
        import webview  # noqa: PLC0415

        return webview.windows[0].create_file_dialog(kind, **kw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "File choosers are only available in the Unmix app.") from e


def _first(res) -> str | None:
    if not res:
        return None
    return str(res if isinstance(res, str) else res[0])


# --- storage location --------------------------------------------------------
def _is_project_dir(d: Path) -> bool:
    return d.is_dir() and d.suffix == EXT and (d / "meta.json").exists()


def _busy() -> bool:
    return any(j.get("status") in ("queued", "running") for j in jobs.values()) or not tasks.empty()


def _tree_mb(d: Path) -> float:
    return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1)


def _storage_info() -> dict:
    return {
        "path": str(DATA), "default_path": str(DEFAULT_DATA), "is_default": DATA == DEFAULT_DATA,
        "missing": LIBRARY_MISSING, "projects": len(sources),
        "size_mb": _tree_mb(DATA), "free_gb": round(shutil.disk_usage(DATA).free / 1e9, 1),
    }


def _activate_library(path: Path) -> None:
    """Point the app at another library folder and load whatever projects it already holds."""
    global DATA, LIBRARY_MISSING
    DATA, LIBRARY_MISSING = path, None
    jobs.clear()
    sources.clear()
    PATHS.clear()
    _save_cfg(library_dir=None if path == DEFAULT_DATA else str(path))
    restore()


def _move_library(src: Path, dst: Path) -> None:
    """Copy every project to dst, verify sizes, and only then delete the originals."""
    items = [d for d in src.iterdir() if _is_project_dir(d)]
    clash = [d.name for d in items if (dst / d.name).exists()]
    if clash:
        raise HTTPException(409, "That folder already contains projects with the same names. Choose an empty folder.")
    need = sum(_tree_mb(d) for d in items) / 1e3
    if shutil.disk_usage(dst).free / 1e9 < need + 0.5:
        raise HTTPException(409, f"Not enough free space there ({need:.1f} GB needed).")
    done: list[Path] = []
    try:
        for d in items:
            shutil.copytree(d, dst / d.name)
            if _tree_mb(d) != _tree_mb(dst / d.name):
                raise OSError(f"Copy of {d.name} did not match the original")
            done.append(d)
    except Exception as e:  # noqa: BLE001  (undo the partial copy; originals are untouched)
        for d in done:
            shutil.rmtree(dst / d.name, ignore_errors=True)
        raise HTTPException(500, f"Could not move the library: {e}") from e
    for d in items:
        shutil.rmtree(d, ignore_errors=True)


class StorageReq(BaseModel):
    path: str | None = None  # None = back to the default location
    move: bool = True


@app.get("/api/storage")
def storage_info():
    return _storage_info()


@app.post("/api/storage/pick")
def storage_pick():
    """Native macOS folder chooser (only available inside the desktop window)."""
    import webview  # noqa: PLC0415

    return {"path": _first(_dialog(webview.FileDialog.FOLDER, directory=str(DATA.parent)))}


@app.post("/api/storage/set")
def storage_set(req: StorageReq):
    if _busy():
        raise HTTPException(409, "Wait for the current processing to finish before changing the location.")
    dst = DEFAULT_DATA if req.path is None else Path(req.path).expanduser()
    try:
        dst.mkdir(parents=True, exist_ok=True)
        dst = dst.resolve()
    except OSError as e:
        raise HTTPException(400, f"Can't use that folder: {e.strerror or e}") from e
    if not os.access(dst, os.W_OK):
        raise HTTPException(400, "Unmix can't write to that folder. Choose another one.")
    if dst == DATA.resolve():
        return _storage_info()
    if dst.is_relative_to(DATA.resolve()) or DATA.resolve().is_relative_to(dst):
        raise HTTPException(400, "Choose a folder that is not inside the current library (or containing it).")
    if req.move and DATA.is_dir():
        _move_library(DATA, dst)
    _activate_library(dst)
    return _storage_info()


class RevealReq(BaseModel):
    id: str | None = None


@app.post("/api/reveal")
def reveal(req: RevealReq):
    """Show a project (or the whole library) in Finder."""
    if req.id:
        subprocess.run(["/usr/bin/open", "-R", str(pdir(req.id))], check=False)
    else:
        subprocess.run(["/usr/bin/open", str(DATA)], check=False)
    return {"ok": True}


class OpenReq(BaseModel):
    path: str


@app.post("/api/open")
def open_project(req: OpenReq):
    return {"id": _open_path(req.path)}


@app.post("/api/open/pick")
def open_pick():
    import webview  # noqa: PLC0415

    path = _first(_dialog(webview.FileDialog.OPEN, directory=str(DATA), file_types=("Unmix project (*.umx)",)))
    return {"id": _open_path(path)} if path else {"id": None}


@app.get("/api/open/pending")
def open_pending():
    """Projects the user double-clicked in Finder (handed over by the desktop shell)."""
    ids = []
    while openq.paths:
        try:
            ids.append(_open_path(openq.paths.pop(0)))
        except HTTPException:
            pass
    return {"ids": ids}


@app.post("/api/projects/{pid}/copy")
def save_copy(pid: str):
    import webview  # noqa: PLC0415

    src = pdir(pid)
    home = Path.home() / "Documents"  # not the library folder: the project already lives there, so the panel would ask to replace it
    to = _first(_dialog(webview.FileDialog.SAVE, directory=str(home if home.is_dir() else Path.home()), save_filename=src.name))
    if not to:
        return {"id": None}
    dst = Path(to)
    if dst.suffix != EXT:
        dst = dst.with_name(dst.name + EXT)
    if dst.resolve() == src.resolve():
        raise HTTPException(400, "Choose a different name or folder for the copy.")
    shutil.rmtree(dst, ignore_errors=True)  # the save panel already asked about replacing it
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("work", "analyze"))
    _reid(dst, pid, "p" + uuid.uuid4().hex[:9])
    return {"id": _open_path(str(dst)), "path": str(dst)}


@app.post("/api/projects/{pid}/save")
def save_project(pid: str):
    """Save panel for a finished unmix: the project is moved to the chosen name and place (it stays one project, no duplicate)."""
    import webview  # noqa: PLC0415

    if jobs.get(pid, {}).get("status") in ("queued", "running"):
        raise HTTPException(409, "This project is still being split")
    src = pdir(pid)
    to = _first(_dialog(webview.FileDialog.SAVE, directory=str(DATA), save_filename=src.name))
    if not to:
        return {"id": None}
    dst = Path(to)
    if dst.suffix != EXT:
        dst = dst.with_name(dst.name + EXT)
    if dst.resolve() == src.resolve():
        return {"id": pid, "path": str(src)}
    shutil.rmtree(dst, ignore_errors=True)  # the save panel already asked about replacing it
    shutil.move(str(src), str(dst))
    _forget(pid)
    return {"id": _open_path(str(dst)), "path": str(dst)}


# --- API -------------------------------------------------------------------
# --- self-update (from the Band OS server; see BandOS backend /unmix/update) -
UPDATE_API = os.environ.get("UNMIX_UPDATE_API", "https://bandos-backend.fly.dev/unmix/update")
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "0.0.0"
update_state: dict = {"phase": "idle", "error": None, "progress": 0.0}


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _install_info() -> dict:
    """Anonymous facts sent with the update check so Band OS can count installs and versions: a random id made
    on first run (not tied to you), the macOS version, chip, Mac model and memory. No files, names or audio."""
    iid = _cfg().get("install_id")
    if not iid:
        iid = str(uuid.uuid4())
        _save_cfg(install_id=iid)
    mem = _sysctl("hw.memsize")
    return {
        "id": iid,
        "os": platform.mac_ver()[0],
        "chip": _sysctl("machdep.cpu.brand_string") or platform.machine(),
        "model": _sysctl("hw.model"),
        "ram": round(int(mem) / 2**30) if mem.isdigit() else "",
    }


def _latest_release() -> dict | None:
    """The newer version offered by the server (version, notes, size, sha256, url), or None."""
    query = urllib.parse.urlencode({"current": VERSION, **_install_info()})
    req = urllib.request.Request(f"{UPDATE_API}?{query}", headers={"User-Agent": f"Unmix/{VERSION}"})
    with urllib.request.urlopen(req, timeout=8) as r:
        rel = json.load(r)["data"]
    return rel if rel.get("available") else None


# --- confirm your email (first launch, installed app only) ---------------------
# The app asks which email downloaded it, Band OS emails a 6 digit code, and a confirmed email is linked to this
# install so the Band OS admin can see who uses Unmix. Skipping keeps the install anonymous.
LINK_API = UPDATE_API.rsplit("/", 1)[0] + "/link"


def _link_call(path: str, body: dict) -> None:
    req = urllib.request.Request(f"{LINK_API}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": f"Unmix/{VERSION}"})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error")
        except Exception:  # noqa: BLE001
            msg = None
        raise HTTPException(400, msg or "Could not reach Band OS. Please try again.")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Could not reach Band OS. Check your internet connection.")


class EmailBody(BaseModel):
    email: str
    code: str = ""


@app.get("/api/install/status")
def install_status():
    cfg = _cfg()
    return {"ask": bool(getattr(sys, "frozen", False)) and not cfg.get("install_email") and not cfg.get("link_skipped"),
            "email": cfg.get("install_email")}


@app.post("/api/install/request")
def install_request(b: EmailBody):
    _link_call("request", {"email": b.email})
    return {"ok": True}


@app.post("/api/install/confirm")
def install_confirm(b: EmailBody):
    _link_call("confirm", {"email": b.email, "code": b.code, "id": _install_info()["id"], "current": VERSION})
    _save_cfg(install_email=b.email.strip().lower())
    return {"ok": True}


@app.post("/api/install/skip")
def install_skip():
    _save_cfg(link_skipped=True)
    return {"ok": True}


@app.get("/api/version")
def version():
    return {"version": VERSION, "can_update": bool(getattr(sys, "frozen", False))}


@app.get("/api/update/check")
def update_check():
    try:
        rel = _latest_release()
    except Exception:  # noqa: BLE001  (offline, rate-limited, no releases yet: just say nothing new)
        return {"available": False}
    if not rel:
        return {"available": False}
    return {"available": True, "version": rel["version"], "notes": rel["notes"], "size_mb": round(rel["size"] / 1e6)}


def _do_update() -> None:
    try:
        rel = _latest_release()
        if not rel:
            raise RuntimeError("No newer version found")
        url = rel["url"]
        if urllib.parse.urlparse(url).scheme != "https":
            raise RuntimeError("Unexpected download location")
        dest = Path(tempfile.mkdtemp(prefix="unmix-update-")) / f"Unmix-{rel['version']}.pkg"
        update_state.update(phase="downloading", progress=0.0, error=None)
        h = hashlib.sha256()
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": f"Unmix/{VERSION}"}), timeout=30) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or rel["size"])
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                update_state["progress"] = min(f.tell() / total, 1.0)
        if h.hexdigest() != rel["sha256"] or dest.stat().st_size != rel["size"]:
            raise RuntimeError("Download is corrupt; try again")
        update_state["phase"] = "installing"
        # The installer re-runs the system check, replaces Unmix.app and its postinstall script relaunches it.
        subprocess.run(["/usr/bin/open", "-a", "Installer", str(dest)], check=True)
        time.sleep(1.5)
        os._exit(0)
    except Exception as e:  # noqa: BLE001
        update_state.update(phase="error", error=str(e))


@app.post("/api/update/install")
def update_install():
    if not getattr(sys, "frozen", False):
        raise HTTPException(400, "Updates only apply to the installed app")
    if update_state["phase"] in ("downloading", "installing"):
        return update_state
    update_state.update(phase="downloading", progress=0.0, error=None)
    threading.Thread(target=_do_update, daemon=True).start()
    return update_state


@app.get("/api/update/status")
def update_status():
    return update_state


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "index.html").read_text()


@app.post("/api/sources")
def create_source(file: UploadFile | None = File(None), url: str = Form("")):
    url = url.strip()
    if not file and not url:
        raise HTTPException(400, "Provide a file or a link")
    if url and not re.match(r"^https?://", url):
        raise HTTPException(400, "Link must start with http(s)://")
    sid = "p" + uuid.uuid4().hex[:9]
    d = DATA / f".new-{sid}"  # becomes "<song name>.umx" once the title is known
    d.mkdir()
    try:
        if url:
            raw, title = download_audio(url, d)
        else:
            title = Path(file.filename or "audio").stem
            raw = d / ("upload" + (Path(file.filename or "").suffix or ".bin"))
            with open(raw, "wb") as f:
                shutil.copyfileobj(file.file, f)
        to_source(raw, d / "source.wav")
        raw.unlink(missing_ok=True)
        meta = {"id": sid, "title": title, "duration": round(sf.info(d / "source.wav").duration, 1), "created": time.time()}
        (d / "meta.json").write_text(json.dumps(meta))
        pkg = _unique_pkg(DATA, title)
        d.rename(pkg)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not read that audio: {e}") from e
    PATHS[sid] = pkg
    sources[sid] = meta
    return meta


@app.get("/api/sources/{sid}/peaks")
def source_peaks(sid: str, n: int = 240):
    """Loudness outline of the source audio (n values 0..1) for the waveform preview."""
    if sid not in sources:
        raise HTTPException(404, "Unknown source")
    path = pdir(sid) / "source.wav"
    n = max(20, min(n, 1000))
    frames = sf.info(path).frames
    peaks = []
    for blk in sf.blocks(path, blocksize=max(1, frames // n + 1), dtype="float32", always_2d=True):
        peaks.append(float(abs(blk).max()))
    top = max(peaks) or 1.0
    return {"peaks": [round(p / top, 3) for p in peaks], "duration": sources[sid].get("duration")}


@app.get("/api/sources/{sid}")
def get_source(sid: str):
    if sid not in sources:
        raise HTTPException(404)
    return sources[sid]


FAST_ABOVE = 300  # seconds: songs longer than this use the lighter single vocal model


@app.post("/api/plan")
def plan(req: PlanReq):
    if req.source_id not in sources:
        raise HTTPException(404, "Unknown source")
    dur = sources[req.source_id]["duration"]
    if req.goal == "clean":  # voice clean up always does both passes
        req.denoise = req.dereverb = True
    elif dur > FAST_ABOVE:  # the model is chosen by weight: long songs get the lighter model
        req.fast = True
    return make_plan(req, dur)


@app.post("/api/run")
def run(req: PlanReq):
    p = plan(req)
    pid = req.source_id
    old = jobs.get(pid)
    if old and old.get("status") in ("queued", "running"):
        raise HTTPException(409, "This project is already being split")
    d = pdir(pid)
    for f in d.iterdir():  # a re-split replaces the previous stems; the source audio stays
        if f.name not in ("source.wav", "meta.json"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    jobs[pid] = {"id": pid, "source_id": pid, "req": req.model_dump(), "plan": p, "steps": p["steps"],
                 "stems": [], "status": "queued", "started": time.time(), "title": sources[pid]["title"],
                 "duration": sources[pid].get("duration"), "mix": (old or {}).get("mix") or {"semitones": 0, "tempo": 1.0}}
    save_job(jobs[pid])
    tasks.put(("job", pid))
    return {"id": pid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


def _dir_mb(d: Path) -> float:
    return round(sum(f.stat().st_size for f in d.glob("*") if f.is_file()) / 1e6, 1)


@app.get("/api/library")
def library():
    items = []
    for pid, src in sources.items():
        j = jobs.get(pid)
        done = bool(j) and j.get("status") == "done"
        items.append({
            "id": pid, "title": (j or src).get("title", "Untitled"), "started": (j or src).get("started") or src.get("created") or 0,
            "duration": src.get("duration"), "status": (j or {}).get("status", "new"),
            "stems": [st["label"] for st in j.get("stems", [])] if done else [],
            "size_mb": _dir_mb(pdir(pid)), "path": str(PATHS[pid]),
        })
    items.sort(key=lambda i: i["started"] or 0, reverse=True)
    return items


class JobPatch(BaseModel):
    title: str | None = None


@app.patch("/api/jobs/{jid}")
def patch_job(jid: str, req: JobPatch):
    if jid not in sources:
        raise HTTPException(404)
    if "title" in req.model_fields_set:
        title = (req.title or "").strip()[:200]
        if not title:
            raise HTTPException(400, "Name cannot be empty")
        sources[jid]["title"] = title
        if jid in jobs:
            jobs[jid]["title"] = title
        old = pdir(jid)
        if old.stem != _safe_name(title):
            try:  # keep the file name in Finder in step with the song name
                new = _unique_pkg(old.parent, title)
                old.rename(new)
                PATHS[jid] = new
                _save_cfg(opened=[str(new) if x == str(old) else x for x in _cfg().get("opened", [])])
            except OSError:
                pass
        _save_meta(jid)
    if jid in jobs:
        save_job(jobs[jid])
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    """Moves the project to the Trash (it is a normal file, so it can be put back from there)."""
    pkg = pdir(jid)
    if jobs.get(jid, {}).get("status") in ("queued", "running"):
        raise HTTPException(409, "This project is still being split")
    _trash(pkg)
    _forget(jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/heal/{stem}")
def heal_stem(jid: str, stem: str):
    job = jobs.get(jid)
    if not job or not any(s["name"] == stem for s in job["stems"]):
        raise HTTPException(404)
    if job.get("healing"):
        raise HTTPException(409, "Already healing a stem")
    job["healing"] = stem
    job.pop("heal_error", None)
    tasks.put(("heal", (jid, stem)))
    return {"ok": True}


class MixReq(BaseModel):
    semitones: int = 0
    tempo: float = 1.0  # speed relative to the original: 1.0 = unchanged


@app.put("/api/jobs/{jid}/mix")
def set_mix(jid: str, req: MixReq):
    """Remember the key shift and tempo in the project file (the stems themselves are rendered on request)."""
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404)
    if abs(req.semitones) > SEMI_MAX or not TEMPO_MIN <= req.tempo <= TEMPO_MAX:
        raise HTTPException(400, "Key shift is limited to one octave and tempo to 50% to 200%")
    job["mix"] = {"semitones": req.semitones, "tempo": round(req.tempo, 4)}
    save_job(job)
    return job["mix"]


@app.post("/api/jobs/{jid}/analyze")
def analyze_job(jid: str, force: bool = False):
    """Key, tempo and time signature of a finished project (older projects are analysed the first time they open; `force` redoes it on request)."""
    job = jobs.get(jid)
    if not job or job.get("status") != "done":
        raise HTTPException(404)
    if (force or job.get("analysis", {}).get("version", 0) < analysis.VERSION) and not job.get("analyzing"):
        job["analyzing"] = True
        tasks.put(("analyze", jid))
    return {"ok": True}


@app.get("/api/jobs/{jid}/files/{name}")
def get_file(jid: str, name: str, semi: int = 0, tempo: float = 1.0, fast: bool = False):
    path = pdir(jid) / Path(name).name
    if jid not in jobs or not path.is_file():
        raise HTTPException(404)
    return FileResponse(with_mix(path, semi, tempo, hq=not fast), headers={"Cache-Control": "no-cache"})


# --- bounce: export to wherever the user chooses -------------------------------
FORMATS = ("wav", "flac", "mp3")
BOUNCES: dict[str, Path] = {}


def _write_audio(src: Path, fmt: str, dst: Path) -> None:
    if fmt == "wav":
        shutil.copyfile(src, dst)
        return
    args = ["-b:a", "320k"] if fmt == "mp3" else []
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), *args, str(dst)], check=True)


def _safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", s).strip(". ")[:100] or "Unmix"


class BounceWhere(BaseModel):
    name: str
    fmt: str


@app.post("/api/bounce/where")
def bounce_where(req: BounceWhere):
    """Step 1 of a mix bounce: the native Save dialog, so the file goes wherever the user wants."""
    import webview  # noqa: PLC0415

    if req.fmt not in FORMATS:
        raise HTTPException(400, "Unknown format")
    to = _first(_dialog(webview.FileDialog.SAVE, directory=str(Path.home() / "Music" if (Path.home() / "Music").is_dir() else Path.home()),
                        save_filename=f"{_safe_name(req.name)}.{req.fmt}"))
    if not to:
        return {"token": None}
    dst = Path(to)
    if not dst.name.lower().endswith("." + req.fmt):
        dst = dst.with_name(dst.name + "." + req.fmt)
    token = uuid.uuid4().hex
    BOUNCES[token] = dst
    return {"token": token, "path": str(dst)}


@app.post("/api/bounce/write")
async def bounce_write(token: str, request: Request):
    """Step 2: receives the rendered mix (WAV) from the mixer and writes it in the chosen format."""
    dst = BOUNCES.pop(token, None)
    if not dst:
        raise HTTPException(404, "Choose where to save first")
    data = await request.body()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "mix.wav"
        tmp.write_bytes(data)
        try:
            _write_audio(tmp, dst.suffix.lstrip(".").lower(), dst)
        except (OSError, subprocess.CalledProcessError) as e:
            raise HTTPException(500, f"Couldn't save there: {e}") from e
    return {"path": str(dst)}


class BounceStems(BaseModel):
    id: str
    fmt: str


@app.post("/api/bounce/stems")
def bounce_stems(req: BounceStems):
    """Every stem as its own file in a folder the user picks (to drag into another DAW)."""
    import webview  # noqa: PLC0415

    job = jobs.get(req.id)
    if not job or req.fmt not in FORMATS:
        raise HTTPException(404)
    home = Path.home()
    folder = _first(_dialog(webview.FileDialog.FOLDER, directory=str(home / "Music" if (home / "Music").is_dir() else home)))
    if not folder:
        return {"path": None}
    try:
        semi, tempo = mix_of(job)
        for st in job["stems"]:
            _write_audio(with_mix(pdir(req.id) / st["variants"][st["default"]], semi, tempo), req.fmt,
                         Path(folder) / f"{_safe_name(job['title'])} - {_safe_name(st['label'])}.{req.fmt}")
    except (OSError, subprocess.CalledProcessError) as e:
        raise HTTPException(500, f"Couldn't save there: {e}") from e
    return {"path": folder}
