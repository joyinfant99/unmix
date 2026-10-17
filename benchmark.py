"""Measure per-model speed on this machine and write timings.json (used for time estimates)."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import app

src = Path(sys.argv[1])  # any audio file, 30 s or longer
tmp = Path(tempfile.mkdtemp())
clips = {}
for n in (10, 30):
    clips[n] = tmp / f"c{n}.wav"
    subprocess.run([app.FFMPEG, "-y", "-loglevel", "error", "-t", str(n), "-i", str(src), "-ar", "44100", "-ac", "2", str(clips[n])], check=True)

def timeit(fn):
    t = time.time(); fn(); return time.time() - t

out = {}
for key in ("vocal_split", "stems4", "stems6", "denoise", "dereverb", "heal"):
    if key == "heal":
        fn = lambda c: app.heal_file(c, tmp / f"h{time.time_ns()}.wav")
    else:
        fn = lambda c, key=key: app.run_model(app.M[key][0], c, tmp / f"{key}{time.time_ns()}")
    print(key, "warm-up (may download)…", flush=True)
    fn(clips[10])
    t10, t30 = timeit(lambda: fn(clips[10])), timeit(lambda: fn(clips[30]))
    rtf = max((t30 - t10) / 20, 0.05)
    out[key] = {"rtf": round(rtf, 3), "base": round(max(t10 - 10 * rtf, 2), 1)}
    print(key, out[key], f"(10s={t10:.1f}s 30s={t30:.1f}s)", flush=True)
    app.TIMINGS_FILE.write_text(json.dumps(out, indent=1))
print("done")
