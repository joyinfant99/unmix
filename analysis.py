"""Key, tempo, beat map and time signature of a song.

Everything here is conservative on purpose: when the evidence is weak a field is None ("N/A" in the UI) instead of a
guess. A key or tempo is only reported when it holds for a large part of the song (`coverage`).
Uses librosa only (already installed for the separator); nothing leaves the Mac.
"""
import numpy as np

SR = 22050
HOP = 512
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler probe-tone profiles (tonic first)
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PROFILES = np.array([np.roll(p, k) for p in (_MAJ, _MIN) for k in range(12)])  # rows 0-11 major, 12-23 minor

MIN_KEY_COVERAGE = 0.55 # share of the song that must agree on one key
MIN_KEY_CORR = 0.65     # the chroma must actually look like a major or minor scale
MIN_KEY_MARGIN = 0.03   # and beat the runner-up clearly
MIN_TEMPO_COVERAGE = 0.55
MAX_TEMPO_DRIFT = 0.025  # the tempo of the first and last part of the song may differ by at most this much
MIN_BEAT_PEAK = 0.25     # autocorrelation peak that counts as 'the music repeats at this tempo'
MIN_ON_GRID = 0.6        # share of strong low hits that must fall on the beat grid before a time signature is claimed


def _key_label(idx: int) -> tuple[int, str]:
    return idx % 12, "major" if idx < 12 else "minor"


def _corr_rows(vec: np.ndarray) -> np.ndarray:
    v = vec - vec.mean()
    p = PROFILES - PROFILES.mean(axis=1, keepdims=True)
    den = np.linalg.norm(v) * np.linalg.norm(p, axis=1)
    return (p @ v) / np.where(den == 0, 1, den)


def detect_key(y: np.ndarray, bass: np.ndarray | None = None) -> dict | None:
    """Major or minor key held for most of the song, or None. `y` is harmonic mono audio at SR."""
    import librosa  # noqa: PLC0415

    if len(y) < SR * 8 or not np.any(y):
        return None
    tuning = float(librosa.estimate_tuning(y=y, sr=SR))
    hop = 4096
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=hop, tuning=tuning)
    if bass is not None and np.any(bass):  # the bass note is the strongest hint for the tonic (settles relative major/minor)
        bc = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=hop, tuning=tuning, fmin=librosa.note_to_hz("C1"), n_octaves=5)
        n = min(chroma.shape[1], bc.shape[1])
        chroma = chroma[:, :n] / (chroma[:, :n].sum(0, keepdims=True) + 1e-9) + 0.5 * bc[:, :n] / (bc[:, :n].sum(0, keepdims=True) + 1e-9)
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=hop * 2)[0][: chroma.shape[1]]
    chroma = chroma[:, : len(rms)]
    active = rms > 0.1 * np.median(rms[rms > 0]) if np.any(rms > 0) else np.zeros_like(rms, bool)
    if active.sum() < 30:
        return None
    fps = SR / hop
    win, step = int(15 * fps), int(7.5 * fps)
    windows: list[tuple[int, np.ndarray]] = []  # (best key of the window, its mean chroma)
    for s in range(0, max(1, chroma.shape[1] - win // 2), step):
        sl = slice(s, s + win)
        m = active[sl]
        if m.sum() < win * 0.4:
            continue
        mean_c = chroma[:, sl][:, m].mean(axis=1)
        c = _corr_rows(mean_c)
        top = int(np.argmax(c))
        if c[top] >= 0.5:
            windows.append((top, mean_c))
    if len(windows) < 2:
        return None
    ids = [w[0] for w in windows]
    best = max(set(ids), key=ids.count)  # the key most windows agree on (a song that modulates still has a main key)
    coverage = ids.count(best) / len(windows)
    glob = _corr_rows(np.mean([c for k, c in windows if k == best], axis=0))
    order = np.argsort(glob)[::-1]
    runner = int(order[1] if order[0] == best else order[0])
    out = {"coverage": round(coverage, 2), "score": round(float(glob[best]), 2), "margin": round(float(glob[best] - glob[runner]), 2),
           "tuning_cents": round(tuning * 100)}
    if glob[best] < MIN_KEY_CORR or glob[best] - glob[runner] < MIN_KEY_MARGIN or coverage < MIN_KEY_COVERAGE:
        out["reason"] = "No single major or minor key stands out for enough of the song"
        return {"found": False, **out}
    pc, mode = _key_label(best)
    return {"found": True, "pc": pc, "mode": mode, "tonic": NOTE_NAMES[pc], **out}


def _beat_support(oenv: np.ndarray, fps: float, bpm: float) -> tuple[float, int]:
    """Share of the playing stretches (8 s windows) whose onsets repeat at this tempo or an octave of it, and how many played."""
    win, hop = int(8 * fps), int(2 * fps)
    floor = 0.25 * np.median(oenv[oenv > 0]) if np.any(oenv > 0) else 0
    good = played = 0
    for s in range(0, max(1, len(oenv) - win + 1), hop):
        w = oenv[s: s + win]
        if len(w) < win or w.mean() < floor:
            continue
        x = w - w.mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        top = float(ac[int(0.3 * fps): int(3.5 * fps)].max())  # strongest repetition in this stretch
        peak = 0.0
        for m in (0.5, 1, 2, 4):
            lag = fps * 60.0 / bpm * m
            lo, hi = int(round(lag * 0.96)), int(round(lag * 1.04)) + 1
            if lo > 1 and hi < len(ac) - 1:
                i = lo + int(ac[lo:hi].argmax())
                if lo < i < hi - 1 and ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]:  # a real peak, not the tail of a neighbour
                    peak = max(peak, float(ac[i]))
        played += 1
        good += peak >= MIN_BEAT_PEAK and peak >= 0.35 * top
    return (good / played if played else 0.0), played


def _fold_octave(bpm: float) -> float:
    """The octave of bpm that sits closest to a natural walking tempo (105), within 65-175."""
    opts = [bpm * 2.0 ** k for k in range(-3, 4) if 65 <= bpm * 2.0 ** k < 175]
    return min(opts, key=lambda b: abs(np.log2(b / 105))) if opts else bpm


def _tempo_candidates(oenv: np.ndarray, fps: float) -> list[float]:
    x = oenv - oenv.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    if ac[0] <= 0:
        return []
    ac = ac / ac[0]
    lo, hi = int(0.25 * fps), min(int(2.0 * fps), len(ac) - 2)
    pk = sorted((i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.05), key=lambda i: -ac[i])[:8]
    out: list[float] = []
    for i in pk:
        b = _fold_octave(60.0 * fps / i)
        if all(abs(b - o) / o > 0.03 for o in out):
            out.append(b)
    return out


def _refine_bpm(oenv: np.ndarray, fps: float, g: float) -> tuple[float, float]:
    """Exact tempo near g, and the time (s) of a beat. Fold the onset envelope at many candidate periods:
    the true period piles the onsets into one phase."""
    k = np.arange(len(oenv))
    best, best_bpm = -1.0, g
    for c in g * np.linspace(0.94, 1.06, 241):
        period = fps * 60.0 / c
        h = np.bincount(((k % period) / period * 48).astype(int) % 48, weights=oenv, minlength=48)
        score = float((h ** 2).sum())
        if score > best:
            best, best_bpm = score, float(c)
    period = fps * 60.0 / best_bpm
    h = np.bincount(((k % period) / period * 96).astype(int) % 96, weights=oenv, minlength=96)
    h = np.convolve(np.r_[h[-1], h, h[0]], [1, 2, 1], "valid")
    return best_bpm, (int(np.argmax(h)) + 0.5) / 96 * period / fps


def _strong_low_hits(env: np.ndarray) -> np.ndarray:
    """Frames of the strongest kick / bass hits in a low-band onset envelope."""
    pk = np.array([f for f in range(2, len(env) - 2) if env[f] == env[f - 2: f + 3].max() and env[f] > env.mean() + env.std()])
    if len(pk) < 16:
        return np.array([], int)
    return pk[env[pk] >= np.percentile(env[pk], 60)]


def _grid_share(hits: np.ndarray, fps: float, bpm: float, t0: float) -> float:
    """Share of the hits that fall on the beat grid of this tempo."""
    period = fps * 60.0 / bpm
    off = ((hits - t0 * fps + period / 2) % period) - period / 2
    return float(np.mean(np.abs(off) / period < 0.12))


def detect_tempo(y: np.ndarray, accent: np.ndarray | None = None) -> dict | None:
    """Tempo that holds for most of the song, its beat times and the time signature (None for any part that is unclear)."""
    import librosa  # noqa: PLC0415

    if len(y) < SR * 8 or not np.any(y):
        return None
    fps = SR / HOP
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    scored = [(*_beat_support(oenv, fps, c), c) for c in _tempo_candidates(oenv, fps)]
    if not scored:
        return {"found": False, "reason": "No repeating beat found", "coverage": 0.0}
    top = max(s[0] for s in scored)
    coverage, played, g = min((s for s in scored if s[0] >= top - 0.05), key=lambda s: abs(np.log2(s[2] / 105)))
    out = {"coverage": round(coverage, 2)}
    if played < 3 or coverage < MIN_TEMPO_COVERAGE:
        return {"found": False, "reason": "The tempo changes or is not steady enough to call", **out}
    g, t0 = _refine_bpm(oenv, fps, g)
    # a steady tempo gives the same answer in every part of the song; a drifting live take does not
    parts = 3 if len(oenv) / fps >= 90 else 2
    seg = len(oenv) // parts
    local = [_refine_bpm(oenv[i * seg: (i + 1) * seg], fps, g)[0] for i in range(parts)]
    if max(local) / min(local) > 1 + MAX_TEMPO_DRIFT:
        return {"found": False, "reason": "The tempo drifts through the song", **out}
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, bpm=g, tightness=120, trim=False)
    if len(beats) < 8:
        return {"found": False, "reason": "Could not follow a steady beat", **out}
    times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # the tracker can lock onto an off-beat pulse (typical in 6/8): line it up with where the onsets actually pile up
    per = 60.0 / g
    drift = float(np.median((times - t0 + per / 2) % per - per / 2))
    if abs(drift) > 0.12 * per:
        times = times - drift
    # keep only beats inside stretches that really repeat at this tempo (an intro with no beat is dropped)
    win, hop = int(8 * fps), int(2 * fps)
    steady = np.zeros(len(oenv), bool)
    for s in range(0, max(1, len(oenv) - win + 1), hop):
        if _beat_support(oenv[s: s + win], fps, g)[0] >= 0.5:
            steady[s: s + win] = True
    times = times[[0 <= t < len(oenv) / fps and bool(steady[min(len(steady) - 1, int(t * fps))]) for t in times]]
    if len(times) < 8:
        return {"found": False, "reason": "Could not follow a steady beat", **out}
    res = {"found": True, "bpm": round(g, 1), "beats": [round(float(t), 3) for t in times], **out}
    res["meter"] = detect_meter(accent if accent is not None else y, times, per)
    return res


def detect_meter(y: np.ndarray, beat_times: np.ndarray, period: float) -> dict | None:
    """4/4, 3/4 or 6/8 from where the accents fall relative to the beats; None when it is not clearly one of them."""
    import librosa  # noqa: PLC0415

    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=250)  # kick / bass hits carry the bar
    fps = SR / HOP
    fr = np.round(np.asarray(beat_times) * fps).astype(int)
    fr = fr[(fr > 2) & (fr < len(env) - 3)]
    if len(fr) < 24 or not np.any(env):
        return None

    def at(frames) -> np.ndarray:
        return np.array([env[max(0, int(f) - 1): int(f) + 2].max() for f in frames])

    # the strongest low hits (kick, bass) must sit on the beat grid, otherwise the grid is not what the band is playing
    hits = _strong_low_hits(env)
    if not len(hits) or _grid_share(hits, fps, 60.0 / period, float(beat_times[0])) < MIN_ON_GRID:
        return None

    acc = at(fr)
    z = (acc - acc.mean()) / (acc.std() + 1e-9)
    n = len(z)

    def acf(lag: int) -> float:
        return float(np.mean(z[:-lag] * z[lag:])) if n > lag * 3 else 0.0

    a = {k: acf(k) for k in (2, 3, 4, 6, 8)}
    # ternary feel: more energy a third of the way between beats than half way
    mid = np.array([((at([f0 + round((f1 - f0) / 3)])[0] + at([f0 + round(2 * (f1 - f0) / 3)])[0]) / 2, at([f0 + round((f1 - f0) / 2)])[0])
                    for f0, f1 in zip(fr[:-1], fr[1:])])
    ternary = bool(len(mid)) and mid[:, 0].mean() > 1.3 * mid[:, 1].mean() and mid[:, 0].mean() > 0.15 * acc.mean()
    s3, s4 = (a[3] + a[6]) / 2, (a[4] + a[8]) / 2
    if ternary and a[2] > 0.15 and a[2] > max(a[3], a[4]) + 0.05:
        sig, per = "6/8", 2
    elif not ternary and s3 > 0.15 and s3 - s4 > 0.08 and a[3] > a[2] and 60.0 / period <= 150:  # faster is often 6/8 felt in one
        sig, per = "3/4", 3
    elif not ternary and s4 > 0.15 and s4 - s3 > 0.08:
        sig, per = "4/4", 4
    else:
        return None
    means = np.array([acc[k::per].mean() for k in range(per)])
    srt = np.sort(means)[::-1]
    # which beat starts the bar: only claimed when one position clearly carries more weight than the rest
    down = int(np.argmax(means)) if srt[0] >= 1.15 * srt[1] else None
    return {"sig": sig, "beats_per_bar": per, "downbeat": down}


# ---------------------------------------------------------------------------------------------------------------
def _mono22(x: np.ndarray, sr: int) -> np.ndarray:
    import librosa  # noqa: PLC0415

    m = x.mean(axis=1) if x.ndim == 2 else x
    return librosa.resample(m.astype(np.float32), orig_sr=sr, target_sr=SR, res_type="soxr_hq") if sr != SR else m.astype(np.float32)


def analyze_arrays(mix: np.ndarray, harmonic: np.ndarray | None = None, bass: np.ndarray | None = None,
                   drums: np.ndarray | None = None) -> dict:
    """All inputs mono at SR. `harmonic` is the music without drums when we have it; otherwise it is derived from `mix`."""
    import librosa  # noqa: PLC0415

    if harmonic is None:
        harmonic = librosa.effects.harmonic(mix, margin=3.0)
    res: dict = {"version": 1}
    try:
        res["key"] = detect_key(harmonic, bass)
    except Exception as e:  # noqa: BLE001  (an analysis failure must never break a project)
        res["key"] = {"found": False, "reason": f"Key analysis failed: {e}"}
    try:
        res["tempo"] = detect_tempo(mix, drums if drums is not None and np.any(drums) else None)
    except Exception as e:  # noqa: BLE001
        res["tempo"] = {"found": False, "reason": f"Tempo analysis failed: {e}"}
    return res


def analyze_project(files: dict[str, "object"], mix_path) -> dict:
    """`files` maps stem name -> wav path (the default variant of each stem). `mix_path` is the source audio."""
    import soundfile as sf  # noqa: PLC0415

    def load(p):
        x, sr = sf.read(p, always_2d=True, dtype="float32")
        return _mono22(x, sr)

    mix = load(mix_path)
    pitched = [k for k in files if k not in ("drums", "cleaned", "vocals_only")]
    harmonic = None
    if "drums" in files and pitched:
        harmonic = sum(load(files[k]) for k in pitched)
    if "cleaned" in files:  # voice clean up of speech: no music to analyse
        return {"version": 1, "key": {"found": False, "reason": "Speech clean up has no musical key"},
                "tempo": {"found": False, "reason": "Speech clean up has no beat"}}
    return analyze_arrays(mix, harmonic, load(files["bass"]) if "bass" in files else None,
                          load(files["drums"]) if "drums" in files else None)
