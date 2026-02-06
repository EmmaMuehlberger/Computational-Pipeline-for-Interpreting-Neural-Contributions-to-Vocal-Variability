#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py, numpy as np
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import os
from typing import Any, Dict, List, Tuple
from datasets import Dataset
import os, pickle, traceback, tempfile, shutil
from pathlib import Path
from typing import Tuple, List, Dict
from multiprocessing import Pool, set_start_method, freeze_support

# --------------- CONFIG -----------------
ROOT_DIR = # Add file path here
OUT_DIR  = # Add file path here
N_PROCS  = 6
# ----------------------------------------

import h5py, numpy as np

def _read_h5_string(f: h5py.File, ref_or_value) -> str:
    if isinstance(ref_or_value, h5py.Reference):
        dset = f[ref_or_value]
        data = dset[()]
    else:
        data = ref_or_value

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    if isinstance(data, str):
        return data

    arr = np.asarray(data)

    if arr.shape == ():
        if arr.dtype.kind in ("S", "U"):
            return str(arr.astype(str))
        if np.issubdtype(arr.dtype, np.integer):
            try:
                return chr(int(arr))
            except Exception:
                return str(arr)
        return str(arr)

    if arr.dtype.kind in ("S", "U"):
        if arr.ndim == 1:
            return "".join([s.decode("utf-8", "ignore") if isinstance(s, (bytes, np.bytes_)) else str(s)
                            for s in arr.tolist()])
        flat = arr.ravel().tolist()
        return "".join([s.decode("utf-8", "ignore") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in flat])

    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(x)) for x in arr.ravel())
    return str(arr)

def reconstruct_din_matrix(din_group: h5py.Group) -> sp.csc_matrix:
    num_rows = int(din_group.attrs.get('MATLAB_sparse', 4))

    if 'data' in din_group and 'ir' in din_group:
        # --- Normal MATLAB sparse ---
        data = din_group['data'][:]
        ir = din_group['ir'][:]
        jc = din_group['jc'][:]
        # MATLAB uses 1-based row indices
        if ir.size > 0 and ir.min() == 1:
            ir = ir - 1
        num_cols = len(jc) - 1
        return sp.csc_matrix((data, ir, jc), shape=(num_rows, num_cols), dtype=bool)

    elif 'jc' in din_group:
        # --- Special case: all-zero sparse matrix (no data/ir) ---
        jc = din_group['jc'][:]
        num_cols = len(jc) - 1
        return sp.csc_matrix((num_rows, num_cols), dtype=bool)

    else:
        # --- Missing din structure entirely ---
        return sp.csc_matrix((num_rows, 0), dtype=bool)

def filter_spikes_in_window(
    all_neuron_data: List[Dict],
    start_sample: int,
    end_sample: int
) -> Dict[str, List]:
    segment_spikes = []

    for neuron in all_neuron_data:
        spike_times = neuron['spike_times']

        # Inclusive start, exclusive end to avoid having end_idx in multiple segments
        start_idx = np.searchsorted(spike_times, start_sample, side='left')
        end_idx   = np.searchsorted(spike_times, end_sample,   side='left')

        for i in range(start_idx, end_idx):
            segment_spikes.append({
                "time_in_segment": int(spike_times[i]) - int(start_sample),
                "neuron_ksid": int(neuron['ksid']),
                "channel":     int(neuron['channel']),
                "quality":     int(neuron['quality']),
            })

    if not segment_spikes:
        return {"time_in_segment": [], "neuron_ksid": [], "channel": [], "quality": []}

    # Globally sort by relative time to guarantee chronological order
    segment_spikes.sort(key=lambda x: x['time_in_segment'])

    return {
        "time_in_segment": [s['time_in_segment'] for s in segment_spikes],
        "neuron_ksid":     [s['neuron_ksid']     for s in segment_spikes],
        "channel":         [s['channel']         for s in segment_spikes],
        "quality":         [s['quality']         for s in segment_spikes],
    }

def convert_single_file(mat_filepath: str, csv_filepath: str, add_silence: bool = True) -> List[Dict[str, Any]]:
    print(f"--- Processing File ---")
    print(f"MAT file: {mat_filepath}")
    print(f"CSV file: {csv_filepath}")

    all_segments: List[Dict[str, Any]] = []

    # ============= Read HDF5 payloads =============
    with h5py.File(mat_filepath, "r") as f:
        # Metadata
        bird_name       = _read_h5_string(f, f["birdname"][()]).strip()
        source_filename = _read_h5_string(f, f["fname"][()]).strip()

        # session can be an array of references or inline strings
        session_id = []
        sess_raw = f["session"][()]
        for ref in np.asarray(sess_raw).ravel():
            # entries might be shape (1,) holding a ref/value; unwrap if so
            item = ref[0] if hasattr(ref, "__len__") and len(np.shape(ref)) == 1 else ref
            session_id.append(_read_h5_string(f, item).strip())

        dph = int(np.asarray(f["dph"])[0][0])

        # -------- Sampling rate --------
        fs_raw = np.asarray(f["fs"])[0][0] if np.asarray(f["fs"]).ndim >= 2 else np.asarray(f["fs"])[()]
        sampling_rate_float = float(fs_raw)
        sampling_rate_int   = int(round(sampling_rate_float))

        # -------- Audio (1-D) --------
        audio_waveform = np.asarray(f["mic"][()]).reshape(-1).astype(np.float32)
        total_samples  = int(audio_waveform.shape[0])

        # -------- DIN: CSC -> dense (4, T) bool --------
        din_sparse  = reconstruct_din_matrix(f["din"])      # returns sp.csc_matrix
        din_matrix  = din_sparse.toarray()                   # (rows x cols)
        if din_matrix.shape[0] != 4 and din_matrix.shape[1] == 4:
            din_matrix = din_matrix.T
        if din_matrix.shape[0] != 4:
            raise ValueError(f"DIN matrix expected 4 rows, got {din_matrix.shape}")
        din_matrix = (din_matrix != 0)                      # bool

        # -------- Spikes --------
        all_neuron_data: List[Dict[str, Any]] = []
        spk_ksids        = np.asarray(f["spk_ksid"][:]).ravel()
        spk_channels     = np.asarray(f["spk_channel"][:]).ravel()
        spk_quality_refs = np.asarray(f["spk_quality"][:])
        spk_times_refs   = np.asarray(f["spk_times"][:])

        quality_map = {"good": 0, "mua": 1}

        n_units = len(spk_ksids)
        for i in range(n_units):
            # quality (string) can be ref or inline
            qref = spk_quality_refs[i, 0] if spk_quality_refs.ndim == 2 else spk_quality_refs[i]
            qstr = _read_h5_string(f, qref).strip().lower()
            quality_val = int(quality_map.get(qstr, -1))

            # times can be a reference to a dataset or inline numeric array
            tref = spk_times_refs[i, 0] if spk_times_refs.ndim == 2 else spk_times_refs[i]
            if isinstance(tref, h5py.Reference):
                times_arr = np.asarray(f[tref][()])
            else:
                times_arr = np.asarray(tref)
            times = np.ravel(times_arr).astype(np.int64)
            times.sort(kind="mergesort")

            all_neuron_data.append({
                "ksid":    int(spk_ksids[i]),
                "channel": int(spk_channels[i]),
                "quality": quality_val,
                "spike_times": times,
            })

    # ============= Read CSV and sort =============
    # Expect columns: onset (samples), duration (samples), cluster_id (optional)
    voc_df = (pd.read_csv(csv_filepath, sep=None, engine="python")
                .sort_values(["onset", "duration"], kind="mergesort")
                .reset_index(drop=True))

    # ============= Sweep timeline with overlap support =============
    current_sample = 0  # rightmost boundary covered when inserting silence

    def _din_slice(start: int, end: int):
        """Return flat 1-D bool arrays for each DIN channel in [start, end)."""
        ds = din_matrix[:, start:end]
        return (
            np.asarray(ds[0], dtype=bool).ravel(),
            np.asarray(ds[1], dtype=bool).ravel(),
            np.asarray(ds[2], dtype=bool).ravel(),
            np.asarray(ds[3], dtype=bool).ravel(),
        )

    for _, row in voc_df.iterrows():
        onset = int(row["onset"])
        duration = int(row["duration"])
        if duration <= 0:
            continue

        # Clamp to audio bounds
        start = max(0, onset)
        end   = min(total_samples, onset + duration)
        if end <= start:
            continue  # degenerate after clamping

        # ---- A) Optional silence BEFORE this vocalization (only when no overlap) ----
        if add_silence and start > current_sample:
            s0, s1 = current_sample, start
            if s1 > s0:
                # slice din for silence window
                isDark_s, isDir_s, isPb_s, isPerc_s = _din_slice(s0, s1)
                spikes_sil = filter_spikes_in_window(all_neuron_data, s0, s1)
                all_segments.append({
                    "source_file": source_filename,
                    "bird_name": bird_name,
                    "date_post_hatch": dph,
                    "sampling_rate": sampling_rate_float,
                    "session_id": session_id,
                    "segment_type": "silence",
                    "onset_sample": s0,
                    "duration_samples": s1 - s0,
                    "cluster_id": -1,
                    "audio": {"array": audio_waveform[s0:s1], "sampling_rate": sampling_rate_int},
                    "isDark":       isDark_s,
                    "isDirected":   isDir_s,
                    "isPlayback":   isPb_s,
                    "isPerceptron": isPerc_s,
                    "spikes_in_segment": spikes_sil,
                })

        # ---- B) Vocalization row (always emit, overlaps allowed) ----
        isDark_v, isDir_v, isPb_v, isPerc_v = _din_slice(start, end)
        spikes_voc = filter_spikes_in_window(all_neuron_data, start, end)
        cluster_id = -1
        if "cluster_id" in row and not pd.isna(row["cluster_id"]):
            try:
                cluster_id = int(row["cluster_id"])
            except Exception:
                pass

        all_segments.append({
            "source_file": source_filename,
            "bird_name": bird_name,
            "date_post_hatch": dph,
            "sampling_rate": sampling_rate_float,
            "session_id": session_id,
            "segment_type": "vocalization",
            "onset_sample": start,
            "duration_samples": end - start,
            "cluster_id": cluster_id,
            "audio": {"array": audio_waveform[start:end], "sampling_rate": sampling_rate_int},
            "isDark":       isDark_v,
            "isDirected":   isDir_v,
            "isPlayback":   isPb_v,
            "isPerceptron": isPerc_v,
            "spikes_in_segment": spikes_voc,
        })

        # Overlap-safe advancement
        current_sample = max(current_sample, end)

    # ---- C) Trailing silence after the last covered point ----
    if add_silence and current_sample < total_samples:
        s0, s1 = current_sample, total_samples
        if s1 > s0:
            isDark_t, isDir_t, isPb_t, isPerc_t = _din_slice(s0, s1)
            din_slice_tail = din_matrix[:, s0:s1]
            spikes_tail = filter_spikes_in_window(all_neuron_data, s0, s1)
            all_segments.append({
                "source_file": source_filename,
                "bird_name": bird_name,
                "date_post_hatch": dph,
                "sampling_rate": sampling_rate_float,
                "session_id": session_id,
                "segment_type": "silence",
                "onset_sample": s0,
                "duration_samples": s1 - s0,
                "cluster_id": -1,
                "audio": {"array": audio_waveform[s0:s1], "sampling_rate": sampling_rate_int},
                "isDark":       isDark_t,
                "isDirected":   isDir_t,
                "isPlayback":   isPb_t,
                "isPerceptron": isPerc_t,
                "spikes_in_segment": spikes_tail,
            })

    print(f"Successfully generated {len(all_segments)} segments.")
    return all_segments

def _ensure_convert_single_file_available():
    if 'convert_single_file' not in globals():
        raise NameError(
            "convert_single_file is not defined. Import it at the top or paste its definition into this file."
        )

# ---------- Pair discovery ----------
def _strip_multi_suffixes(name: str, suffixes: List[str]) -> str:
    lower = name.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lower.endswith(suf):
                name = name[: -len(suf)]
                lower = lower[: -len(suf)]
                changed = True
    return name

_SUFFIXES_MAT   = [".nidq.mat", ".mat", ".nidq"]
_SUFFIXES_CSV_TAIL = [".nidq.bin", ".bin", ".nidq", ".wav", ".nidq.wav"]

def _is_mat(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nidq.mat") or n.endswith(".mat") or n.endswith(".nidq")

def _normalize_mat_key(p: Path) -> str:
    return _strip_multi_suffixes(p.name, _SUFFIXES_MAT)

def _csv_key(p: Path) -> str:
    if p.suffix.lower() != ".csv":
        return ""
    base = p.stem
    if base.lower().startswith("annotations_"):
        base = base[len("annotations_"):]
    base = _strip_multi_suffixes(base, _SUFFIXES_CSV_TAIL)
    return base

def find_all_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        mats_by_key: Dict[str, Path] = {}
        csvs_by_key: Dict[str, Path] = {}

        for p in folder.iterdir():
            if not p.is_file():
                continue
            if _is_mat(p):
                key = _normalize_mat_key(p)
                mats_by_key[key] = p
            elif p.suffix.lower() == ".csv":
                key = _csv_key(p)
                if key:
                    csvs_by_key[key] = p

        keys = sorted(set(mats_by_key) & set(csvs_by_key))
        for k in keys:
            pairs.append((mats_by_key[k], csvs_by_key[k]))

        for k in sorted(set(mats_by_key) - set(csvs_by_key)):
            print(f"[WARN] No CSV for MAT: {mats_by_key[k]}")
        for k in sorted(set(csvs_by_key) - set(mats_by_key)):
            print(f"[WARN] No MAT for CSV: {csvs_by_key[k]}")

    return pairs

# --------- Environment caps & HDF5 lock tweaks ---------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ------------- Safe atomic write + progress markers -------------
def _safe_write_pickle_atomic(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False) as tf:
        tmp_name = tf.name
        pickle.dump(obj, tf, protocol=4)
    shutil.move(tmp_name, out_path)

def _process_pair_task(args: Tuple[str, str, str]) -> None:
    mat_path_s, csv_path_s, out_dir_s = args
    mat_path = Path(mat_path_s)
    csv_path = Path(csv_path_s)
    out_dir  = Path(out_dir_s)
    stem = f"{mat_path.stem}__{csv_path.stem}"

    out_pkl      = out_dir / (stem + ".pkl")
    err_txt      = out_dir / (stem + ".err.txt")

    try:
        _ensure_convert_single_file_available()
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = convert_single_file(str(mat_path), str(csv_path))

        _safe_write_pickle_atomic(rows, out_pkl)
        if err_txt.exists():
            err_txt.unlink(missing_ok=True)

        print(f"[OK] {mat_path.name} + {csv_path.name} -> {len(rows)} rows -> {out_pkl.name}", flush=True)

    except Exception:
        tb = traceback.format_exc()
        err_txt.write_text(tb, encoding="utf-8")
        print(f"[ERROR] {mat_path.name} -> {err_txt.name}", flush=True)

# ------------- Optional: assemble later into HF -------------
def assemble_pickles_to_hf(out_dir: str, features=None):
    outp = Path(out_dir)
    rows_all = []
    for pkl in sorted(outp.glob("*.pkl")):
        with pkl.open("rb") as fh:
            rows = pickle.load(fh)
        rows_all.extend(rows)
    if features is None:
        ds = Dataset.from_list(rows_all)
    else:
        ds = Dataset.from_list(rows_all, features=features)
    return ds

# -------------------- CLI entrypoint --------------------
def main():
    root = Path(ROOT_DIR)
    out_dir = Path(OUT_DIR)
    if not root.exists():
        raise FileNotFoundError(root)

    pairs = find_all_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No MAT/CSV pairs found under {root}")

    print(f"Found {len(pairs)} pair(s). Writing to: {out_dir}", flush=True)

    # quick sanity: run ONE pair serially to ensure convert_single_file works
    test_mat, test_csv = pairs[0]
    print(f"[sanity] Running one pair serially: {test_mat.name}", flush=True)
    _process_pair_task((str(test_mat), str(test_csv), str(out_dir)))

    tasks = [(str(m), str(c), str(out_dir)) for (m, c) in pairs]

    with Pool(processes=N_PROCS, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(_process_pair_task, tasks, chunksize=1):
            pass

    print("Done.", flush=True)

if __name__ == "__main__":
    freeze_support()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()

