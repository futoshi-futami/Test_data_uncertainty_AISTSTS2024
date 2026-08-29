from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from huc_core import METHODS, build_tree, evaluate, normalize_probs, run_postprocessing

SEEDS = [20260823, 20260824, 20260825, 20260826, 20260827]
NUMERIC_BASES = ["LR", "GNB", "DT", "RF", "XGB", "MLP"]

SLURP_MAP = {
    "calendar_query": "calendar", "calendar_set": "calendar", "calendar_remove": "calendar",
    "weather_query": "weather", "weather_query_forecast": "weather",
    "play_music": "play", "play_radio": "play",
    "audio_volume_up": "audio", "audio_volume_down": "audio", "audio_volume_mute": "audio",
    "iot_hue_lighton": "iot", "iot_hue_lightoff": "iot",
    "news_query": "news", "news_query_world": "news", "news_query_sports": "news",
}
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
NSL_DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm"}
NSL_PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
NSL_R2L = {"guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"}
NSL_U2R = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"}
NBAIOT_CLASSES = [
    "benign", "gafgyt_combo", "gafgyt_junk", "gafgyt_scan", "gafgyt_tcp", "gafgyt_udp",
    "mirai_ack", "mirai_scan", "mirai_syn", "mirai_udp", "mirai_udpplain",
]


@dataclass
class Bundle:
    dataset: str
    X: Any
    y: np.ndarray
    class_names: list[str]
    paths: list[list[str]]
    metadata: pd.DataFrame
    manifest: pd.DataFrame
    source: dict[str, Any]
    feature_kind: str
    bases: list[str]
    exact_nsl_split: bool = False


@dataclass
class Split:
    base: np.ndarray
    stage1: np.ndarray
    stage2: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_file(url: str, path: Path, min_bytes: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_bytes:
        return path
    last = None
    for attempt in range(1, 7):
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers={"User-Agent": "HUC-real-four/1.0"}) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                if tmp.stat().st_size < min_bytes:
                    raise RuntimeError(f"downloaded only {tmp.stat().st_size} bytes")
                tmp.replace(path)
                return path
        except Exception as e:
            last = e
            log(f"download attempt {attempt} failed: {url}: {e}")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"download failed: {url}: {last}")


def curl_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000_000:
        return path
    subprocess.run([
        "curl", "-L", "--fail", "--retry", "10", "--retry-delay", "10",
        "--connect-timeout", "30", "--speed-time", "90", "--speed-limit", "1024",
        "-o", str(path), url,
    ], check=True)
    return path


def slurp_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    text = next((str(row[k]).strip() for k in ("sentence", "text", "utterance", "transcript", "sentence_annotation") if isinstance(row.get(k), str) and str(row[k]).strip()), "")
    scenario = str(row.get("scenario") or "").strip()
    intent = str(row.get("intent") or "").strip()
    action = str(row.get("action") or "").strip()
    candidates = [intent, action, f"{scenario}_{action}" if scenario and action else ""]
    chosen = next((x for x in candidates if x in SLURP_MAP), "")
    if chosen:
        scenario = SLURP_MAP[chosen]
    return text, scenario, chosen


def load_slurp(cache: Path) -> Bundle:
    focused = "https://raw.githubusercontent.com/majishah/RACC-Framework/e4cc758f61e14e79639f0934b3fbedf2db83d067/datasets/slurp_focused_domain.jsonl"
    path = cache / "slurp_focused_domain.jsonl"
    source_mode = "RACC focused-domain file"
    try:
        get_file(focused, path, 1000)
    except Exception:
        source_mode = "official SLURP train file fallback"
        focused = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/train.jsonl"
        path = get_file(focused, cache / "slurp_train.jsonl", 100000)
    rows = []
    with path.open(encoding="utf-8") as f:
        for source_row, line in enumerate(f):
            if source_mode.startswith("RACC") and source_row >= 600:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            text, scenario, intent = slurp_fields(raw)
            if intent in SLURP_MAP:
                rows.append({"source_row": source_row, "text": text, "scenario": scenario, "intent": intent, "raw_id": raw.get("id", source_row)})
    df = pd.DataFrame(rows)
    if source_mode.startswith("RACC"):
        if len(df) != 492:
            raise RuntimeError(f"SLURP historical selection should be 492 rows, got {len(df)}")
    else:
        selected = []
        per_class = 33
        for intent in SLURP_MAP:
            part = df[df.intent == intent].sort_values("source_row").head(per_class)
            selected.append(part)
        df = pd.concat(selected, ignore_index=True).sort_values("source_row").head(492).reset_index(drop=True)
    names = list(SLURP_MAP)
    lookup = {x: i for i, x in enumerate(names)}
    y = df.intent.map(lookup).to_numpy(int)
    paths = [[SLURP_MAP[x], x] for x in names]
    metadata = pd.DataFrame({"word_count": df.text.str.split().map(len).astype(float), "char_count": df.text.str.len().astype(float)})
    return Bundle(
        "SLURP-492", df.text.tolist(), y, names, paths, metadata,
        df[["source_row", "raw_id", "scenario", "intent", "text"]].copy(),
        {"url": focused, "sha256": sha256(path), "selection": source_mode, "rows": len(df)},
        "text", ["TFIDF-LR"], False,
    )


def nsl_label(x: str) -> str:
    x = str(x).strip().rstrip(".")
    if x == "normal": return "normal"
    if x in NSL_DOS: return "dos"
    if x in NSL_PROBE: return "probe"
    if x in NSL_R2L: return "r2l"
    if x in NSL_U2R: return "u2r"
    raise KeyError(f"unmapped NSL-KDD label {x}")


def load_nsl(cache: Path) -> Bundle:
    urls = [
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B_20Percent.txt",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
    ]
    path = cache / "KDDTrain+_20Percent.txt"
    if not path.exists():
        last = None
        for url in urls:
            try:
                get_file(url, path, 100000)
                break
            except Exception as e:
                last = e
        else:
            raise RuntimeError(last)
    df = pd.read_csv(path, names=NSL_COLUMNS, header=None)
    if len(df) != 25192:
        raise RuntimeError(f"NSL-KDD row count is {len(df)}, expected 25192")
    names = ["normal", "dos", "probe", "r2l", "u2r"]
    lookup = {x: i for i, x in enumerate(names)}
    family = df.label.map(nsl_label)
    y = family.map(lookup).to_numpy(int)
    X = df.drop(columns=["label", "difficulty"])
    metadata = pd.DataFrame({
        "protocol_type": X.protocol_type.astype(str),
        "duration": pd.to_numeric(X.duration, errors="coerce").fillna(0),
        "src_bytes": pd.to_numeric(X.src_bytes, errors="coerce").fillna(0),
        "dst_bytes": pd.to_numeric(X.dst_bytes, errors="coerce").fillna(0),
    })
    paths = [["normal"], ["attack", "network", "dos"], ["attack", "network", "probe"], ["attack", "access", "r2l"], ["attack", "access", "u2r"]]
    manifest = pd.DataFrame({"source_row": np.arange(len(df)), "class": family})
    return Bundle(
        "NSL-KDD-25192", X, y, names, paths, metadata, manifest,
        {"url": urls[0], "sha256": sha256(path), "rows": len(df), "class_counts": family.value_counts().to_dict()},
        "mixed", NUMERIC_BASES.copy(), True,
    )


def configure_rarfile():
    import rarfile
    if shutil.which("unrar"):
        rarfile.UNRAR_TOOL = "unrar"
    if shutil.which("unar"):
        rarfile.UNAR_TOOL = "unar"
    if shutil.which("7z"):
        rarfile.SEVENZIP_TOOL = "7z"
    return rarfile


def rar_csv(rar_path: Path, wanted: str, features: list[str], nrows: int = 10000) -> pd.DataFrame:
    rarfile = configure_rarfile()
    with rarfile.RarFile(str(rar_path)) as rf:
        entries = [x for x in rf.infolist() if not x.isdir() and x.filename.lower().endswith(".csv")]
        found = next((x for x in entries if Path(x.filename).stem.lower() == wanted.lower()), None)
        if found is None:
            found = next((x for x in entries if Path(x.filename).stem.lower().endswith(wanted.lower())), None)
        if found is None:
            raise FileNotFoundError(f"{wanted}.csv absent from {rar_path}: {[x.filename for x in entries]}")
        with rf.open(found) as f:
            return pd.read_csv(f, nrows=nrows, usecols=features)


def load_nbaiot(cache: Path) -> Bundle:
    url = "https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip"
    archive = curl_file(url, cache / "nbaiot.zip")
    selected_root = cache / "selected_device"
    marker = selected_root / ".done"
    device = "Danmini_Doorbell"
    if not marker.exists():
        shutil.rmtree(selected_root, ignore_errors=True)
        selected_root.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if device in m]
            if not members:
                raise RuntimeError("Danmini_Doorbell is absent from UCI archive")
            for m in members:
                zf.extract(m, selected_root)
        marker.write_text("ok")
    dirs = [p for p in selected_root.rglob(device) if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(device)
    d = dirs[0]
    benign_path = next(iter(d.rglob("benign_traffic.csv")), None)
    if benign_path is None:
        raise FileNotFoundError("benign_traffic.csv")
    features = list(pd.read_csv(benign_path, nrows=0).columns[:10])
    counts = {name: 91 if i < 10 else 90 for i, name in enumerate(NBAIOT_CLASSES)}
    frames = []
    def take(frame: pd.DataFrame, label: str, source: str, seed: int):
        part = frame.sample(n=counts[label], random_state=seed).copy()
        part["__label"] = label
        part["__source"] = source
        part["__source_row"] = part.index.astype(int)
        frames.append(part)
    take(pd.read_csv(benign_path, nrows=10000, usecols=features), "benign", str(benign_path.relative_to(selected_root)), 42)
    gaf = next(iter(d.glob("*gafgyt*.rar")), None)
    mir = next(iter(d.glob("*mirai*.rar")), None)
    if gaf is None or mir is None:
        raise FileNotFoundError("N-BaIoT attack RAR files")
    for i, attack in enumerate(["combo", "junk", "scan", "tcp", "udp"]):
        take(rar_csv(gaf, attack, features), f"gafgyt_{attack}", f"{gaf.name}:{attack}.csv", 100 + i)
    for i, attack in enumerate(["ack", "scan", "syn", "udp", "udpplain"]):
        take(rar_csv(mir, attack, features), f"mirai_{attack}", f"{mir.name}:{attack}.csv", 200 + i)
    df = pd.concat(frames, ignore_index=True)
    if len(df) != 1000 or df.__label.nunique() != 11:
        raise RuntimeError(f"N-BaIoT selection has shape {df.shape} and {df.__label.nunique()} classes")
    lookup = {x: i for i, x in enumerate(NBAIOT_CLASSES)}
    y = df.__label.map(lookup).to_numpy(int)
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    paths = [["benign"]] + [["attack", name.split("_", 1)[0], name.split("_", 1)[1]] for name in NBAIOT_CLASSES[1:]]
    metadata = X.iloc[:, :4].copy()
    metadata.columns = [f"{i}:{c}" for i, c in enumerate(metadata.columns)]
    manifest = pd.concat([df[["__label", "__source", "__source_row"]], X], axis=1)
    selected_csv = cache / "selected_1000.csv"
    manifest.to_csv(selected_csv, index=False)
    return Bundle(
        "N-BaIoT-Danmini-1000", X, y, NBAIOT_CLASSES.copy(), paths, metadata, manifest,
        {"url": url, "archive_sha256": sha256(archive), "device": device, "features": features, "selected_sha256": sha256(selected_csv), "rows": len(df)},
        "numeric", NUMERIC_BASES.copy(), False,
    )


def api_get(session: requests.Session, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = "https://api.inaturalist.org/v1" + endpoint
    last = None
    for attempt in range(1, 7):
        try:
            r = session.get(url, params=params, timeout=(30, 90))
            if r.status_code == 429:
                time.sleep(10 * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"iNaturalist API failed: {url}: {last}")


def genus_taxon(session: requests.Session, name: str) -> dict[str, Any]:
    data = api_get(session, "/taxa", {"q": name, "rank": "genus", "per_page": 50, "is_active": "true"})
    exact = [x for x in data.get("results", []) if str(x.get("name", "")).lower() == name.lower()]
    if not exact:
        raise RuntimeError(f"genus {name} not found")
    return max(exact, key=lambda x: int(x.get("observations_count") or 0))


def species_candidates(session: requests.Session, genus: dict[str, Any]) -> list[dict[str, Any]]:
    data = api_get(session, "/observations/species_counts", {"taxon_id": genus["id"], "quality_grade": "research", "verifiable": "true", "per_page": 100})
    ans = []
    for row in data.get("results", []):
        t = row.get("taxon") or {}
        if t.get("rank") == "species" and t.get("is_active", True):
            ans.append(t)
    return ans


def observations(session: requests.Session, taxon_id: int, limit: int) -> list[dict[str, Any]]:
    all_rows = []
    for page in range(1, 4):
        data = api_get(session, "/observations", {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true", "verifiable": "true", "per_page": 200, "page": page, "order_by": "id", "order": "asc"})
        rows = data.get("results", [])
        all_rows.extend(rows)
        if len(all_rows) >= limit or len(rows) < 200:
            break
    return all_rows[:limit]


def medium_url(photo: dict[str, Any]) -> str:
    url = str(photo.get("url") or "")
    for size in ("square", "small", "thumb"):
        url = url.replace(f"/{size}.", "/medium.")
    return url


def download_photo(record: dict[str, Any], path: Path) -> dict[str, Any] | None:
    if path.exists() and path.stat().st_size > 1000:
        try:
            with Image.open(path) as im: im.verify()
            return record
        except Exception:
            path.unlink(missing_ok=True)
    for attempt in range(4):
        try:
            r = requests.get(record["url"], timeout=(20, 60), headers={"User-Agent": "HUC-real-four/1.0"})
            r.raise_for_status()
            path.write_bytes(r.content)
            with Image.open(path) as im: im.verify()
            return record
        except Exception:
            path.unlink(missing_ok=True)
            time.sleep(attempt + 1)
    return None


def photo_stats(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        arr = np.asarray(rgb.resize((128, 128)), dtype=np.float32) / 255.0
    gray = .2989 * arr[..., 0] + .5870 * arr[..., 1] + .1140 * arr[..., 2]
    return float(gray.mean()), float(gray.std()), float(w / max(1, h))


def resnet_features(paths: list[Path], output: Path) -> np.ndarray:
    if output.exists():
        f = np.load(output)["features"]
        if f.shape == (len(paths), 512): return f
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    class Images(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            with Image.open(paths[i]) as im: return transform(im.convert("RGB"))
    loader = DataLoader(Images(), batch_size=64, shuffle=False, num_workers=2)
    model = resnet18(weights=weights); model.fc = torch.nn.Identity(); model.eval()
    chunks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            chunks.append(model(batch).numpy().astype(np.float32))
            if i % 10 == 0: log(f"ResNet-18 batches {i + 1}/{math.ceil(len(paths) / 64)}")
    f = np.concatenate(chunks)
    np.savez_compressed(output, features=f)
    return f


def load_inaturalist(cache: Path, images_per_species: int) -> Bundle:
    root = cache / "inaturalist"; images_dir = root / "images"; images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"
    session = requests.Session(); session.headers.update({"User-Agent": "HUC-real-four/1.0"})
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        chosen = []
        for genus_name in ["Amanita", "Lithobates", "Enallagma"]:
            genus = genus_taxon(session, genus_name)
            accepted = 0
            for species in species_candidates(session, genus):
                if accepted == 10: break
                records = []
                seen = set()
                for ob in observations(session, int(species["id"]), max(240, images_per_species * 3)):
                    photos = ob.get("photos") or []
                    if not photos: continue
                    ph = photos[0]; pid = int(ph.get("id") or 0); url = medium_url(ph)
                    if pid <= 0 or pid in seen or not url: continue
                    seen.add(pid)
                    records.append({"observation_id": int(ob["id"]), "photo_id": pid, "url": url, "license": ph.get("license_code"), "species_id": int(species["id"]), "species_name": species["name"], "genus_id": int(genus["id"]), "genus_name": genus_name})
                valid = []
                for start in range(0, len(records), 40):
                    chunk = records[start:start + 40]
                    with cf.ThreadPoolExecutor(max_workers=8) as ex:
                        futures = [ex.submit(download_photo, r, images_dir / f"{r['photo_id']}.jpg") for r in chunk]
                        for fut in cf.as_completed(futures):
                            result = fut.result()
                            if result is not None: valid.append(result)
                    valid.sort(key=lambda r: (r["observation_id"], r["photo_id"]))
                    if len(valid) >= images_per_species: break
                if len(valid) < images_per_species: continue
                chosen.extend(valid[:images_per_species]); accepted += 1
                log(f"iNaturalist {genus_name}: {species['name']} {images_per_species} images")
            if accepted != 10: raise RuntimeError(f"iNaturalist {genus_name}: only {accepted}/10 species")
        manifest = pd.DataFrame(chosen).sort_values(["genus_name", "species_name", "observation_id", "photo_id"]).reset_index(drop=True)
        stats = [photo_stats(images_dir / f"{int(pid)}.jpg") for pid in manifest.photo_id]
        manifest[["brightness", "contrast", "aspect_ratio"]] = pd.DataFrame(stats)
        manifest["image_path"] = manifest.photo_id.map(lambda x: f"images/{int(x)}.jpg")
        manifest.to_csv(manifest_path, index=False)
    species_names = sorted(manifest.species_name.unique().tolist())
    if len(species_names) != 30:
        raise RuntimeError(f"iNaturalist has {len(species_names)} species")
    manifest = manifest.groupby("species_name", sort=True).head(images_per_species).sort_values(["genus_name", "species_name", "observation_id", "photo_id"]).reset_index(drop=True)
    image_paths = [root / x for x in manifest.image_path]
    X = resnet_features(image_paths, root / "resnet18_features.npz")
    lookup = {x: i for i, x in enumerate(species_names)}
    y = manifest.species_name.map(lookup).to_numpy(int)
    genus = manifest.groupby("species_name").genus_name.first().to_dict()
    paths = [[genus[x], x] for x in species_names]
    metadata = manifest[["brightness", "contrast", "aspect_ratio"]].copy()
    return Bundle(
        f"iNaturalist-real-30species-{len(manifest)}", X, y, species_names, paths, metadata, manifest,
        {"api": "https://api.inaturalist.org/v1", "genera": ["Amanita", "Lithobates", "Enallagma"], "images_per_species": images_per_species, "manifest_sha256": sha256(manifest_path), "feature_extractor": "ImageNet-1K pretrained ResNet-18, 512 dimensions", "note": "new real API subset; not the historical iNaturalist-2019 file list"},
        "numeric", NUMERIC_BASES.copy(), False,
    )


def split_exact(y: np.ndarray, seed: int, sizes: Sequence[int]) -> list[np.ndarray]:
    remaining = np.arange(len(y)); pieces = []
    for j, size in enumerate(sizes[:-1]):
        take, remaining = train_test_split(remaining, train_size=int(size), random_state=seed + 1009 * j, stratify=y[remaining])
        pieces.append(np.sort(take))
    pieces.append(np.sort(remaining))
    if [len(x) for x in pieces] != list(sizes): raise RuntimeError("split size mismatch")
    return pieces


def make_split(bundle: Bundle, seed: int) -> Split:
    n = len(bundle.y)
    if bundle.exact_nsl_split:
        sizes = [8000, 5000, 5000, 3000, 4192]
    else:
        sizes = [round(.45 * n), round(.20 * n), round(.15 * n), round(.10 * n)]
        sizes.append(n - sum(sizes))
    return Split(*split_exact(bundle.y, seed, sizes))


def groups(bundle: Bundle, split: Split) -> tuple[list[str], dict[str, np.ndarray]]:
    md = bundle.metadata.reset_index(drop=True); train = md.iloc[split.base]
    names = ["all"]; arr = [np.ones(len(md))]
    if bundle.dataset.startswith("SLURP"):
        med = float(train.word_count.median()); v = md.word_count.to_numpy(float)
        names += [f"word_count≤{med:g}", f"word_count>{med:g}"]; arr += [(v <= med).astype(float), (v > med).astype(float)]
    elif bundle.dataset.startswith("NSL"):
        p = md.protocol_type.astype(str).to_numpy()
        for value in ["tcp", "udp", "icmp"]: names.append(f"protocol={value}"); arr.append((p == value).astype(float))
        for col in ["duration", "src_bytes", "dst_bytes"]:
            med = float(train[col].median()); v = md[col].to_numpy(float); names.append(f"{col}>{med:g}"); arr.append((v > med).astype(float))
    elif bundle.dataset.startswith("iNaturalist"):
        for col in ["brightness", "contrast"]:
            med = float(train[col].median()); v = md[col].to_numpy(float); names += [f"{col}≤{med:.4f}", f"{col}>{med:.4f}"]; arr += [(v <= med).astype(float), (v > med).astype(float)]
        v = md.aspect_ratio.to_numpy(float); names += ["aspect_ratio≥1", "aspect_ratio<1"]; arr += [(v >= 1).astype(float), (v < 1).astype(float)]
    else:
        for col in md.columns[:3]:
            v = pd.to_numeric(md[col], errors="coerce").fillna(0).to_numpy(float); med = float(pd.Series(v[split.base]).median())
            names += [f"{col}≤{med:.6g}", f"{col}>{med:.6g}"]; arr += [(v <= med).astype(float), (v > med).astype(float)]
    G = np.column_stack(arr)
    return names, {name: G[getattr(split, name)] for name in ["base", "stage1", "stage2", "validation", "test"]}


def features(bundle: Bundle, split: Split) -> dict[str, Any]:
    names = ["base", "stage1", "stage2", "validation", "test"]
    if bundle.feature_kind == "text":
        return {name: [bundle.X[i] for i in getattr(split, name)] for name in names}
    if bundle.feature_kind == "mixed":
        cat = ["protocol_type", "service", "flag"]
        num = [x for x in bundle.X.columns if x not in cat]
        try: one = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError: one = OneHotEncoder(handle_unknown="ignore", sparse=False)
        pre = ColumnTransformer([("cat", one, cat), ("num", StandardScaler(), num)], sparse_threshold=0.0)
        pre.fit(bundle.X.iloc[split.base]); all_x = np.asarray(pre.transform(bundle.X), np.float32)
    else:
        all_x = np.asarray(bundle.X, np.float32); scale = StandardScaler().fit(all_x[split.base]); all_x = scale.transform(all_x).astype(np.float32)
    return {name: all_x[getattr(split, name)] for name in names}


def base_model(name: str, seed: int):
    if name == "TFIDF-LR":
        union = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)),
        ])
        return Pipeline([("features", union), ("model", LogisticRegression(max_iter=2000, C=2.0, solver="lbfgs", random_state=seed))])
    if name == "LR": return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    if name == "GNB": return GaussianNB()
    if name == "DT": return DecisionTreeClassifier(min_samples_leaf=2, random_state=seed)
    if name == "RF": return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    if name == "MLP": return MLPClassifier(hidden_layer_sizes=(96,), max_iter=300, early_stopping=True, n_iter_no_change=20, random_state=seed)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=.08, subsample=.9, colsample_bytree=.9, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", n_jobs=2, random_state=seed)
    raise KeyError(name)


def aligned(model: Any, X: Any, k: int) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = model.named_steps["model"].classes_ if isinstance(model, Pipeline) else model.classes_
    out = np.zeros((len(raw), k))
    for j, c in enumerate(classes): out[:, int(c)] = raw[:, j]
    return normalize_probs(out)


def fit_base(bundle: Bundle, split: Split, F: dict[str, Any], name: str, seed: int) -> dict[str, np.ndarray]:
    model = base_model(name, seed); model.fit(F["base"], bundle.y[split.base])
    return {part: aligned(model, F[part], len(bundle.class_names)) for part in ["stage1", "stage2", "validation", "test"]}


def summarize(df: pd.DataFrame, output: Path) -> None:
    per_base = df.groupby(["dataset", "base_model", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    per_base.to_csv(output / "results_by_base_method.csv", index=False)
    overall = df.groupby(["dataset", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    overall.to_csv(output / "results_by_method.csv", index=False)
    metrics = ["UC", "HUC", "C-UC", "C-HUC", "Accuracy", "AUC"]
    lines = [f"# {df.dataset.iloc[0]}", "", "Values are mean ± standard deviation over seeds.", ""]
    for base in df.base_model.drop_duplicates():
        lines += [f"## {base}", "", "| Method | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
        x = per_base[per_base.base_model == base]
        lookup = {(r.method, r.metric): (r["mean"], r["std"]) for _, r in x.iterrows()}
        for method in METHODS:
            if method not in set(x.method): continue
            vals = []
            for metric in metrics:
                m, s = lookup.get((method, metric), (np.nan, np.nan)); vals.append("NA" if pd.isna(m) else f"{m:.6f} ± {0 if pd.isna(s) else s:.6f}")
            lines.append("| " + method + " | " + " | ".join(vals) + " |")
        lines.append("")
    (output / "RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")


def run(bundle: Bundle, output: Path, seeds: list[int], max_iter: int, threshold: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    bundle.manifest.to_csv(output / "data_manifest.csv", index=False)
    (output / "source.json").write_text(json.dumps(bundle.source, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"class_id": range(len(bundle.class_names)), "class_name": bundle.class_names, "path": ["/".join(x) for x in bundle.paths]}).to_csv(output / "classes.csv", index=False)
    tree = build_tree(bundle.paths)
    rows, split_rows, state_rows, errors = [], [], [], []
    start = time.time()
    for seed in seeds:
        split = make_split(bundle, seed); group_names, G = groups(bundle, split); F = features(bundle, split)
        Y = {part: bundle.y[getattr(split, part)] for part in ["base", "stage1", "stage2", "validation", "test"]}
        for part in ["base", "stage1", "stage2", "validation", "test"]:
            for idx in getattr(split, part): split_rows.append({"seed": seed, "row": int(idx), "split": part, "class_id": int(bundle.y[idx])})
        for base in bundle.bases:
            log(f"{bundle.dataset}: seed={seed} Base={base}")
            try: P = fit_base(bundle, split, F, base, seed)
            except Exception as e:
                errors.append({"seed": seed, "base_model": base, "method": "BASE", "error": repr(e), "traceback": traceback.format_exc()}); continue
            for method in METHODS:
                t0 = time.time()
                try:
                    q, state = run_postprocessing(method, tree, P, Y, G, max_iter=max_iter, threshold=threshold)
                    values = evaluate(tree, q, Y["test"], G["test"])
                    selected = state.get("second", state).get("selected_round", 0) if isinstance(state, dict) else 0
                    attempted = state.get("second", state).get("attempted_rounds", 0) if isinstance(state, dict) else 0
                    state_rows.append({"dataset": bundle.dataset, "seed": seed, "base_model": base, "method": method, "state": state})
                    for metric, value in values.items():
                        rows.append({"dataset": bundle.dataset, "seed": seed, "base_model": base, "method": method, "metric": metric, "value": value, "base_n": len(split.base), "stage1_n": len(split.stage1), "stage2_n": len(split.stage2), "validation_n": len(split.validation), "test_n": len(split.test), "selected_round": selected, "attempted_rounds": attempted, "elapsed_seconds": time.time() - t0})
                    log(f"  {method}: UC={values['UC']:.5f} HUC={values['HUC']:.5f} Accuracy={values['Accuracy']:.4f}")
                except Exception as e:
                    errors.append({"seed": seed, "base_model": base, "method": method, "error": repr(e), "traceback": traceback.format_exc()})
                    log(f"  ERROR {method}: {e}")
    pd.DataFrame(rows).to_csv(output / "results_long.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_manifest.csv", index=False)
    with (output / "states.jsonl").open("w", encoding="utf-8") as f:
        for row in state_rows: f.write(json.dumps(row, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n")
    (output / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows: raise RuntimeError("no results")
    summarize(pd.DataFrame(rows), output)
    completion = {"dataset": bundle.dataset, "rows": len(bundle.y), "classes": len(bundle.class_names), "bases": bundle.bases, "seeds": seeds, "methods": METHODS, "result_rows": len(rows), "error_count": len(errors), "elapsed_seconds": time.time() - start, "source": bundle.source, "groups": group_names}
    (output / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [f"# {bundle.dataset} execution report", "", f"- Data rows: {len(bundle.y)}", f"- classes: {len(bundle.class_names)}", f"- Base: {', '.join(bundle.bases)}", f"- seeds: {len(seeds)}", f"- Postprocessing: {len(METHODS)}", f"- result rows: {len(rows)}", f"- errors: {len(errors)}", f"- elapsed seconds: {completion['elapsed_seconds']:.1f}", "", "Base, first-stage Postprocessing, second-stage Postprocessing, validation, and test data are separated."]
    (output / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if errors: raise RuntimeError(f"{len(errors)} errors remain; see errors.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["slurp", "nsl_kdd", "nbaiot", "inaturalist"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--threshold", type=float, default=5e-4)
    p.add_argument("--images-per-species", type=int, default=80)
    a = p.parse_args(); cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    if a.dataset == "slurp": bundle = load_slurp(cache)
    elif a.dataset == "nsl_kdd": bundle = load_nsl(cache)
    elif a.dataset == "nbaiot": bundle = load_nbaiot(cache)
    else: bundle = load_inaturalist(cache, a.images_per_species)
    max_iter = max(a.max_iter, 40) if a.dataset == "nsl_kdd" else a.max_iter
    run(bundle, Path(a.output_dir), a.seeds, max_iter, a.threshold)


if __name__ == "__main__":
    main()
