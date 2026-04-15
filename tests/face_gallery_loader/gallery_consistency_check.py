"""
Gallery Consistency Check — Internal sanity-check tool.

Validates that embeddings within each person folder of the real face gallery
are mutually similar and flags suspicious outliers using cosine similarity.

This script is READ-ONLY with respect to gallery contents.
No embeddings are modified, moved, or rewritten.

Usage (from project root):
    python tests/face_gallery_loader/gallery_consistency_check.py
    python tests/face_gallery_loader/gallery_consistency_check.py \\
        --gallery-root "C:\\...\\data\\generated_face_gallery_real" \\
        --similarity-threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuspiciousEmbedding:
    file_name: str
    average_similarity: float
    threshold: float


@dataclass
class PersonFolderResult:
    person_id: str
    embedding_count: int
    status: str                                 # passed | warning | single_embedding_only | error
    min_pairwise_similarity: Optional[float]
    max_pairwise_similarity: Optional[float]
    mean_pairwise_similarity: Optional[float]
    suspicious_embeddings: list[SuspiciousEmbedding] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)


@dataclass
class GallerySummary:
    gallery_path: str
    total_person_folders: int
    folders_checked: int
    folders_passed: int
    folders_with_warnings: int
    folders_with_errors: int
    folders_single_embedding: int
    total_suspicious_embeddings: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedding(path: Path) -> Optional[np.ndarray]:
    """Load a single .npy embedding file.

    Returns a 1-D float32 numpy array, or None if the file cannot be loaded
    or is not a valid 1-D float array.
    """
    try:
        arr = np.load(path)
    except Exception as exc:
        print(f"  [WARN] Cannot load {path.name}: {exc}", file=sys.stderr)
        return None

    if arr.ndim != 1:
        print(
            f"  [WARN] {path.name} is not a 1-D array (shape={arr.shape}); skipping.",
            file=sys.stderr,
        )
        return None

    if not np.issubdtype(arr.dtype, np.floating):
        # Attempt a safe cast.
        try:
            arr = arr.astype(np.float32)
        except Exception:
            print(
                f"  [WARN] {path.name} cannot be cast to float; skipping.",
                file=sys.stderr,
            )
            return None

    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Both vectors are L2-normalised defensively before the dot product.
    Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a / norm_a, b / norm_b))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_person_folder(folder_path: Path, threshold: float) -> PersonFolderResult:
    """Analyse all .npy embeddings in a single person folder.

    Returns a PersonFolderResult with pairwise cosine similarity statistics
    and a list of suspicious (outlier) embeddings.
    """
    npy_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".npy")

    embeddings: list[np.ndarray] = []
    file_names: list[str] = []
    load_errors: list[str] = []

    for npy_path in npy_files:
        emb = load_embedding(npy_path)
        if emb is not None:
            embeddings.append(emb)
            file_names.append(npy_path.name)
        else:
            load_errors.append(npy_path.name)

    n = len(embeddings)

    # --- zero valid embeddings ------------------------------------------------
    if n == 0:
        return PersonFolderResult(
            person_id=folder_path.name,
            embedding_count=0,
            status="error",
            min_pairwise_similarity=None,
            max_pairwise_similarity=None,
            mean_pairwise_similarity=None,
            load_errors=load_errors,
        )

    # --- single embedding -----------------------------------------------------
    if n == 1:
        return PersonFolderResult(
            person_id=folder_path.name,
            embedding_count=1,
            status="single_embedding_only",
            min_pairwise_similarity=None,
            max_pairwise_similarity=None,
            mean_pairwise_similarity=None,
            load_errors=load_errors,
        )

    # --- 2+ embeddings: pairwise cosine similarity matrix ---------------------
    # Build NxN matrix.
    sim_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    # Upper triangle (excluding diagonal) for global stats.
    upper = sim_matrix[np.triu_indices(n, k=1)]
    min_sim = float(np.min(upper))
    max_sim = float(np.max(upper))
    mean_sim = float(np.mean(upper))

    # Per-embedding average similarity to *all others* (exclude diagonal).
    suspicious: list[SuspiciousEmbedding] = []
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        avg = float(np.mean(others))
        if avg < threshold:
            suspicious.append(
                SuspiciousEmbedding(
                    file_name=file_names[i],
                    average_similarity=round(avg, 6),
                    threshold=threshold,
                )
            )

    status = "warning" if suspicious else "passed"

    return PersonFolderResult(
        person_id=folder_path.name,
        embedding_count=n,
        status=status,
        min_pairwise_similarity=round(min_sim, 6),
        max_pairwise_similarity=round(max_sim, 6),
        mean_pairwise_similarity=round(mean_sim, 6),
        suspicious_embeddings=suspicious,
        load_errors=load_errors,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(gallery_path: Path, results: list[PersonFolderResult]) -> GallerySummary:
    """Aggregate per-folder results into a gallery-wide summary."""
    folders_passed = sum(1 for r in results if r.status == "passed")
    folders_warnings = sum(1 for r in results if r.status == "warning")
    folders_errors = sum(1 for r in results if r.status == "error")
    folders_single = sum(1 for r in results if r.status == "single_embedding_only")
    folders_checked = folders_passed + folders_warnings  # had 2+ valid embeddings
    total_suspicious = sum(len(r.suspicious_embeddings) for r in results)

    return GallerySummary(
        gallery_path=str(gallery_path),
        total_person_folders=len(results),
        folders_checked=folders_checked,
        folders_passed=folders_passed,
        folders_with_warnings=folders_warnings,
        folders_with_errors=folders_errors,
        folders_single_embedding=folders_single,
        total_suspicious_embeddings=total_suspicious,
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_report(results: list[PersonFolderResult], summary: GallerySummary) -> None:
    """Print a human-readable consistency report to stdout."""
    print("\n===== GALLERY CONSISTENCY CHECK =====\n")
    print(f"Gallery path : {summary.gallery_path}")
    print(f"Threshold    : (see per-folder suspicion detail)\n")

    for r in results:
        print(f"----- {r.person_id} -----")
        print(f"  Embeddings : {r.embedding_count}")

        if r.min_pairwise_similarity is not None:
            print(f"  Min sim    : {r.min_pairwise_similarity:.4f}")
            print(f"  Max sim    : {r.max_pairwise_similarity:.4f}")
            print(f"  Mean sim   : {r.mean_pairwise_similarity:.4f}")
        else:
            print("  Pairwise   : N/A")

        print(f"  Status     : {r.status}")

        if r.load_errors:
            print(f"  Load errors ({len(r.load_errors)}):")
            for err in r.load_errors:
                print(f"    - {err}")

        if r.suspicious_embeddings:
            print(f"  Suspicious embeddings ({len(r.suspicious_embeddings)}):")
            for sus in r.suspicious_embeddings:
                print(
                    f"    - {sus.file_name} | avg_sim={sus.average_similarity:.4f}"
                    f" | BELOW threshold {sus.threshold}"
                )
        print()

    print("===== SUMMARY =====\n")
    print(f"Gallery path             : {summary.gallery_path}")
    print(f"Total person folders     : {summary.total_person_folders}")
    print(f"Folders checked (2+ emb) : {summary.folders_checked}")
    print(f"Folders passed           : {summary.folders_passed}")
    print(f"Folders with warnings    : {summary.folders_with_warnings}")
    print(f"Folders with errors      : {summary.folders_with_errors}")
    print(f"Folders single embedding : {summary.folders_single_embedding}")
    print(f"Total suspicious embeds  : {summary.total_suspicious_embeddings}")
    print("\n=====================================\n")


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def write_json_report(
    results: list[PersonFolderResult],
    summary: GallerySummary,
    gallery_root: Path,
) -> Path:
    """Serialize results and summary to gallery_root/gallery_consistency_report.json."""
    report = {
        "summary": asdict(summary),
        "persons": [asdict(r) for r in results],
    }

    output_path = gallery_root / "gallery_consistency_report.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_GALLERY_ROOT = (
    r"C:\Users\talif\Desktop\Camera-regogintion\data\generated_face_gallery_real"
)
DEFAULT_THRESHOLD = 0.35


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gallery consistency check — validate internal embedding similarity."
    )
    parser.add_argument(
        "--gallery-root",
        default=DEFAULT_GALLERY_ROOT,
        help="Path to the face gallery root directory.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum average cosine similarity; embeddings below this are flagged (default: 0.35).",
    )
    args = parser.parse_args()

    gallery_root = Path(args.gallery_root)
    threshold: float = args.similarity_threshold

    if not gallery_root.exists() or not gallery_root.is_dir():
        print(
            f"[ERROR] Gallery root does not exist or is not a directory:\n  {gallery_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect only sub-directories (person folders), sorted deterministically.
    person_dirs = sorted(
        p for p in gallery_root.iterdir() if p.is_dir()
    )

    if not person_dirs:
        print("[ERROR] No person folders found in gallery root.", file=sys.stderr)
        sys.exit(1)

    results: list[PersonFolderResult] = []
    for person_dir in person_dirs:
        result = analyze_person_folder(person_dir, threshold)
        results.append(result)

    summary = build_summary(gallery_root, results)
    print_report(results, summary)

    json_path = write_json_report(results, summary, gallery_root)
    print(f"JSON report written to:\n  {json_path}\n")


if __name__ == "__main__":
    main()
