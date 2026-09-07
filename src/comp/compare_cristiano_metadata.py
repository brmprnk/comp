#!/usr/bin/env python3
"""
Compare sample IDs across three layers:
  1. cristiano_metadata.xlsx  (original publication)
  2. cris_metadata.csv        (local database)
  3. extracted_features/biomart_10kb  (features actually used in analyses)

Outputs a per-sample CSV summarising presence at each layer, plus a text summary.

Usage:
    python src/comp/compare_cristiano_metadata.py \
        accessory_files/cristiano_metadata.xlsx \
        accessory_files/cris_metadata.csv \
        extracted_features/biomart_10kb \
        results/cristiano_metadata_comparison.csv
"""

import csv
import glob
import os
import sys

import openpyxl


def load_xlsx_samples(xlsx_path: str) -> dict[str, dict]:
    """Return {patient_id: {patient_type, sample_type, timepoint, ...}} from Sheet 1 & 4."""
    wb = openpyxl.load_workbook(xlsx_path)

    def parse_sheet(sheet_num: int, skip_rows: int = 2) -> dict[str, dict]:
        ws = wb[str(sheet_num)]
        rows = list(ws.iter_rows(min_row=skip_rows, values_only=True))
        headers = [str(h).strip() if h else "" for h in rows[0]]
        result = {}
        for row in rows[1:]:
            if not row[0] or not isinstance(row[0], str):
                continue
            pid = row[0].strip()
            result[pid] = dict(zip(headers, row))
        return result

    sheet1 = parse_sheet(1)  # All samples (patient metadata)
    sheet4 = parse_sheet(4)  # WGS analyses

    # Merge: sheet1 has richer patient metadata, sheet4 confirms WGS was done
    merged = {}
    all_ids = sheet1.keys() | sheet4.keys()
    for pid in all_ids:
        info = sheet1.get(pid, sheet4.get(pid, {}))
        merged[pid] = {
            "patient_type":  info.get("Patient Type", ""),
            "sample_type":   info.get("Sample Type", ""),
            "timepoint":     info.get("Timepoint", ""),
            "stage":         info.get("Stage", ""),
            "in_xlsx_sheet1": pid in sheet1,
            "in_xlsx_wgs":    pid in sheet4,
        }
    return merged


def load_cris_metadata(csv_path: str) -> dict[str, dict]:
    """Return {sample_name (CG*): {ee_id, disease, ...}}."""
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cg = row["sample_name"].strip()
            result[cg] = {
                "ee_id":    row["ID"].strip(),
                "disease":  row["disease"].strip(),
                "fragNum":  row["fragNum"].strip(),
                "readlen":  row["readlen"].strip(),
            }
    return result


def load_biomart_ids(biomart_dir: str) -> set[str]:
    """Return set of EE IDs that have a _BIN.csv in biomart_10kb."""
    ids = set()
    for path in glob.glob(os.path.join(biomart_dir, "*_BIN.csv")):
        basename = os.path.basename(path)
        # strip .hg38.frag.tsv_BIN.csv or _BIN.csv
        for suffix in (".hg38.frag.tsv_BIN.csv", "_BIN.csv"):
            if basename.endswith(suffix):
                ids.add(basename[: -len(suffix)])
                break
    return ids


def main() -> None:
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <xlsx> <cris_metadata_csv> <biomart_dir> <output_csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    xlsx_path, meta_csv, biomart_dir, output_csv = sys.argv[1:]

    xlsx_samples = load_xlsx_samples(xlsx_path)
    cris_meta    = load_cris_metadata(meta_csv)
    biomart_ids  = load_biomart_ids(biomart_dir)

    # Build reverse maps
    ee_to_cg = {v["ee_id"]: k for k, v in cris_meta.items()}
    cg_to_ee = {k: v["ee_id"] for k, v in cris_meta.items()}

    # Union of all CG IDs seen anywhere
    all_cg_ids = xlsx_samples.keys() | cris_meta.keys()

    rows = []
    for cg in sorted(all_cg_ids):
        xlsx_info = xlsx_samples.get(cg, {})
        meta_info = cris_meta.get(cg)
        ee_id     = meta_info["ee_id"] if meta_info else ""
        in_biomart = ee_id in biomart_ids if ee_id else False

        rows.append({
            "sample_name":     cg,
            "ee_id":           ee_id,
            "in_xlsx_sheet1":  xlsx_info.get("in_xlsx_sheet1", False),
            "in_xlsx_wgs":     xlsx_info.get("in_xlsx_wgs", False),
            "in_cris_metadata": meta_info is not None,
            "in_biomart_10kb": in_biomart,
            "patient_type":    xlsx_info.get("patient_type", meta_info["disease"] if meta_info else ""),
            "timepoint":       xlsx_info.get("timepoint", ""),
            "stage":           xlsx_info.get("stage", ""),
            "disease":         meta_info["disease"] if meta_info else "",
            "frag_count":      meta_info["fragNum"] if meta_info else "",
        })

    # Also add EE IDs in biomart that have no CG counterpart (other datasets)
    known_ee = {r["ee_id"] for r in rows if r["ee_id"]}
    extra_ee = biomart_ids - known_ee
    for ee in sorted(extra_ee):
        rows.append({
            "sample_name":      "",
            "ee_id":            ee,
            "in_xlsx_sheet1":   False,
            "in_xlsx_wgs":      False,
            "in_cris_metadata": False,
            "in_biomart_10kb":  True,
            "patient_type":     "other_dataset",
            "timepoint":        "",
            "stage":            "",
            "disease":          "",
            "frag_count":       "",
        })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_name", "ee_id",
        "in_xlsx_sheet1", "in_xlsx_wgs", "in_cris_metadata", "in_biomart_10kb",
        "patient_type", "timepoint", "stage", "disease", "frag_count",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    cris_rows = [r for r in rows if r["patient_type"] != "other_dataset"]
    other_rows = [r for r in rows if r["patient_type"] == "other_dataset"]

    def count(rows, **filters):
        result = rows
        for k, v in filters.items():
            result = [r for r in result if r[k] == v]
        return len(result)

    print("=" * 60)
    print("Cristiano dataset — sample retention summary")
    print("=" * 60)
    print(f"{'Layer':<40} {'Count':>6}")
    print("-" * 48)
    print(f"{'xlsx sheet 1 (all patients)':<40} {count(cris_rows, in_xlsx_sheet1=True):>6}")
    print(f"{'xlsx sheet 4 (WGS analyses)':<40} {count(cris_rows, in_xlsx_wgs=True):>6}")
    print(f"{'cris_metadata.csv':<40} {count(cris_rows, in_cris_metadata=True):>6}")
    print(f"{'biomart_10kb (features extracted)':<40} {count(cris_rows, in_biomart_10kb=True):>6}")
    print()

    # Breakdown of missing at each step
    in_xlsx_not_meta = [r for r in cris_rows if r["in_xlsx_wgs"] and not r["in_cris_metadata"]]
    in_meta_not_biomart = [r for r in cris_rows if r["in_cris_metadata"] and not r["in_biomart_10kb"]]
    in_meta_not_xlsx = [r for r in cris_rows if r["in_cris_metadata"] and not r["in_xlsx_wgs"]]

    print(f"In xlsx WGS but NOT in cris_metadata ({len(in_xlsx_not_meta)}):")
    for r in sorted(in_xlsx_not_meta, key=lambda x: x["sample_name"]):
        print(f"  {r['sample_name']:<20} {r['patient_type']:<25} {r['timepoint']}")

    print(f"\nIn cris_metadata but NOT in xlsx WGS ({len(in_meta_not_xlsx)}):")
    rep     = [r for r in in_meta_not_xlsx if "_Rep" in r["sample_name"]]
    non_rep = [r for r in in_meta_not_xlsx if "_Rep" not in r["sample_name"]]
    print(f"  _Rep replicates : {len(rep)}")
    print(f"  Longitudinal / other timepoints: {len(non_rep)}")
    for r in sorted(non_rep, key=lambda x: x["sample_name"])[:15]:
        print(f"    {r['sample_name']:<25} {r['disease']}")
    if len(non_rep) > 15:
        print(f"    ... and {len(non_rep) - 15} more")

    print(f"\nIn cris_metadata but NOT extracted to biomart_10kb ({len(in_meta_not_biomart)}):")
    for r in sorted(in_meta_not_biomart, key=lambda x: x["sample_name"]):
        print(f"  {r['sample_name']:<20} {r['ee_id']:<12} {r['disease']}")

    print(f"\nOther datasets in biomart_10kb (not Cristiano): {len(other_rows)}")
    print(f"\nFull per-sample table written to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
