#!/usr/bin/env python3
"""Per-instruction (SASS) view of an ncu report, for the questions the source-line view cannot answer:
which sync instructions execute how often, where warps arrive at a barrier split, where the stall samples really land.

Uses `ncu --import <rep> --page source --print-source sass --csv` (needs a report that contains the SourceCounters
section: `--set full` on ncu 2026.x, or `--section SourceCounters`). Prints
  * the top-N PCs by warp stall samples,
  * every branch with Divergent Branches > 0 (average threads executed tells how it splits),
  * execution counts and average active threads on BAR.SYNC / BRA.DIV / WARPSYNC / COLLECTIVE / MEMBAR / ATOM / RED / CCTL,
  * with --diff, the instruction-by-instruction difference in executions and samples against a second report of the
    same binary (addresses are matched by offset from the first instruction).

Example (found the cooperative-groups barrier slow path described in reference/06-diagnosis-playbook.md, Pattern I.1):
    python3 sass_source_counters.py full_cg.ncu-rep --diff full_hand.ncu-rep --region 0x4650-0x48d0
"""
import argparse
import csv
import io
import shutil
import subprocess
import sys


def load(rep, kernel_index=0):
    ncu = shutil.which("ncu") or "ncu"
    out = subprocess.run([ncu, "--import", rep, "--page", "source", "--print-source", "sass", "--csv"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"ncu --import failed for {rep}: {out.stderr.strip()[:400]}")
    rows = list(csv.reader(io.StringIO(out.stdout)))
    # one header row per kernel: "Kernel Name" then the column header line containing "Address"
    headers = [i for i, r in enumerate(rows) if r and any(c.strip() == "Address" for c in r)]
    if not headers:
        sys.exit("no per-instruction table in this report (collect SourceCounters / --set full)")
    if kernel_index >= len(headers):
        sys.exit(f"report has {len(headers)} kernels, index {kernel_index} out of range")
    h = headers[kernel_index]
    end = headers[kernel_index + 1] - 1 if kernel_index + 1 < len(headers) else len(rows)
    hdr = rows[h]
    idx = {c.strip(): i for i, c in enumerate(hdr)}
    recs = []
    for r in rows[h + 1:end]:
        if len(r) < len(hdr) or not r[idx["Address"]].startswith("0x"):
            continue
        recs.append(r)
    base = min(int(r[idx["Address"]], 16) for r in recs)

    def num(r, col):
        try:
            return float(r[idx[col]].replace(",", "") or 0)
        except (KeyError, ValueError):
            return 0.0

    table = []
    for r in recs:
        table.append({
            "off": int(r[idx["Address"]], 16) - base,
            "src": r[idx["Source"]].strip(),
            "exec": num(r, "Instructions Executed"),
            "thr": num(r, "Avg. Threads Executed"),
            "div": num(r, "Divergent Branches"),
            "stall": num(r, "Warp Stall Sampling (All Samples)"),
            "stall_ni": num(r, "Warp Stall Sampling (Not-issued Samples)"),
        })
    return table


def fmt_row(t):
    return f"{t['off']:#07x}  exec={t['exec']:>10.0f}  thr={t['thr']:>4.1f}  div={t['div']:>6.0f}  stall={t['stall']:>6.0f}  {t['src'][:80]}"


def parse_region(s):
    lo, hi = s.split("-")
    return int(lo, 16), int(hi, 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--diff", help="second report of the same binary to compare against")
    ap.add_argument("--kernel-index", type=int, default=0, help="which kernel in the report (default 0)")
    ap.add_argument("--top", type=int, default=15, help="how many stall PCs to list")
    ap.add_argument("--region", type=parse_region, help="offset range to dump completely, e.g. 0x4650-0x48d0")
    ap.add_argument("--out", help="write to this file instead of stdout")
    a = ap.parse_args()

    sink = open(a.out, "w") if a.out else sys.stdout

    def p(*args):
        print(*args, file=sink)

    t = load(a.report, a.kernel_index)
    total_exec = sum(x["exec"] for x in t)
    total_stall = sum(x["stall"] for x in t)
    p(f"# {a.report}: {len(t)} instructions, {total_exec:.0f} warp-instructions executed, {total_stall:.0f} stall samples")

    p(f"\n## top {a.top} PCs by stall samples (a barrier's samples land on the instruction after its BAR.SYNC)")
    for x in sorted(t, key=lambda x: -x["stall"])[:a.top]:
        if x["stall"] > 0:
            p(f"{x['stall']:>7.0f} {100 * x['stall'] / max(total_stall, 1):5.1f}%  {x['off']:#07x}  {x['src'][:80]}")

    p("\n## divergent branches (Divergent Branches > 0)")
    for x in t:
        if x["div"] > 0:
            p(fmt_row(x))

    p("\n## synchronization / ordering instructions")
    keys = ("BAR.SYNC", "BRA.DIV", "WARPSYNC", "COLLECTIVE", "MEMBAR", "ATOM", "RED.", "CCTL", "NANOSLEEP", "ERRBAR")
    for x in t:
        if any(k in x["src"] for k in keys) and (x["exec"] > 0 or x["stall"] > 0):
            p(fmt_row(x))
    split = [x for x in t if "BRA.DIV" in x["src"] and x["exec"] > 0 and x["thr"] < 31.5]
    if split:
        p("\n!! BRA.DIV executed with fewer than 32 threads on average: warps reach a non-aligned barrier.sync split and take the")
        p("   WARPSYNC.COLLECTIVE fallback. See reference/06-diagnosis-playbook.md, Pattern I.1.")

    if a.region:
        lo, hi = a.region
        p(f"\n## region {lo:#x}-{hi:#x}")
        for x in t:
            if lo <= x["off"] <= hi:
                p(fmt_row(x))

    if a.diff:
        u = load(a.diff, a.kernel_index)
        by = {x["off"]: x for x in u}
        p(f"\n## diff against {a.diff} (exec / stall: first - second), instructions with any difference")
        p(f"   totals: exec {total_exec:.0f} vs {sum(x['exec'] for x in u):.0f}, stall {total_stall:.0f} vs {sum(x['stall'] for x in u):.0f}")
        for x in t:
            y = by.get(x["off"])
            if y is None:
                p(f"{x['off']:#07x}  only in first   {x['src'][:70]}")
                continue
            de, ds = x["exec"] - y["exec"], x["stall"] - y["stall"]
            if de or ds:
                p(f"{x['off']:#07x}  dexec={de:>+10.0f}  dstall={ds:>+7.0f}  {x['src'][:70]}")
    if a.out:
        sink.close()


if __name__ == "__main__":
    main()
