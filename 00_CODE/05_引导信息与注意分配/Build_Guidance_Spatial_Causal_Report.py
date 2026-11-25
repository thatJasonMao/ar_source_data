"""Build a UTF-8 HTML companion and a compact machine-readable manifest."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DOC = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"
TABLE = OUT / "tables"
DATA = OUT / "data"
FIG = OUT / "figures"
DOC.mkdir(parents=True, exist_ok=True)


def inline(text: str) -> str:
    value = html.escape(text)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    return value


def markdown_to_html(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_list = False
    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_table and (not stripped or "|" not in stripped):
            out.append("</tbody></table>")
            in_table = False
        if stripped.startswith("#"):
            if in_list:
                out.append("</ul>")
                in_list = False
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
        elif stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
                continue
            if not in_table:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append("<table><tbody>")
                tag = "th"
                in_table = True
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
        elif stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


def main() -> None:
    report_md = (DOC / "完整中文工作报告.md").read_text(encoding="utf-8")
    body = markdown_to_html(report_md)
    figures = sorted(FIG.glob("*.png"))
    gallery = "\n".join(
        f'<figure><a href="../Results_Guidance_Spatial_Causal_Analysis/figures/{f.name}"><img loading="lazy" src="../Results_Guidance_Spatial_Causal_Analysis/figures/{f.name}" alt="{html.escape(f.stem)}"></a><figcaption>{html.escape(f.stem)}</figcaption></figure>'
        for f in figures
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>五种疏散引导的空间—视觉—HRV—行为机制分析</title>
<style>
:root{{--ink:#18212b;--muted:#5e6b78;--line:#dbe3ea;--accent:#116a73;--paper:#fff;--bg:#f3f6f8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.72 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1100px;margin:30px auto;padding:44px 58px;background:var(--paper);box-shadow:0 8px 30px #1e344516}}
h1{{font-size:2rem;border-bottom:3px solid var(--accent);padding-bottom:12px}} h2{{margin-top:2.2rem;color:var(--accent)}} h3{{margin-top:1.6rem}}
p,li{{max-width:90ch}} code{{background:#eef3f5;padding:.1em .35em;border-radius:4px}} a{{color:#075e8c}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;overflow:auto}} th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left}} th{{background:#eaf3f3}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin-top:24px}} figure{{margin:0;border:1px solid var(--line);padding:10px;background:#fff}} img{{width:100%;height:auto}} figcaption{{color:var(--muted);font-size:.85rem;margin-top:6px}}
@media(max-width:720px){{main{{margin:0;padding:24px 18px}}}}
</style></head><body><main>{body}<h2>结果图集</h2><div class="gallery">{gallery}</div></main></body></html>"""
    (DOC / "report.html").write_text(document, encoding="utf-8")

    hrv = pd.read_csv(DATA / "hrv_trial_features.csv")
    run = json.loads((DATA / "analysis_run_summary.json").read_text(encoding="utf-8"))
    extension = json.loads((DATA / "multimodal_extension_summary.json").read_text(encoding="utf-8"))
    spatiotemporal = json.loads((DATA / "spatiotemporal_extension_summary.json").read_text(encoding="utf-8"))
    revision = json.loads((DATA / "requested_revision_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "title": "五种疏散引导的空间—视觉—HRV—行为机制分析",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "primary_trials": run["primary_trial_n"],
        "primary_participants": run["primary_participant_n"],
        "hrv_usable_trials": int(pd.to_numeric(hrv["HRVPrimaryUsable"], errors="coerce").sum()),
        "serial_complete_trials": extension["serial_complete_trial_n"],
        "rest_baseline_usable_participants": spatiotemporal["rest_usable_participant_n"],
        "spatiotemporal_zone_time_rows": spatiotemporal["zone_time_row_n"],
        "functional_zone_types": spatiotemporal["functional_zone_n"],
        "spatial_mediation_complete_trials": spatiotemporal["mediation_complete_trial_n"],
        "time_axis": extension["all_dynamic_axes"],
        "primary_visual_factors": extension["primary_visual_factors"],
        "primary_hrv_factors": ["HRVStressProxy"],
        "primary_spatial_factors": extension["primary_spatial_factors"],
        "incremental_scope": spatiotemporal["scope"],
        "latest_assumption": revision["assumption"],
        "latest_ebm_exclusions": revision["ebm_exclusions"],
        "tiered_mediation_joint_tests": revision["mediation_joint_test_n"],
        "tiered_mediation_joint_fdr_significant": revision["mediation_joint_fdr_significant_n"],
        "reports": [
            "../Documentation_Guidance_Spatial_Causal_Analysis/完整中文工作报告.md",
            "../Documentation_Guidance_Spatial_Causal_Analysis/report.html",
        ],
        "figures": [f"figures/{f.name}" for f in figures],
    }
    (OUT / "artifact.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
