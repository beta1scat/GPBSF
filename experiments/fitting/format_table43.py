"""Format Table 4.3 (V4 Camera-conditioned Ground-Truth Oracle Fitting Evaluation)

Parses runs/fitting/v4_robust_oracle_test.summary.json and generates:
1. Aligned terminal ASCII table
2. Ready-to-paste LaTeX tabular block for Table 4.3 (tab:v4_fitting_oracle) in chapter4.tex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_row(name_cn: str, name_en: str, data: dict, is_overall: bool = False) -> tuple[str, str]:
    success_rate = data.get("fit_success_rate", 1.0) * 100.0

    c_med = (data.get("center_error_m", {}).get("median") or 0.0) * 1000.0
    c_p95 = (data.get("center_error_m", {}).get("p95") or 0.0) * 1000.0

    s_med = (data.get("surface_chamfer_m", {}).get("median") or 0.0) * 1000.0
    s_p95 = (data.get("surface_chamfer_m", {}).get("p95") or 0.0) * 1000.0

    o_med = (data.get("observed_to_fitted_m", {}).get("median") or 0.0) * 1000.0
    o_p95 = (data.get("observed_to_fitted_m", {}).get("p95") or 0.0) * 1000.0

    r_med = data.get("runtime_ms", {}).get("median") or 0.0
    r_p95 = data.get("runtime_ms", {}).get("p95") or 0.0

    label_ascii = f"{name_cn} ({name_en})" if not is_overall else f"{name_cn}"
    ascii_line = (
        f"{label_ascii:<18} | {success_rate:6.2f}% | "
        f"{c_med:5.2f} / {c_p95:5.2f} | "
        f"{s_med:5.2f} / {s_p95:5.2f} | "
        f"{o_med:5.2f} / {o_p95:5.2f} | "
        f"{r_med:5.2f} / {r_p95:6.2f}"
    )

    if not is_overall:
        latex_line = (
            f"\t\t\t{name_cn} ({name_en})   & {success_rate:.2f} & "
            f"{c_med:.2f} / {c_p95:.2f} & "
            f"{s_med:.2f} / {s_p95:.2f} & "
            f"{o_med:.2f} / {o_p95:.2f} & "
            f"{r_med:.2f} / {r_p95:.2f} \\\\"
        )
    else:
        latex_line = (
            f"\t\t\t\\textbf{{{name_cn}}} & \\textbf{{{success_rate:.2f}}} & "
            f"\\textbf{{{c_med:.2f} / {c_p95:.2f}}} & "
            f"\\textbf{{{s_med:.2f} / {s_p95:.2f}}} & "
            f"\\textbf{{{o_med:.2f} / {o_p95:.2f}}} & "
            f"\\textbf{{{r_med:.2f} / {r_p95:.2f}}} \\\\"
        )

    return ascii_line, latex_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="runs/fitting/v4_robust_oracle_test.summary.json",
        help="Path to evaluation summary JSON file",
    )
    parser.add_argument(
        "--csv",
        default="runs/fitting/v4_robust_oracle_test.csv",
        help="Optional path to per-sample CSV file",
    )
    args = parser.parse_args()

    sum_path = Path(args.summary).resolve()
    if not sum_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {sum_path}. Please run evaluate.py first.")

    with sum_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_prim = data.get("by_primitive", {})
    overall = data.get("overall", {})

    print("\n" + "=" * 88)
    print(" Table 4.3: V4 Camera-Conditioned Ground-Truth Oracle Fitting Quantitative Evaluation")
    print("=" * 88)
    header = (
        f"{'Primitive':<18} | {'Return':<7} | {'Center Err (mm)':<15} | "
        f"{'Recon Err (mm)':<15} | {'90% Trim Res(mm)':<16} | {'Runtime (ms)':<15}"
    )
    print(header)
    print("-" * 88)

    prims = [
        ("长方体", "Cuboid", "cuboid"),
        ("圆锥台", "Frustum", "frustum"),
        ("椭球体", "Ellipsoid", "ellipsoid"),
    ]

    latex_rows = []
    for cn, en, key in prims:
        row_data = by_prim.get(key, {})
        asc, ltx = format_row(cn, en, row_data, is_overall=False)
        print(asc)
        latex_rows.append(ltx)

    print("-" * 88)
    asc_ov, ltx_ov = format_row("全基元综合统计", "Overall", overall, is_overall=True)
    print(asc_ov)
    print("=" * 88)

    print("\n--- [LaTeX Code for Table 4.3 (tab:v4_fitting_oracle) in chapters/chapter4.tex] ---")
    latex_output = f"""\\begin{{table}}[htbp]
	\\centering
	\\caption[V4相机条件化测试集上的真值路由拟合定量结果]{{V4相机条件化测试集上的真值路由拟合定量评测结果}}
	\\label{{tab:v4_fitting_oracle}}
	\\resizebox{{\\textwidth}}{{!}}{{
		\\begin{{tabular}}{{lccccc}}
			\\toprule
			\\multirow{{2}}{{*}}{{\\shortstack{{几何基元\\\\类型}}}} & \\multirow{{2}}{{*}}{{\\shortstack{{参数返回率\\\\(\\%)}}}} & \\multicolumn{{4}}{{c}}{{各误差与求解耗时指标（中位数 / 第95百分位数）}} \\\\
			\\cmidrule(lr){{3-6}}
			& & \\shortstack{{中心位置误差\\\\(mm)}} & \\shortstack{{全曲面真值\\\\重建误差 (mm)}} & \\shortstack{{90\\%截断拟合\\\\残差 (mm)}} & \\shortstack{{求解耗时\\\\(ms)}} \\\\
			\\midrule
{latex_rows[0]}
{latex_rows[1]}
{latex_rows[2]}
			\\midrule
{ltx_ov}
			\\bottomrule
		\\end{{tabular}}
	}}
\\end{{table}}"""
    print(latex_output)
    print("-" * 88 + "\n")


if __name__ == "__main__":
    main()
