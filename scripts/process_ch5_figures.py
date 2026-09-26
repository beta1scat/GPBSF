#!/usr/bin/env python3
"""Process, crop, and split Chapter 5 figure PDFs for doctoral thesis integration.

Operations:
  1. Crop 5_1_system_overview.pdf: extract part (a) (full-width overview), remove redundant bottom parts (b) & (c).
  2. Split 5_4_grasp_pipeline.pdf:
     - 5_4_grasp_candidates.pdf: part (a) (cuboid, frustum, ellipsoid grasp frames).
     - 5_5_grasp_filtering.pdf: parts (b)-(e) (approach filter, gripper clearance, collision, bin side constraint).
  3. Generate PNG previews for inspection.
"""

from pathlib import Path
import fitz  # PyMuPDF

FIG_DIR = Path(r"d:\0-research\00-papers\thesis\graduate-thesis\figures\chapter5")

def process_fig5_1():
    src_pdf = FIG_DIR / "5_1_system_overview.pdf"
    doc = fitz.open(src_pdf)
    page = doc[0]

    # Part (a) is above y = 206.0
    crop_rect = fitz.Rect(0, 0, page.rect.width, 205.5)
    
    # Redact '(a) 方法概览' label so it's a unified overview figure
    # Search for text '(a)'
    rects = page.search_for("(a)")
    for r in rects:
        if r.y1 < 25.0:
            # Expand slightly to cover "方法概览" as well
            erase_rect = fitz.Rect(0, 0, r.x1 + 65.0, r.y1 + 4.0)
            page.draw_rect(erase_rect, color=(1, 1, 1), fill=(1, 1, 1))

    # Set cropbox to part (a)
    page.set_cropbox(crop_rect)
    
    out_pdf = FIG_DIR / "5_1_system_overview_cropped.pdf"
    doc.save(out_pdf)
    
    # Also render high-res PNG preview
    pix = page.get_pixmap(dpi=300)
    pix.save(str(FIG_DIR / "5_1_system_overview_preview.png"))
    print(f"Processed Fig 5.1 -> {out_pdf.name}")


def process_fig5_4():
    src_pdf = FIG_DIR / "5_4_grasp_pipeline.pdf"
    
    # 1. Extract Part (a): Grasp candidates
    doc_a = fitz.open(src_pdf)
    page_a = doc_a[0]
    crop_a = fitz.Rect(0, 0, page_a.rect.width, 111.0)
    page_a.set_cropbox(crop_a)
    # Erase '(a) ' so header becomes '按基元类型生成候选抓取位姿'
    page_a.draw_rect(fitz.Rect(0, 0, 14.5, 11.0), color=(1, 1, 1), fill=(1, 1, 1))
    out_a = FIG_DIR / "5_4_grasp_candidates.pdf"
    doc_a.save(out_a)
    pix_a = page_a.get_pixmap(dpi=300)
    pix_a.save(str(FIG_DIR / "5_4_grasp_candidates_preview.png"))
    print(f"Processed Fig 5.4 candidates -> {out_a.name}")

    # 2. Extract Part (b)-(e): Grasp filtering
    doc_b = fitz.open(src_pdf)
    page_b = doc_b[0]
    crop_b = fitz.Rect(0, 112.0, page_b.rect.width, page_b.rect.height)
    page_b.set_cropbox(crop_b)
    
    # Relabel subpanels (b)-(e) -> (a)-(d)
    mapping = [
        ('(b) ', '(a) '),
        ('(c) ', '(b) '),
        ('(d) ', '(c) '),
        ('(e) ', '(d) ')
    ]
    d = page_b.get_text('dict')
    for b in d['blocks']:
        if 'lines' in b:
            for l in b['lines']:
                for s in l['spans']:
                    for old_s, new_s in mapping:
                        if old_s in s['text']:
                            bbox = s['bbox']
                            page_b.draw_rect(fitz.Rect(bbox), color=(1, 1, 1), fill=(1, 1, 1))
                            page_b.insert_text((bbox[0], bbox[3] - 2.5), new_s, fontsize=8.5, fontname='times-bold', color=(0, 0, 0))

    out_b = FIG_DIR / "5_5_grasp_filtering.pdf"
    doc_b.save(out_b)
    pix_b = page_b.get_pixmap(dpi=300)
    pix_b.save(str(FIG_DIR / "5_5_grasp_filtering_preview.png"))
    print(f"Processed Fig 5.5 filtering -> {out_b.name}")


def process_fig5_3():
    src_pdf = FIG_DIR / "5_3_primitive_fitting.pdf"
    doc = fitz.open(src_pdf)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    pix.save(str(FIG_DIR / "5_3_primitive_fitting_preview.png"))
    print(f"Rendered Fig 5.3 preview -> 5_3_primitive_fitting_preview.png")


if __name__ == "__main__":
    process_fig5_1()
    process_fig5_4()
    process_fig5_3()
