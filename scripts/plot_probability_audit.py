#!/usr/bin/env python3
"""Export dependency-free SVG calibration and selective risk/coverage charts.

Input: scripts.probability_audit generated CSVs. No model calls.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path

WIDTH,HEIGHT=720,510
LEFT,TOP,PLOT=86,75,365


def fmt(v):
    return f'{v:.1f}'


def svg_head(title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(title)}">',
            '<rect width="720" height="510" fill="white"/>',
            f'<text x="50" y="34" font-size="21" font-family="sans-serif" fill="#111827">{escape(title)}</text>',
            f'<text x="50" y="56" font-size="13" font-family="sans-serif" fill="#596273">{escape(subtitle)}</text>',
            f'<rect x="{LEFT}" y="{TOP}" width="{PLOT}" height="{PLOT}" fill="#f8fafc" stroke="#cbd5e1"/>']


def pos(p):
    return LEFT+PLOT*p,TOP+PLOT*(1-p)


def axes_text(xlab,ylab):
    lines=[]
    for x in [0,.25,.5,.75,1]:
        xx,yy=pos(x)
        lines.append(f'<line x1="{fmt(xx)}" x2="{fmt(xx)}" y1="{TOP}" y2="{TOP+PLOT}" stroke="#e5e7eb"/>')
        lines.append(f'<line x1="{LEFT}" x2="{LEFT+PLOT}" y1="{fmt(yy)}" y2="{fmt(yy)}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{fmt(xx)}" y="{TOP+PLOT+22}" text-anchor="middle" font-size="13" font-family="sans-serif">{x:.2f}</text>')
        lines.append(f'<text x="{LEFT-15}" y="{fmt(yy+4)}" text-anchor="end" font-size="13" font-family="sans-serif">{x:.2f}</text>')
    lines.append(f'<text x="{LEFT+PLOT/2}" y="{HEIGHT-17}" text-anchor="middle" font-size="14" font-family="sans-serif">{escape(xlab)}</text>')
    lines.append(f'<text x="25" y="{TOP+PLOT/2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 25 {TOP+PLOT/2})">{escape(ylab)}</text>')
    return lines


def curve_for_reliability(rows,backend,field):
    title=f'Calibration: {backend} — {field}'
    out=svg_head(title,'P(selected label) vs observed correct rate; 10 bins, test split')
    out+=axes_text('Mean selected probability','Observed accuracy')
    x0,y0=pos(0);x1,y1=pos(1)
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="6 5"/>')
    points=[]
    for r in sorted(rows,key=lambda r:float(r['mean_predicted_probability'])):
        p=float(r['mean_predicted_probability']);y=float(r['observed_correct_rate']);n=int(r['n'])
        x,ypos=pos(p)[0],pos(y)[1]
        points.append(f'{x:.1f},{ypos:.1f}')
        out.append(f'<circle cx="{x:.1f}" cy="{ypos:.1f}" r="{min(11,max(3,3+n**0.5/3)):.1f}" fill="#2563eb" fill-opacity=".75"><title>n={n}; predicted={p:.3f}; observed={y:.3f}</title></circle>')
    if len(points)>=2:
        out.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1d4ed8" stroke-width="2"/>')
    out.append(f'<text x="485" y="120" font-family="sans-serif" font-size="14">Valid bins: {len(rows)}</text>')
    out.append('<text x="485" y="145" font-family="sans-serif" font-size="12" fill="#475569">Dashed = ideal calibration</text>')
    out.append('</svg>')
    return '\n'.join(out)


def curve_for_risk(rows,backend):
    title=f'Selective risk / coverage: {backend}'
    out=svg_head(title,'Threshold sweep on test split (no threshold tuning on test)')
    out+=axes_text('Coverage (all planned cases)','Error rate among accepted cases')
    good=[]
    for r in rows:
        if r['empirical_risk'] in ('',None):continue
        good.append((float(r['coverage']),float(r['empirical_risk']),float(r['threshold']),int(r['accepted'])))
    good.sort()
    if len(good)>=2:
        p=' '.join(f'{pos(x)[0]:.1f},{pos(y)[1]:.1f}' for x,y,_,_ in good)
        out.append(f'<polyline points="{p}" stroke="#0f766e" stroke-width="2.5" fill="none"/>')
    for x,y,t,n in good:
        xx,yy=pos(x)[0],pos(y)[1]
        out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.2" fill="#0f766e"><title>tau={t:.3f}; coverage={x:.3f}; risk={y:.3f}; accepted={n}</title></circle>')
    out.append('<text x="485" y="120" font-family="sans-serif" font-size="12" fill="#475569">Empirical risk, not a</text>')
    out.append('<text x="485" y="139" font-family="sans-serif" font-size="12" fill="#475569">production safety guarantee</text>')
    out.append('</svg>')
    return '\n'.join(out)


def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as fp:
        return list(csv.DictReader(fp))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report_dir',type=Path)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    root=args.report_dir
    reliability=read_csv(root/'reliability.csv')
    risk=read_csv(root/'risk_curve.csv')
    out=args.out or root/'charts';out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in reliability:
        groups[(row['backend'],row['field'])].append(row)
    for (be,field),rows in groups.items():
        (out/f'reliability_{be}_{field}.svg').write_text(curve_for_reliability(rows,be,field),'utf-8')
    grouped=defaultdict(list)
    for row in risk:grouped[row['backend']].append(row)
    for be,rows in grouped.items():
        (out/f'risk_coverage_{be}.svg').write_text(curve_for_risk(rows,be),'utf-8')
    print(f'Wrote {len(list(out.glob("*.svg")))} SVG plots to {out}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
