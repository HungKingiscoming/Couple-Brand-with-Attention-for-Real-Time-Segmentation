"""
debug_weights.py — Inspect checkpoint structure để debug weight loading.

Cách dùng:
    python debug_weights.py --ckpt path/to/checkpoint.pth
    python debug_weights.py --ckpt path/to/checkpoint.pth --filter decode_head
    python debug_weights.py --student student.pth --teacher teacher.pth
    python debug_weights.py --student student.pth --teacher teacher.pth --compare_head
"""

import argparse
import torch
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def load_state(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model', 'net', 'network'):
            if key in ckpt:
                print(f"  [checkpoint key used: '{key}']")
                return ckpt[key]
    return ckpt


def group_by_prefix(state: dict, depth: int = 2) -> dict:
    """Group keys by prefix up to `depth` dots."""
    groups = defaultdict(list)
    for k in state:
        parts = k.split('.')
        prefix = '.'.join(parts[:depth])
        groups[prefix].append(k)
    return dict(groups)


def shape_str(t) -> str:
    if isinstance(t, torch.Tensor):
        return str(list(t.shape))
    return str(type(t))


def print_sep(title='', char='=', width=70):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char*side} {title} {char*side}")
    else:
        print(char * width)


# ============================================================
# Single checkpoint inspection
# ============================================================

def inspect_single(path: str, filter_str: str = None, max_keys: int = 200):
    print_sep(f"CHECKPOINT: {Path(path).name}")
    state = load_state(path)

    keys = list(state.keys())
    total = len(keys)
    print(f"Total keys: {total}")

    if filter_str:
        keys = [k for k in keys if filter_str in k]
        print(f"Filtered to {len(keys)} keys (filter='{filter_str}')")

    # Group summary
    print_sep("KEY GROUP SUMMARY (depth=2)", char='-')
    groups = group_by_prefix({k: state[k] for k in keys}, depth=2)
    for prefix, ks in sorted(groups.items()):
        shapes = [shape_str(state[k]) for k in ks[:3]]
        extra = f" ... +{len(ks)-3} more" if len(ks) > 3 else ""
        print(f"  {prefix:<55} [{len(ks):>4} keys]  e.g. {shapes[0]}{extra}")

    # Full key list (up to max_keys)
    print_sep(f"FULL KEY LIST (showing up to {max_keys})", char='-')
    shown = keys[:max_keys]
    for k in shown:
        print(f"  {k:<75}  {shape_str(state[k])}")
    if len(keys) > max_keys:
        print(f"  ... ({len(keys) - max_keys} more keys, use --filter to narrow down)")


# ============================================================
# Head structure comparison
# ============================================================

def compare_head(student_path: str, teacher_path: str):
    print_sep("HEAD STRUCTURE COMPARISON")

    s_state = load_state(student_path)
    t_state = load_state(teacher_path)

    s_head = {k: v for k, v in s_state.items() if 'decode_head' in k or 'head' in k}
    t_head = {k: v for k, v in t_state.items() if 'decode_head' in k or 'head' in k}

    print(f"\nStudent head keys ({len(s_head)}):")
    for k, v in sorted(s_head.items()):
        print(f"  {k:<75}  {shape_str(v)}")

    print(f"\nTeacher head keys ({len(t_head)}):")
    for k, v in sorted(t_head.items()):
        print(f"  {k:<75}  {shape_str(v)}")

    # Tìm keys chung và keys khác
    s_suffixes = {}
    for k in s_head:
        suffix = re.sub(r'^(decode_head\.|model\.|backbone\.)', '', k)
        s_suffixes[suffix] = k

    t_suffixes = {}
    for k in t_head:
        suffix = re.sub(r'^(decode_head\.|model\.|backbone\.)', '', k)
        t_suffixes[suffix] = k

    common = set(s_suffixes) & set(t_suffixes)
    only_s = set(s_suffixes) - set(t_suffixes)
    only_t = set(t_suffixes) - set(s_suffixes)

    print(f"\nCommon suffixes ({len(common)}) — có thể load:")
    for s in sorted(common):
        sv = s_head[s_suffixes[s]]
        tv = t_head[t_suffixes[s]]
        match = "✓ shape match" if shape_str(sv) == shape_str(tv) else f"✗ SHAPE MISMATCH  student={shape_str(sv)}  teacher={shape_str(tv)}"
        print(f"  {s:<60}  {match}")

    print(f"\nOnly in STUDENT ({len(only_s)}) — sẽ bị random init:")
    for s in sorted(only_s):
        print(f"  {s:<60}  {shape_str(s_head[s_suffixes[s]])}")

    print(f"\nOnly in TEACHER ({len(only_t)}) — sẽ bị bỏ qua:")
    for s in sorted(only_t):
        print(f"  {s:<60}  {shape_str(t_head[t_suffixes[s]])}")


# ============================================================
# Backbone structure comparison
# ============================================================

def compare_backbone(student_path: str, teacher_path: str):
    print_sep("BACKBONE STRUCTURE COMPARISON")

    s_state = load_state(student_path)
    t_state = load_state(teacher_path)

    # Lấy backbone keys (loại trừ head)
    s_bb = {k: v for k, v in s_state.items()
            if 'decode_head' not in k and 'head' not in k}
    t_bb = {k: v for k, v in t_state.items()
            if 'decode_head' not in k and 'head' not in k}

    # Normalize: bỏ prefix backbone./model./module.
    def normalize(key):
        for pref in ('backbone.', 'model.', 'module.'):
            if key.startswith(pref):
                return key[len(pref):]
        return key

    s_norm = {normalize(k): (k, v) for k, v in s_bb.items()}
    t_norm = {normalize(k): (k, v) for k, v in t_bb.items()}

    common     = set(s_norm) & set(t_norm)
    shape_ok   = [(k, s_norm[k][1], t_norm[k][1]) for k in common
                  if shape_str(s_norm[k][1]) == shape_str(t_norm[k][1])]
    shape_bad  = [(k, s_norm[k][1], t_norm[k][1]) for k in common
                  if shape_str(s_norm[k][1]) != shape_str(t_norm[k][1])]
    only_s     = set(s_norm) - set(t_norm)
    only_t     = set(t_norm) - set(s_norm)

    print(f"\nStudent backbone keys:  {len(s_norm)}")
    print(f"Teacher backbone keys:  {len(t_norm)}")
    print(f"Common + shape OK:      {len(shape_ok)}  ← sẽ được load")
    print(f"Common + shape MISMATCH:{len(shape_bad)}  ← sẽ bị skip")
    print(f"Only in student:        {len(only_s)}  ← random init")
    print(f"Only in teacher:        {len(only_t)}  ← bị bỏ qua")

    if shape_bad:
        print(f"\nShape mismatches:")
        for k, sv, tv in sorted(shape_bad)[:20]:
            print(f"  {k:<65}  student={shape_str(sv)}  teacher={shape_str(tv)}")

    if only_s:
        print(f"\nStudent-only keys (first 30):")
        for k in sorted(only_s)[:30]:
            print(f"  {k:<65}  {shape_str(s_norm[k][1])}")

    if only_t:
        print(f"\nTeacher-only keys (first 30) — kiểm tra xem có phải stem.X không:")
        for k in sorted(only_t)[:30]:
            print(f"  {k:<65}  {shape_str(t_norm[k][1])}")


# ============================================================
# Stem key analysis — tìm đúng mapping cho _remap_stem_key
# ============================================================

def analyze_stem(path: str):
    print_sep("STEM KEY ANALYSIS")
    state = load_state(path)

    stem_keys = [k for k in state if 'stem' in k.lower()]
    if not stem_keys:
        # Thử tìm theo index pattern (stem.0, stem.1, ...)
        stem_keys = [k for k in state
                     if re.match(r'(backbone\.|model\.|module\.)?stem\.\d+', k)]

    if not stem_keys:
        print("Không tìm thấy stem keys. In tất cả keys bắt đầu từ đầu model:")
        all_keys = sorted(state.keys())
        for k in all_keys[:40]:
            print(f"  {k:<75}  {shape_str(state[k])}")
        return

    print(f"Tìm thấy {len(stem_keys)} stem keys:")
    for k in sorted(stem_keys):
        print(f"  {k:<75}  {shape_str(state[k])}")

    # Phát hiện N2 (số blocks trong stage 2)
    stage2_indices = set()
    for k in stem_keys:
        m = re.search(r'stem\.(\d+)', k)
        if m:
            stage2_indices.add(int(m.group(1)))
    if stage2_indices:
        max_idx = max(stage2_indices)
        print(f"\nMax stem block index: {max_idx}")
        print(f"Gợi ý N2 (num_blocks_per_stage[0]): {max_idx - 1} "
              f"(nếu stem.0=conv1, stem.1=conv2, stem.2..N+1=stage2)")


# ============================================================
# Quick load simulation — giả lập load để thấy % trước khi train
# ============================================================

def simulate_load(student_path: str, teacher_path: str = None):
    """Giả lập load_pretrained_gcnet và in kết quả chi tiết."""
    print_sep("LOAD SIMULATION")

    s_state = load_state(student_path)
    if teacher_path:
        t_state = load_state(teacher_path)
        states  = [('STUDENT ckpt → model', s_state),
                   ('TEACHER ckpt → teacher model', t_state)]
    else:
        states = [('STUDENT ckpt → model', s_state)]

    for label, state in states:
        print(f"\n--- {label} ---")

        bb_keys = {k: v for k, v in state.items()
                   if 'decode_head' not in k}
        hd_keys = {k: v for k, v in state.items()
                   if 'decode_head' in k}

        print(f"  Backbone keys: {len(bb_keys)}")
        print(f"  Head keys:     {len(hd_keys)}")

        # Head key naming patterns
        head_patterns = defaultdict(int)
        for k in hd_keys:
            suffix = k[len('decode_head.'):]
            top = suffix.split('.')[0]
            head_patterns[top] += 1

        print(f"  Head top-level modules:")
        for mod, cnt in sorted(head_patterns.items()):
            sample_keys = [k for k in hd_keys if f'decode_head.{mod}.' in k][:2]
            print(f"    {mod:<30} ({cnt:>3} keys)  e.g. {[k.split('decode_head.')[1] for k in sample_keys]}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Debug checkpoint weight structure")
    parser.add_argument('--ckpt',         type=str, default=None,
                        help='Single checkpoint to inspect')
    parser.add_argument('--student',      type=str, default=None,
                        help='Student checkpoint path')
    parser.add_argument('--teacher',      type=str, default=None,
                        help='Teacher checkpoint path')
    parser.add_argument('--filter',       type=str, default=None,
                        help='Filter keys containing this string')
    parser.add_argument('--max_keys',     type=int, default=200,
                        help='Max keys to print in full list (default: 200)')
    parser.add_argument('--compare_head',     action='store_true',
                        help='So sánh head structure giữa student và teacher')
    parser.add_argument('--compare_backbone', action='store_true',
                        help='So sánh backbone structure giữa student và teacher')
    parser.add_argument('--analyze_stem',     action='store_true',
                        help='Phân tích stem keys để debug _remap_stem_key')
    parser.add_argument('--simulate_load',    action='store_true',
                        help='Giả lập load và in % coverage')
    parser.add_argument('--all',              action='store_true',
                        help='Chạy tất cả analyses')
    args = parser.parse_args()

    # Single checkpoint
    if args.ckpt:
        inspect_single(args.ckpt, filter_str=args.filter, max_keys=args.max_keys)
        if args.analyze_stem or args.all:
            analyze_stem(args.ckpt)
        return

    # Cần ít nhất student
    if not args.student:
        parser.print_help()
        print("\nVí dụ:")
        print("  python debug_weights.py --ckpt weights/gcnet-s.pth")
        print("  python debug_weights.py --ckpt weights/gcnet-s.pth --filter decode_head")
        print("  python debug_weights.py --student gcnet-s.pth --teacher gcnet-l.pth --all")
        return

    # Student only
    inspect_single(args.student, filter_str=args.filter, max_keys=args.max_keys)

    if args.analyze_stem or args.all:
        analyze_stem(args.student)

    # Cần cả hai để so sánh
    if args.teacher:
        if args.compare_head or args.all:
            compare_head(args.student, args.teacher)
        if args.compare_backbone or args.all:
            compare_backbone(args.student, args.teacher)
        if args.simulate_load or args.all:
            simulate_load(args.student, args.teacher)
    elif args.simulate_load or args.all:
        simulate_load(args.student)


if __name__ == '__main__':
    main()
