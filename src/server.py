"""image-colors — report an image's colour composition, canvas excluded.

Built for design work: learn which colours an image uses, and enough geometry
about each one to tie it back to an element.

Every decision here was measured rather than assumed. The ones worth knowing:

* **Exact enumeration, never clustering.** Every distinct 24-bit value is
  enumerated and counted. On a vector diagram (984 distinct colours) the real
  design palette *is* the top of that list; k-means shifted all eight real
  colours (worst dE 0.148) and median cut wasted half its slots on anti-aliasing
  greys. Nothing here averages, so every hex is byte-exact.
* **Candidate selection is geometry-driven.** Colours are walked in descending
  frequency and kept until `max_candidates` have *passed* the geometry test --
  not "take the top 48, then filter". On black-on-white artwork the AA ramp
  between the two contains ~250 distinct greys appearing thousands of times
  each, so a top-48 cut fills with greys and the real colours are never
  examined.
* **Two anti-aliasing filters, because one is not enough.** A geometry test
  (a colour must form a blob of at least `min_blob_px`) catches AA as small
  fragments around text, but *not* the long AA bands along a large shape's
  edge -- those easily exceed the threshold. So a second test rejects any colour
  explainable as a linear blend of two more prevalent colours. Measured
  separation on the test diagram: AA colours <= 0.57, real colours >= 10.54.
* **floodFill, not a whole-image mask.** Byte-identical answer at ~1/176th the
  cost, and it scales with region size rather than image size.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageOps

mcp = FastMCP("image-colors")

# Compare against the seed, not the neighbour. With cv2's default *floating*
# range a smooth gradient drifts past the tolerance one step at a time and the
# fill wanders across the whole ramp.
_FLOODFILL = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)

DEFAULTS = dict(
    min_count=16,        # candidate floor, in pixels
    min_blob_px=40,      # a colour must form a blob this big to be real
    max_candidates=48,   # how many colours to return
    max_examine=256,     # bound on how many candidates we are willing to test
    blend_tol=3.0,       # linear-mixture error below which a colour is AA
    near_tol=8.0,        # RGB distance within which a colour is a duplicate
)

_BLEND_T_MIN, _BLEND_T_MAX = 0.05, 0.95   # a mixture, not an endpoint


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------

def _hex(rgb) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _unhex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _key(rgb) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def _hsl(rgb) -> list[int]:
    r, g, b = (float(v) / 255.0 for v in rgb[:3])
    h, light, s = colorsys.rgb_to_hls(r, g, b)   # returns hls, not hsl
    return [round(h * 360), round(s * 100), round(light * 100)]


def _blend_err(target, a, b) -> float | None:
    """Error of the best `t*a + (1-t)*b` that reproduces `target`, or None.

    Plain floats, not numpy: these are 3-vectors and numpy's per-call overhead
    dominates. None means the best fit is an endpoint, i.e. not a real mixture.
    """
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    denom = dx * dx + dy * dy + dz * dz
    if denom == 0:
        return None
    t = ((target[0] - b[0]) * dx + (target[1] - b[1]) * dy
         + (target[2] - b[2]) * dz) / denom
    if not (_BLEND_T_MIN < t < _BLEND_T_MAX):
        return None
    r = target[0] - (t * a[0] + (1 - t) * b[0])
    g = target[1] - (t * a[1] + (1 - t) * b[1])
    bl = target[2] - (t * a[2] + (1 - t) * b[2])
    return math.sqrt(r * r + g * g + bl * bl)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def _load(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (rgb uint8 HxWx3, opaque mask or None when fully opaque).

    Transparent pixels are dropped rather than composited onto a matte. For a
    transparent-background asset the RGB under A=0 is arbitrary -- usually
    (0,0,0) -- and would otherwise be reported as a large "filled" colour.

    EXIF orientation is applied, so bboxes match what a viewer shows.

    Known gaps: ICC profiles are ignored (a Display-P3 PNG reads as sRGB), CMYK
    converts naively, and 16-bit input is scaled by PIL. See README.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.array(im.convert("RGBA"))
    rgb = np.ascontiguousarray(arr[:, :, :3])
    alpha = arr[:, :, 3]
    return rgb, (None if int(alpha.min()) == 255 else alpha > 0)


def _pack(rgb: np.ndarray) -> np.ndarray:
    """Pack (...,3) uint8 into (...) uint32, one integer per distinct colour."""
    return ((rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32))


def _canvas(rgb: np.ndarray, alpha: np.ndarray | None) -> tuple[int | None, int]:
    """(canvas key, how many of the four corners agreed).

    Modal colour of the four corners -- no counting, and it does not assume the
    canvas is white. Ties break toward the top-left, so the answer is
    deterministic. If a corner is transparent it does not vote.
    """
    h, w = rgb.shape[:2]
    votes = [_key(rgb[y, x]) for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
             if alpha is None or alpha[y, x]]
    if not votes:
        return None, 0
    best, agree = votes[0], 0
    for v in votes:
        n = votes.count(v)
        if n > agree:
            best, agree = v, n
    return best, agree


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def _shape_stats(mask: np.ndarray, area_px: int) -> tuple[float | None, float | None]:
    """(solidity, thickness) for the largest blob.

    `fill` (area / bbox area) alone mislabels shapes: a thick ring scores ~0.8
    and reads as "filled", a 45-degree square scores 0.5 and reads as "mixed".
    Solidity -- contour area / convex-hull area -- is ~1.0 for any convex fill
    regardless of rotation, and drops for rings, L-shapes and arrows.

    Neither `fill` nor solidity can see that a 160x2 hairline is thin: it fills
    its own bbox and is convex, so both read 1.0. Mean thickness, 2*area /
    perimeter, separates them cleanly -- ~2 for that line, ~29 for a 58px circle.

    Solidity's two areas both come from the contour: contourArea slightly
    under-measures a rasterised shape, so mixing in the pixel count would let a
    plain circle score 1.02.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = round(cv2.contourArea(contour) / hull_area, 3) if hull_area > 0 else None
    perimeter = cv2.arcLength(contour, True)
    thickness = round(2.0 * area_px / perimeter, 2) if perimeter > 0 else None
    return solidity, thickness


def _shape(fill: float, thickness: float | None) -> str:
    """A hint, not a classification -- `fill` cannot see through a hole."""
    if thickness is not None and thickness <= 4:
        return "hairline"          # a 2px rule, a glyph stem
    if fill >= 0.60:
        return "filled"
    if fill >= 0.25:
        return "mixed"
    return "stroke"


def _describe(packed: np.ndarray, key: int, count: int, min_blob: int) -> dict | None:
    """Geometry for one colour, or None when it forms nothing but fragments.

    Labelling runs on the colour's own bounding box, not the whole image: a
    colour confined to a corner costs its own area rather than the image's.
    """
    mask = packed == key
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    areas = stats[1:, 4]
    keep = areas >= min_blob
    if not keep.any():
        return None

    blobs = stats[1:][keep]
    areas = areas[keep]
    order = np.argsort(-areas)
    blobs, areas = blobs[order], areas[order]
    x, y, bw, bh, area = (int(v) for v in blobs[0])
    x, y = x + x0, y + y0                      # back to image coordinates
    rgb = ((key >> 16) & 255, (key >> 8) & 255, key & 255)

    largest_label = int(np.flatnonzero(keep)[order[0]]) + 1
    solidity, thickness = _shape_stats((labels == largest_label).astype(np.uint8), area)

    return {
        "hex": _hex(rgb),
        "rgb": list(rgb),
        "hsl": _hsl(rgb),
        "count": count,
        "blobs": int(len(blobs)),
        "sizes": [int(v) for v in areas[:8]],
        "largest": {"bbox": [x, y, bw, bh], "area": area,
                    "fill": round(area / float(bw * bh), 3),
                    "solidity": solidity, "thickness": thickness},
        "shape_hint": _shape(area / float(bw * bh), thickness),
    }


def _drop_near_duplicates(entries: list[dict], canvas: tuple | None,
                          near_tol: float) -> tuple[list[dict], list[str]]:
    """Drop colours a more prevalent colour already covers.

    A gradient's last few steps sit a couple of RGB units from its endpoint --
    inside the blend test's endpoint guard, so that test cannot see them, and
    visually identical to the endpoint anyway.
    """
    pool = [(_unhex(e["hex"]), e["count"]) for e in entries]
    if canvas is not None:
        pool.append((canvas[0], canvas[1]))

    kept, dropped = [], []
    for e in entries:
        target, mine = _unhex(e["hex"]), e["count"]
        near = any(t != target and mine < c and math.dist(target, t) <= near_tol
                   for t, c in pool)
        (dropped if near else kept).append(e["hex"] if near else e)
    return kept, dropped


def _drop_blends(entries: list[dict], canvas: tuple | None,
                 blend_tol: float) -> tuple[list[dict], list[str]]:
    """Split out colours that are just anti-aliasing.

    A colour is AA when two *more prevalent* colours mix linearly into it.
    The prevalence guard matters: without it a deliberate mid-grey in a
    black-and-white palette would be dropped as "a blend of black and white".
    """
    pool = [(_unhex(e["hex"]), e["count"]) for e in entries]
    if canvas is not None:
        pool.append((canvas[0], canvas[1]))

    kept, dropped = [], []
    for e in entries:
        target, mine = _unhex(e["hex"]), e["count"]
        is_aa = False
        for (a, ca), (b, cb) in combinations(pool, 2):
            if mine >= min(ca, cb):        # a mixture must be rarer than its parents
                continue
            if a == target or b == target:  # never explain a colour by itself
                continue
            err = _blend_err(target, a, b)
            if err is not None and err < blend_tol:
                is_aa = True
                break
        (dropped if is_aa else kept).append(e["hex"] if is_aa else e)
    return kept, dropped


def _analyze(path: str, min_count: int, min_blob: int, max_candidates: int,
             max_examine: int, blend_tol: float, near_tol: float) -> dict:
    rgb, alpha = _load(path)
    h, w = rgb.shape[:2]
    packed = _pack(rgb)

    visible = packed.ravel() if alpha is None else packed[alpha]
    canvas_key, agree = _canvas(rgb, alpha)
    canvas_rgb = None if canvas_key is None else (
        (canvas_key >> 16) & 255, (canvas_key >> 8) & 255, canvas_key & 255)

    excluded = 0
    if canvas_key is not None:
        excluded = int((visible == canvas_key).sum())
        visible = visible[visible != canvas_key]

    uniq, cnt = np.unique(visible, return_counts=True)
    order = np.argsort(-cnt)
    # A blob of min_blob pixels needs at least that many pixels, so anything
    # rarer cannot qualify. Free, and it prunes a lot.
    order = order[cnt[order] >= max(min_count, min_blob)]

    # Walk by descending frequency and keep the first max_candidates that pass
    # geometry -- not the top max_candidates, then filter. Frequency order is
    # what makes the staleness cut below safe.
    entries: list[dict] = []
    examined = stale = 0
    for i in order:
        if len(entries) >= max_candidates or examined >= max_examine:
            break
        if stale >= max_candidates:
            break          # a whole palette's worth of misses in a row; the
                           # remaining colours are rarer and overwhelmingly AA
        examined += 1
        described = _describe(packed, int(uniq[i]), int(cnt[i]), min_blob)
        if described is None:
            stale += 1
        else:
            entries.append(described)
            stale = 0

    canvas_pool = None if canvas_rgb is None else (canvas_rgb, excluded)
    entries, dupes = _drop_near_duplicates(entries, canvas_pool, near_tol)
    entries, blends = _drop_blends(entries, canvas_pool, blend_tol)
    dropped = dupes + blends

    result = {
        "path": path,
        "size": [w, h],
        "has_alpha": alpha is not None,
        "canvas": None if canvas_rgb is None else {
            "hex": _hex(canvas_rgb), "rgb": list(canvas_rgb),
            "hsl": _hsl(canvas_rgb), "excluded_px": excluded,
            "corner_agreement": f"{agree}/4",
        },
        "colors": entries,
        "dropped_aa": {"count": len(dropped), "hexes": dropped[:8]},
        "examined_candidates": examined,
    }
    if not entries:
        result["note"] = (
            "no colour formed a blob large enough to count as a flat fill; this "
            "is normal for photographs, gradients and heavy JPEG artefacts, "
            "where the exact-enumeration premise does not hold")
    return result


@lru_cache(maxsize=32)
def _cached(path: str, mtime_ns: int, min_count: int, min_blob: int,
            max_candidates: int, max_examine: int, blend_tol: float,
            near_tol: float) -> str:
    return json.dumps(_analyze(path, min_count, min_blob, max_candidates,
                               max_examine, blend_tol, near_tol))


def _resolve(path: str) -> tuple[str, int]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"no such image: {resolved}")
    return resolved, os.stat(resolved).st_mtime_ns


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def _colors(path: str, min_blob_px: int, max_candidates: int) -> dict:
    resolved, mtime = _resolve(path)
    return json.loads(_cached(resolved, mtime, DEFAULTS["min_count"], min_blob_px,
                              max_candidates, DEFAULTS["max_examine"],
                              DEFAULTS["blend_tol"], DEFAULTS["near_tol"]))


@mcp.tool()
def image_colors(path: str, min_blob_px: int = DEFAULTS["min_blob_px"],
                 max_candidates: int = DEFAULTS["max_candidates"]) -> dict:
    """List the colours an image uses, canvas excluded.

    Each colour comes with the geometry of the regions it forms: how many
    blobs, their sizes, and the bbox/fill/solidity of the largest. That geometry
    is what ties a colour to an element -- five blobs of identical size in one
    column is a row of identical shapes; `fill` near 0.785 is a circle (pi/4);
    a low `fill` with high `solidity` means a rotated or irregular solid.

    Colours are exact: every distinct 24-bit value is enumerated, nothing is
    averaged or clustered. Anti-aliasing is rejected by geometry plus a
    linear-mixture test, so `colors` should be the design palette. `count` is
    reported per colour but coverage ratios are deliberately not.

    bbox is [x, y, width, height] in image pixels, origin top-left.
    """
    return _colors(path, min_blob_px, max_candidates)


@mcp.tool()
def image_pixel(path: str, x: int, y: int, tolerance: int = 10) -> dict:
    """Colour at one pixel, snapped to the flat region containing it.

    x/y are image pixels, origin top-left. A rough aim is fine: the point is a
    *seed*, and the flat region it falls in is grown from there, so the answer
    is the element's fill colour rather than whatever that one pixel was.

    `hex` is the mode of the region -- the most common exact value, guaranteed
    to exist in the image -- not an average. `dominant_share` says how much of
    the region it covers; `flat` is that share >= 0.95. When `flat` is false
    the seed landed on a gradient or an edge and the answer should not be
    treated as a single design colour.

    `tolerance` is per-channel and measured against the seed (FIXED_RANGE), so
    a gradient cannot drift the fill away. It is deliberately tight: 18 would
    merge #121212 with #1C1C1C, which are distinct surfaces in dark UI.
    """
    rgb, alpha = _load(path)
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"({x}, {y}) is outside the image, which is {w}x{h}")
    if alpha is not None and not alpha[y, x]:
        return {"seed": [x, y], "error": "seed is a transparent pixel",
                "hex": None}

    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(rgb, mask, (x, y), 0, (tolerance,) * 3, (tolerance,) * 3, _FLOODFILL)
    sel = mask[1:-1, 1:-1].astype(bool)
    if not sel.any():
        return {"seed": [x, y], "error": "seed rejected"}

    packed = _pack(rgb[sel])
    uniq, cnt = np.unique(packed, return_counts=True)
    top = int(np.argmax(cnt))
    key, hits = int(uniq[top]), int(cnt[top])
    rgb_mode = ((key >> 16) & 255, (key >> 8) & 255, key & 255)
    share = hits / float(packed.size)
    ys, xs = np.nonzero(sel)

    return {
        "seed": [x, y],
        "hex": _hex(rgb_mode),
        "rgb": list(rgb_mode),
        "hsl": _hsl(rgb_mode),
        "region": {
            "area_px": int(packed.size),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "distinct": int(uniq.size),
        },
        "dominant_share": round(share, 4),
        "flat": bool(share >= 0.95),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
