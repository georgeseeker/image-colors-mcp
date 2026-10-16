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
* **A pixel is a colour within `match_tol`, because a lossy file has no exact
  values left.** Exact enumeration holds for anything a vector tool wrote, and
  fails for anything that has been through JPEG, a resize or a re-encode: a
  flat box then has no single value holding more than ~29% of it, so equality
  fragments the region into thousands of one-pixel islands and the fill is
  reported as a `hairline` of a few thousand pixels. Matching per channel
  within 4 rebuilds the region, and is measurably free on a clean export.
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

INSTRUCTIONS = """\
Report the colours an image uses, canvas excluded, plus enough geometry about
each one to tie it back to an element.

Reach for `image_colors` instead of looking at a screenshot: a model reading an
image cannot tell #282c34 from #2b303b, and this answers with values that really
exist in the file -- never averaged, quantised or clustered. `bbox` is
[x, y, w, h] in image pixels, origin top-left.

`count` is how many pixels are that colour within `match_tol` per channel;
`exact_px` is how many match it exactly. On a clean export the two are equal.
`count` far above `exact_px` means the image went through a lossy step (JPEG, a
resize, a re-encode): the fills are still real and `hex` is still the region's
mode, but `count` is the honest measure of the region and `exact_px` is not.
The result carries a `note` saying so when that happens.

The tail of `colors` can hold single-pixel rows left over from a lossy fill's
grain; they are marked `hairline` and their counts are one or two orders of
magnitude below the real fills. Lead with the first few entries.

Photographs and pure gradients have no flat fills and will come back with an
empty `colors` and an explanatory `note`.
"""

mcp = FastMCP("image-colors", instructions=INSTRUCTIONS)

DEFAULTS = dict(
    min_count=16,        # candidate floor, in pixels
    min_blob_px=40,      # a colour must form a blob this big to be real
    max_candidates=48,   # how many colours to return
    max_examine=256,     # bound on how many candidates we are willing to test
    blend_tol=3.0,       # linear-mixture error below which a colour is AA
    near_tol=8.0,        # RGB distance within which a colour is a duplicate
    match_tol=4,         # per-channel distance at which a pixel is that colour
    max_walk=20000,      # bound on how many histogram bins we walk past
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


def _channels(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (packed >> 16) & 255, (packed >> 8) & 255, packed & 255


def _within(packed: np.ndarray, key: int, tol: int,
            ch: tuple | None = None) -> np.ndarray:
    """Pixels that *are* this colour, allowing `tol` per channel.

    Lossy storage does not leave a flat fill at one exact value. Measured on a
    flat blue box, lossy variants differ in how far the noise reaches:

        source                     mode share   +-1   +-2   +-4   +-8
        clean vector export            95%      95%   95%   95%   95%
        JPEG q55                       82%      83%   88%   90%   92%
        downscaled to 40% and back     88%      89%   90%   91%   92%
        screenshot re-encoded          29%      94%   97%   98%   99%

    The last row is the awkward one: no single value holds more than 29% of a
    flat box, so equality fragments it into thousands of one-pixel islands --
    reported as a 7719-pixel "hairline" for a region that is really ~180000
    pixels and unmistakably `filled`. `tol=4` puts 98% of it back into one
    connected blob, and costs the clean export nothing at all, because a clean
    export has no neighbours within 4 to absorb.

    Per channel rather than Euclidean: it is the box the existing `near_tol=8`
    duplicate test already encloses (4,4,4 is 6.93 apart), so this widens the
    region test without widening what counts as a duplicate colour.

    `ch` is the unpacked image, passed in because a candidate is described with
    one of these and unpacking three times per candidate dominated the runtime
    (7.6s against 0.46s on a 1800x1350 before this). Lookup tables rather than
    abs-and-compare: three gathers and two ands beats six arithmetic passes plus
    two temporaries.
    """
    if tol <= 0:
        return packed == key
    r, g, b = _channels(packed) if ch is None else ch
    rk, gk, bk = (key >> 16) & 255, (key >> 8) & 255, key & 255
    lr = np.zeros(256, bool)
    lg = np.zeros(256, bool)
    lb = np.zeros(256, bool)
    lr[max(0, rk - tol):rk + tol + 1] = True
    lg[max(0, gk - tol):gk + tol + 1] = True
    lb[max(0, bk - tol):bk + tol + 1] = True
    return lr[r] & lg[g] & lb[b]


def _close(a: int, b: int, tol: int) -> bool:
    """Scalar twin of `_within`, for deciding whether two colours are the same."""
    return (abs(((a >> 16) & 255) - ((b >> 16) & 255)) <= tol
            and abs(((a >> 8) & 255) - ((b >> 8) & 255)) <= tol
            and abs((a & 255) - (b & 255)) <= tol)


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


def _describe(packed: np.ndarray, key: int, exact: int, min_blob: int,
              match_tol: int, ch: tuple | None = None) -> dict | None:
    """Geometry for one colour, or None when it forms nothing but fragments.

    Labelling runs on the colour's own bounding box, not the whole image: a
    colour confined to a corner costs its own area rather than the image's.

    The mask is `match_tol`-tolerant, so `count` is the size of the region the
    colour occupies rather than how often one exact value appears. `exact_px`
    is that narrower number: the two together say how lossy the image is, and
    they are equal on anything that came straight out of a vector tool.

    Tolerance chains along a noise ridge. JPEG leaves thin rows at block
    boundaries whose every pixel holds a different value, each too rare to be a
    candidate on its own; inside a tolerance they become one long sliver and
    pass `min_blob`. So a blob no wider than the tolerance in either direction
    is dropped *when it is mostly not the colour it is named for* -- it is
    something the tolerance assembled rather than something that is there. The
    share test is what keeps a genuine hairline: a 2px rule that really is one
    value scores 1.0 and survives, and on a clean export every blob scores 1.0,
    so nothing is dropped at all.
    """
    mask = _within(packed, key, match_tol, ch)
    count = int(mask.sum())
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
    if match_tol > 0 and keep.any():
        here = (packed[y0:y1, x0:x1] == key)
        per_blob = np.bincount(labels[here], minlength=stats.shape[0])[1:]
        thin = (stats[1:, 2] <= match_tol) | (stats[1:, 3] <= match_tol)
        assembled = per_blob / np.maximum(areas, 1) < 0.5
        keep &= ~(thin & assembled)
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
        "exact_px": exact,
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
             max_examine: int, blend_tol: float, near_tol: float,
             match_tol: int, max_walk: int) -> dict:
    rgb, alpha = _load(path)
    h, w = rgb.shape[:2]
    packed = _pack(rgb)
    ch = _channels(packed) if match_tol > 0 else None

    visible = packed.ravel() if alpha is None else packed[alpha]
    canvas_key, agree = _canvas(rgb, alpha)
    canvas_rgb = None if canvas_key is None else (
        (canvas_key >> 16) & 255, (canvas_key >> 8) & 255, canvas_key & 255)

    excluded = 0
    if canvas_key is not None:
        # Tolerant too, and that matters on a lossy image: the corners sample
        # one value out of the noise (#fdfdfe) while the modal background value
        # is another (#fefefe, 358020 px). Excluding the corner value alone
        # would leave the whole background in the pool, where it comes back as
        # the dominant "colour" of a diagram that has no such element.
        on_canvas = _within(visible, canvas_key, match_tol)
        excluded = int(on_canvas.sum())
        visible = visible[~on_canvas]

    uniq, cnt = np.unique(visible, return_counts=True)
    order = np.argsort(-cnt)
    # A blob of min_blob pixels needs at least that many pixels, so anything
    # rarer cannot qualify. Free, and it prunes a lot.
    order = order[cnt[order] >= max(min_count, min_blob)]

    # Walk by descending frequency and keep the first max_candidates that pass
    # geometry -- not the top max_candidates, then filter. Frequency order is
    # what makes the staleness cut below safe.
    #
    # A colour within match_tol of one already accepted is the same region seen
    # through a different noise value, so it is skipped rather than described.
    # Without that, a lossy image loses its palette twice over: every one of
    # the ~110 near-neighbours of the first colour burns an examine slot, and
    # the survivors come back as duplicates with counts too close for the
    # `mine < c` tie-break in _drop_near_duplicates to resolve. Suppressed
    # candidates cost neither an examine slot nor a geometry pass.
    entries: list[dict] = []
    accepted: list[int] = []
    examined = stale = walked = 0
    for i in order:
        if len(entries) >= max_candidates or examined >= max_examine:
            break
        if stale >= max_candidates:
            break          # a whole palette's worth of misses in a row; the
                           # remaining colours are rarer and overwhelmingly AA
        if walked >= max_walk:
            break
        walked += 1
        key = int(uniq[i])
        if any(_close(key, a, match_tol) for a in accepted):
            continue
        examined += 1
        described = _describe(packed, key, int(cnt[i]), min_blob, match_tol, ch)
        if described is None:
            stale += 1
        else:
            entries.append(described)
            accepted.append(key)
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
        "match_tol": match_tol,
    }
    if not entries:
        result["note"] = (
            "no colour formed a blob large enough to count as a flat fill; this "
            "is normal for photographs and gradients, where the single-fill "
            "premise does not hold")
    else:
        # A dithered fill keeps its pixels but spreads them over thousands of
        # values, so the biggest region's exact share is the cheapest honest
        # reading of how lossy the image is. Clean exports sit at 1.0.
        lead = max(entries, key=lambda e: e["count"])
        share = lead["exact_px"] / float(lead["count"])
        if share < 0.6:
            result["note"] = (
                "this image has been through a lossy step (JPEG, a resize, or a "
                "re-encode): its flat fills are dithered, so no single value "
                "holds a region. `hex` values are still exact and are still the "
                "region's mode, but `count` is tolerance-matched and `exact_px` "
                "is the old exact number -- compare the two to see how lossy "
                "the largest region is (%.2f exact)." % share)
    return result


@lru_cache(maxsize=32)
def _cached(path: str, mtime_ns: int, min_count: int, min_blob: int,
            max_candidates: int, max_examine: int, blend_tol: float,
            near_tol: float, match_tol: int, max_walk: int) -> str:
    return json.dumps(_analyze(path, min_count, min_blob, max_candidates,
                               max_examine, blend_tol, near_tol, match_tol,
                               max_walk))


def _resolve(path: str) -> tuple[str, int]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"no such image: {resolved}")
    return resolved, os.stat(resolved).st_mtime_ns


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def _colors(path: str, min_blob_px: int, max_candidates: int,
            match_tol: int) -> dict:
    resolved, mtime = _resolve(path)
    return json.loads(_cached(resolved, mtime, DEFAULTS["min_count"], min_blob_px,
                              max_candidates, DEFAULTS["max_examine"],
                              DEFAULTS["blend_tol"], DEFAULTS["near_tol"],
                              match_tol, DEFAULTS["max_walk"]))


@mcp.tool()
def image_colors(path: str, min_blob_px: int = DEFAULTS["min_blob_px"],
                 max_candidates: int = DEFAULTS["max_candidates"],
                 match_tol: int = DEFAULTS["match_tol"]) -> dict:
    """List the colours an image uses, canvas excluded.

    Each colour comes with the geometry of the regions it forms: how many
    blobs, their sizes, and the bbox/fill/solidity of the largest. That geometry
    is what ties a colour to an element -- five blobs of identical size in one
    column is a row of identical shapes; `fill` near 0.785 is a circle (pi/4);
    a low `fill` with high `solidity` means a rotated or irregular solid.

    `hex` is always an exact value that exists in the image -- nothing is
    averaged or clustered. `count` is how many pixels are that colour within
    `match_tol` per channel, and `exact_px` is how many match it exactly.
    `count > exact_px` means the fill has been dithered by a lossy step; the
    region is real and `hex` is its mode, but the image is not a clean export.
    On a clean export the two are equal.

    Anti-aliasing is rejected by geometry plus a linear-mixture test, so
    `colors` should be the design palette. Coverage ratios are deliberately
    not reported.

    bbox is [x, y, width, height] in image pixels, origin top-left.
    """
    return _colors(path, min_blob_px, max_candidates, match_tol)


if __name__ == "__main__":
    mcp.run(transport="stdio")
