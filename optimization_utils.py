# --- HIGH-PERFORMANCE OPTIMIZATION ENGINE WITH GROWING TREES ---
import numpy as np
from numba import njit

# Tree polygon vertices (Counter-Clockwise)
TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                   -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2,
                   -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)
NV = 15

@njit(cache=True)
def get_poly(cx, cy, deg, scale=1.0):
    rad = deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    px = (TREE_X * scale) * c - (TREE_Y * scale) * s + cx
    py = (TREE_X * scale) * s + (TREE_Y * scale) * c + cy
    return px, py

@njit(cache=True)
def get_bbox(px, py):
    return px.min(), py.min(), px.max(), py.max()

@njit(cache=True)
def seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    d = (by - ay) * (dx - cx) - (bx - ax) * (dy - cy)
    if d == 0: return False
    u = ((bx - ax) * (ay - cy) - (by - ay) * (ax - cx)) / d
    v = ((dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)) / d
    return 0 <= u <= 1 and 0 <= v <= 1

@njit(cache=True)
def pip(px_pt, py_pt, poly_x, poly_y):
    inside = False
    j = NV - 1
    for i in range(NV):
        if ((poly_y[i] > py_pt) != (poly_y[j] > py_pt)) and \
           (px_pt < (poly_x[j] - poly_x[i]) * (py_pt - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@njit(cache=True)
def overlap(px1, py1, bb1, px2, py2, bb2):
    if bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]: return False
    for i in range(NV):
        if pip(px1[i], py1[i], px2, py2): return True
        if pip(px2[i], py2[i], px1, py1): return True
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if seg_intersect(px1[i], py1[i], px1[ni], py1[ni], px2[j], py2[j], px2[nj], py2[nj]): return True
    return False

@njit(cache=True)
def calc_side_cached(cached_bboxes, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        gx0 = min(gx0, cached_bboxes[i, 0])
        gy0 = min(gy0, cached_bboxes[i, 1])
        gx1 = max(gx1, cached_bboxes[i, 2])
        gy1 = max(gy1, cached_bboxes[i, 3])
    return max(gx1 - gx0, gy1 - gy0)

@njit(cache=True)
def get_global_bbox_cached(cached_bboxes, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        x0, y0, x1, y1 = cached_bboxes[i]
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return gx0, gy0, gx1, gy1

@njit(cache=True)
def find_corner_trees_cached(cached_bboxes, n):
    gx0, gy0, gx1, gy1 = get_global_bbox_cached(cached_bboxes, n)
    eps = 0.01
    corner_indices = np.zeros(n, dtype=np.int32)
    count = 0
    for i in range(n):
        x0, y0, x1, y1 = cached_bboxes[i]
        if abs(x0 - gx0) < eps or abs(x1 - gx1) < eps or \
           abs(y0 - gy0) < eps or abs(y1 - gy1) < eps:
            corner_indices[count] = i
            count += 1
    return corner_indices, count

@njit(cache=True)
def check_overlap_single_cached(idx, px1, py1, bb1, cached_px, cached_py, cached_bboxes, n):
    for j in range(n):
        if j != idx:
            if overlap(px1, py1, bb1, cached_px[j], cached_py[j], cached_bboxes[j]):
                return True
    return False

@njit(cache=True)
def check_overlap_pair_cached(i, j, pxi, pyi, bbi, pxj, pyj, bbj, cached_px, cached_py, cached_bboxes, n):
    if overlap(pxi, pyi, bbi, pxj, pyj, bbj):
        return True
    for k in range(n):
        if k != i and k != j:
            if overlap(pxi, pyi, bbi, cached_px[k], cached_py[k], cached_bboxes[k]):
                return True
            if overlap(pxj, pyj, bbj, cached_px[k], cached_py[k], cached_bboxes[k]):
                return True
    return False

# NEW: Growing Trees SA Engine with soft physics
@njit(cache=True)
def sa_numba_growing(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed, compression):
    """
    Growing Trees Strategy: Start with trees at 90% scale (allowing overlap), 
    gradually scale back to 100% while optimizing. Uses Bounding Box Gravity.
    """
    np.random.seed(seed)
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    
    # Cache setup
    cached_px = np.zeros((n, NV), dtype=np.float64)
    cached_py = np.zeros((n, NV), dtype=np.float64)
    cached_bboxes = np.zeros((n, 4), dtype=np.float64)
    
    current_scale = 0.90 
    
    # Init geometry with reduced scale
    for i in range(n):
        px, py = get_poly(cxs[i], cys[i], cangs[i], current_scale)
        cached_px[i], cached_py[i] = px, py
        cached_bboxes[i] = get_bbox(px, py)

    bs = calc_side_cached(cached_bboxes, n)
    cs = bs 
    
    alpha = (Tmin/T0)**(1.0/iterations) if iterations > 0 else 0.99
    
    for it in range(iterations):
        # Grow Phase: Linearly scale 0.90 -> 1.00 over first 70% of iterations
        if it < iterations * 0.7:
            current_scale = 0.90 + (0.10 * (it / (iterations * 0.7)))
        else:
            current_scale = 1.0
            
        i = np.random.randint(0, n)
        old_x, old_y, old_ang = cxs[i], cys[i], cangs[i]
        old_px, old_py, old_bbox = cached_px[i].copy(), cached_py[i].copy(), cached_bboxes[i].copy()
        
        # Keep moves high during growth phase
        dm = move_scale * (1.0 if it < iterations*0.7 else (T0 * alpha**it)/T0)
        
        # Move selection
        r = np.random.random()
        if r < 0.5:  # Translate + Bounding Box Gravity
            dx = (np.random.random() - 0.5) * dm
            dy = (np.random.random() - 0.5) * dm
            
            # BOUNDING BOX GRAVITY: Pull towards center of OTHER trees' bounds, not (0,0)
            gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
            for k in range(n):
                if k == i: continue
                gx0 = min(gx0, cached_bboxes[k, 0])
                gy0 = min(gy0, cached_bboxes[k, 1])
                gx1 = max(gx1, cached_bboxes[k, 2])
                gy1 = max(gy1, cached_bboxes[k, 3])
            
            c_x, c_y = (gx0+gx1)/2, (gy0+gy1)/2
            cxs[i] += dx + (c_x - cxs[i]) * compression * 0.01
            cys[i] += dy + (c_y - cys[i]) * compression * 0.01
            
        elif r < 0.98:  # Rotate
            cangs[i] += (np.random.random() - 0.5) * rot_scale
            cangs[i] = cangs[i] % 360
            
        else:  # Axis Compression Jump (rare, aggressive)
            gx0, gy0, gx1, gy1 = cached_bboxes[i]
            if (gx1-gx0) > (gy1-gy0): 
                cxs[i] *= 0.99
            else: 
                cys[i] *= 0.99

        # Check with current scale
        npx, npy = get_poly(cxs[i], cys[i], cangs[i], current_scale)
        nbbox = get_bbox(npx, npy)
        
        has_overlap = False
        for k in range(n):
            if k != i and overlap(npx, npy, nbbox, cached_px[k], cached_py[k], cached_bboxes[k]):
                has_overlap = True
                break
        
        if not has_overlap:
            cached_bboxes[i] = nbbox
            ns = calc_side_cached(cached_bboxes, n)
            delta = ns - cs
            
            # Metropolis with growth phase leniency
            if delta < 0 or (current_scale < 1.0) or (np.random.random() < np.exp(-delta / (T0 * alpha**it))):
                cs = ns
                cached_px[i], cached_py[i] = npx, npy
                # Only update best when near full scale
                if current_scale >= 0.999 and ns < bs:
                    bs = ns
                    bxs[:], bys[:], bangs[:] = cxs, cys, cangs
            else:
                cxs[i], cys[i], cangs[i] = old_x, old_y, old_ang
                cached_bboxes[i] = old_bbox
        else:
            cxs[i], cys[i], cangs[i] = old_x, old_y, old_ang

    return bxs, bys, bangs, bs
