import numpy as np
from numba import njit

# Tree polygon vertices
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
def pip(px_pt, py_pt, poly_x, poly_y):
    inside = False
    j = NV - 1
    for i in range(NV):
        if ((poly_y[i] > py_pt) != (poly_y[j] > py_pt) and
            px_pt < (poly_x[j] - poly_x[i]) * (py_pt - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@njit(cache=True)
def seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    def ccw(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p3y - p1y) * (p2x - p1x) > (p2y - p1y) * (p3x - p1x)
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)

@njit(cache=True)
def overlap(px1, py1, bb1, px2, py2, bb2):
    # Fast BBox check
    if bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]:
        return False
    # Point in Polygon check
    for i in range(NV):
        if pip(px1[i], py1[i], px2, py2): return True
        if pip(px2[i], py2[i], px1, py1): return True
    # Edge Intersection check
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if seg_intersect(px1[i], py1[i], px1[ni], py1[ni],
                           px2[j], py2[j], px2[nj], py2[nj]):
                return True
    return False

@njit(cache=True)
def calc_side(xs, ys, angs, n):
    if n == 0: return 0.0
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i], 1.0)
        x0, y0, x1, y1 = get_bbox(px, py)
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return max(gx1 - gx0, gy1 - gy0)

@njit(cache=True)
def calc_side_cached(cached_bboxes, n):
    if n == 0: return 0.0
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        x0, y0, x1, y1 = cached_bboxes[i]
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return max(gx1 - gx0, gy1 - gy0)

@njit(cache=True)
def get_global_bbox(xs, ys, angs, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i], 1.0)
        x0, y0, x1, y1 = get_bbox(px, py)
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return gx0, gy0, gx1, gy1

@njit(cache=True)
def get_global_bbox_cached(cached_bboxes, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        x0, y0, x1, y1 = cached_bboxes[i]
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return gx0, gy0, gx1, gy1

@njit(cache=True)
def find_corner_trees(xs, ys, angs, n):
    gx0, gy0, gx1, gy1 = get_global_bbox(xs, ys, angs, n)
    eps = 0.01
    corner_indices = np.zeros(n, dtype=np.int32)
    count = 0
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i], 1.0)
        x0, y0, x1, y1 = get_bbox(px, py)
        if abs(x0 - gx0) < eps or abs(x1 - gx1) < eps or \
           abs(y0 - gy0) < eps or abs(y1 - gy1) < eps:
            corner_indices[count] = i
            count += 1
    return corner_indices, count

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

@njit(cache=True)
def sa_numba(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed, compression, collision_scale=1.0):
    np.random.seed(seed)

    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()

    # Initialize Cache
    cached_px = np.zeros((n, NV), dtype=np.float64)
    cached_py = np.zeros((n, NV), dtype=np.float64)
    cached_bboxes = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        px, py = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
        cached_px[i] = px
        cached_py[i] = py
        cached_bboxes[i] = get_bbox(px, py)

    # Best Cache
    best_px = cached_px.copy()
    best_py = cached_py.copy()
    best_bboxes = cached_bboxes.copy()

    bs = calc_side_cached(cached_bboxes, n)
    cs = bs
    T = T0
    alpha = (Tmin / T0) ** (1.0 / iterations) if iterations > 0 else 0.99
    no_imp = 0

    for it in range(iterations):
        move_type = np.random.randint(0, 8)  # 8 move types
        sc = T / T0

        if move_type < 4:
            # Single tree moves
            i = np.random.randint(0, n)
            ox, oy, oa = cxs[i], cys[i], cangs[i]
            
            cx = np.mean(cxs[:n])
            cy = np.mean(cys[:n])

            if move_type == 0: # Random translation + Gravity
                # Gravity: bias towards (0,0)
                # x_new = x_curr + random_step + (0 - x_curr) * compression
                dx = (np.random.random() - 0.5) * 2 * move_scale * sc
                dy = (np.random.random() - 0.5) * 2 * move_scale * sc
                
                # Apply gravity
                dx -= cxs[i] * compression * sc
                dy -= cys[i] * compression * sc
                
                cxs[i] += dx
                cys[i] += dy
                
            elif move_type == 1: # Move towards center (Enhanced Gravity)
                dx, dy = cx - cxs[i], cy - cys[i]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc
                    # Stronger pull
                    cxs[i] += dx / d * step + (0 - cxs[i]) * compression * sc
                    cys[i] += dy / d * step + (0 - cys[i]) * compression * sc
            elif move_type == 2: # Rotation
                cangs[i] += (np.random.random() - 0.5) * 2 * rot_scale * sc
                cangs[i] = cangs[i] % 360
            else: # Mixed + Gravity
                dx = (np.random.random() - 0.5) * move_scale * sc
                dy = (np.random.random() - 0.5) * move_scale * sc
                
                dx -= cxs[i] * compression * sc
                dy -= cys[i] * compression * sc
                
                cxs[i] += dx
                cys[i] += dy
                cangs[i] += (np.random.random() - 0.5) * rot_scale * sc
                cangs[i] = cangs[i] % 360

            # Update cache for i
            npx, npy = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbb = get_bbox(npx, npy)
            
            # Check overlap using cache
            if check_overlap_single_cached(i, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i], cangs[i] = ox, oy, oa
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            # Valid move, update cache
            cached_px[i] = npx
            cached_py[i] = npy
            cached_bboxes[i] = nbb

        elif move_type == 4 and n > 1:
            # Swap
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            while j == i: j = np.random.randint(0, n)

            oxi, oyi = cxs[i], cys[i]
            oxj, oyj = cxs[j], cys[j]
            
            cxs[i], cys[i] = oxj, oyj
            cxs[j], cys[j] = oxi, oyi
            
            # Update cache for i and j
            npxi, npyi = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbbi = get_bbox(npxi, npyi)
            npxj, npyj = get_poly(cxs[j], cys[j], cangs[j], collision_scale)
            nbbj = get_bbox(npxj, npyj)

            if check_overlap_pair_cached(i, j, npxi, npyi, nbbi, npxj, npyj, nbbj, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i] = oxi, oyi
                cxs[j], cys[j] = oxj, oyj
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            # Valid move
            cached_px[i], cached_py[i], cached_bboxes[i] = npxi, npyi, nbbi
            cached_px[j], cached_py[j], cached_bboxes[j] = npxj, npyj, nbbj

        elif move_type == 5:
            # Bbox center move + Gravity
            i = np.random.randint(0, n)
            ox, oy = cxs[i], cys[i]

            gx0, gy0, gx1, gy1 = get_global_bbox_cached(cached_bboxes, n)
            bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            dx, dy = bcx - cxs[i], bcy - cys[i]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-6:
                step = np.random.random() * move_scale * sc * 0.5
                cxs[i] += dx / d * step
                cys[i] += dy / d * step
                
                # Gravity
                cxs[i] -= cxs[i] * compression * sc
                cys[i] -= cys[i] * compression * sc

            npx, npy = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbb = get_bbox(npx, npy)

            if check_overlap_single_cached(i, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i] = ox, oy
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            cached_px[i], cached_py[i], cached_bboxes[i] = npx, npy, nbb

        elif move_type == 6:
            # Corner tree focus - Enhanced
            corner_indices, count = find_corner_trees_cached(cached_bboxes, n)
            if count > 0:
                idx = corner_indices[np.random.randint(0, count)]
                ox, oy, oa = cxs[idx], cys[idx], cangs[idx]

                gx0, gy0, gx1, gy1 = get_global_bbox_cached(cached_bboxes, n)
                bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                dx, dy = bcx - cxs[idx], bcy - cys[idx]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    # Larger move scale for corners (2x)
                    step = np.random.random() * move_scale * sc * 0.6 # 0.3 -> 0.6
                    cxs[idx] += dx / d * step
                    cys[idx] += dy / d * step
                    
                    # Gravity for corners
                    cxs[idx] -= cxs[idx] * compression * sc * 1.5 # Stronger gravity for corners
                    cys[idx] -= cys[idx] * compression * sc * 1.5
                    
                    cangs[idx] += (np.random.random() - 0.5) * rot_scale * sc * 0.5
                    cangs[idx] = cangs[idx] % 360

                npx, npy = get_poly(cxs[idx], cys[idx], cangs[idx], collision_scale)
                nbb = get_bbox(npx, npy)

                if check_overlap_single_cached(idx, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                    cxs[idx], cys[idx], cangs[idx] = ox, oy, oa
                    no_imp += 1
                    T *= alpha
                    if T < Tmin: T = Tmin
                    continue
                
                cached_px[idx], cached_py[idx], cached_bboxes[idx] = npx, npy, nbb
            else:
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
        else:
            # Coordinated move
            i = np.random.randint(0, n)
            j = (i + 1) % n

            oxi, oyi = cxs[i], cys[i]
            oxj, oyj = cxs[j], cys[j]

            dx = (np.random.random() - 0.5) * move_scale * sc * 0.5
            dy = (np.random.random() - 0.5) * move_scale * sc * 0.5
            
            # Gravity
            dx -= (cxs[i] + cxs[j])/2 * compression * sc
            dy -= (cys[i] + cys[j])/2 * compression * sc

            cxs[i] += dx
            cys[i] += dy
            cxs[j] += dx
            cys[j] += dy

            npxi, npyi = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbbi = get_bbox(npxi, npyi)
            npxj, npyj = get_poly(cxs[j], cys[j], cangs[j], collision_scale)
            nbbj = get_bbox(npxj, npyj)

            if check_overlap_pair_cached(i, j, npxi, npyi, nbbi, npxj, npyj, nbbj, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i] = oxi, oyi
                cxs[j], cys[j] = oxj, oyj
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            cached_px[i], cached_py[i], cached_bboxes[i] = npxi, npyi, nbbi
            cached_px[j], cached_py[j], cached_bboxes[j] = npxj, npyj, nbbj

        ns = calc_side_cached(cached_bboxes, n)
        delta = ns - cs

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            cs = ns
            if ns < bs:
                bs = ns
                bxs[:] = cxs
                bys[:] = cys
                bangs[:] = cangs
                
                # Update Best Cache
                best_px[:] = cached_px
                best_py[:] = cached_py
                best_bboxes[:] = cached_bboxes
                
                no_imp = 0
            else:
                no_imp += 1
        else:
            # Revert to Best
            cxs[:] = bxs
            cys[:] = bys
            cangs[:] = bangs
            cs = bs
            
            # Revert Cache to Best
            cached_px[:] = best_px
            cached_py[:] = best_py
            cached_bboxes[:] = best_bboxes
            
            no_imp += 1

        # Reheat
        if no_imp > 600:
            T = min(T * 3.0, T0 * 0.7)
            no_imp = 0

        T *= alpha
        if T < Tmin:
            T = Tmin

    return bxs, bys, bangs, bs
