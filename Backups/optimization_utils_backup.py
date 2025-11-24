# --- REPLACE contents of optimization_utils.py ---
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

# NEW: Growing Trees SA Engine
@njit(cache=True)
def sa_numba_growing(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed, compression):
    np.random.seed(seed)
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    
    # Cache setup
    cached_px = np.zeros((n, NV), dtype=np.float64)
    cached_py = np.zeros((n, NV), dtype=np.float64)
    cached_bboxes = np.zeros((n, 4), dtype=np.float64)
    
    current_scale = 0.90 
    
    # Init geometry
    for i in range(n):
        px, py = get_poly(cxs[i], cys[i], cangs[i], current_scale)
        cached_px[i], cached_py[i] = px, py
        cached_bboxes[i] = get_bbox(px, py)

    bs = calc_side_cached(cached_bboxes, n)
    cs = bs 
    
    alpha = (Tmin/T0)**(1.0/iterations)
    
    for it in range(iterations):
        # 1. Grow Phase: Linearly scale 0.90 -> 1.00 over first 70% of iterations
        if it < iterations * 0.7:
            current_scale = 0.90 + (0.10 * (it / (iterations * 0.7)))
        else:
            current_scale = 1.0
            
        i = np.random.randint(0, n)
        old_x, old_y, old_ang = cxs[i], cys[i], cangs[i]
        old_px, old_py, old_bbox = cached_px[i].copy(), cached_py[i].copy(), cached_bboxes[i].copy()
        
        dm = move_scale * (1.0 if it < iterations*0.7 else (T0 * alpha**it)/T0) # Keep moves high during growth
        
        # Moves
        r = np.random.random()
        if r < 0.5: # Translate + Gravity
            dx = (np.random.random() - 0.5) * dm
            dy = (np.random.random() - 0.5) * dm
            
            # BOUNDING BOX GRAVITY: Pull towards center of current bounds, not (0,0)
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
            
        elif r < 0.98: # Rotate
            cangs[i] += (np.random.random() - 0.5) * rot_scale
        else: # Axis Compression Jump (rare)
            # Squeeze the widest axis globally for this specific tree
            gx0, gy0, gx1, gy1 = cached_bboxes[i]
            if (gx1-gx0) > (gy1-gy0): cxs[i] *= 0.99
            else: cys[i] *= 0.99

        # Check Logic
        npx, npy = get_poly(cxs[i], cys[i], cangs[i], current_scale)
        nbbox = get_bbox(npx, npy)
        
        has_overlap = False
        for k in range(n):
            if k != i and overlap(npx, npy, nbbox, cached_px[k], cached_py[k], cached_bboxes[k]):
                has_overlap = True; break
        
        if not has_overlap:
            cached_bboxes[i] = nbbox
            ns = calc_side_cached(cached_bboxes, n)
            delta = ns - cs
            
            # Metropolis: Accept improvement OR probabilistic accept OR purely geometric acceptance during growth
            if delta < 0 or (current_scale < 1.0) or (np.random.random() < np.exp(-delta / (T0 * alpha**it))):
                cs = ns
                cached_px[i], cached_py[i] = npx, npy
                if current_scale >= 0.999 and ns < bs:
                    bs = ns
                    bxs[:], bys[:], bangs[:] = cxs, cys, cangs
            else:
                 cxs[i], cys[i], cangs[i] = old_x, old_y, old_ang
                 cached_bboxes[i] = old_bbox
        else:
            cxs[i], cys[i], cangs[i] = old_x, old_y, old_ang

    return bxs, bys, bangs, bs

# Keep original sa_numba for backward compatibility
@njit(cache=True)
def sa_numba(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed, compression, collision_scale=1.0, target_side=0.0):
    best_bboxes = cached_bboxes.copy()

    bs = calc_side_cached(cached_bboxes, n)
    cs = bs
    T = T0
    alpha = (Tmin / T0) ** (1.0 / iterations) if iterations > 0 else 0.99
    no_imp = 0
    
    # Backup buffers
    opx1 = np.zeros(NV, dtype=np.float64)
    opy1 = np.zeros(NV, dtype=np.float64)
    obb1 = np.zeros(4, dtype=np.float64)
    opx2 = np.zeros(NV, dtype=np.float64)
    opy2 = np.zeros(NV, dtype=np.float64)
    obb2 = np.zeros(4, dtype=np.float64)

    for it in range(iterations):
        move_type = np.random.randint(0, 8)
        sc = T / T0
        
        idx1, idx2 = -1, -1
        ox1, oy1, oa1 = 0.0, 0.0, 0.0
        ox2, oy2, oa2 = 0.0, 0.0, 0.0

        if move_type < 4:
            # Single tree moves
            i = np.random.randint(0, n)
            idx1 = i
            ox1, oy1, oa1 = cxs[i], cys[i], cangs[i]
            opx1[:] = cached_px[i]
            opy1[:] = cached_py[i]
            obb1[:] = cached_bboxes[i]
            
            cx = np.mean(cxs[:n])
            cy = np.mean(cys[:n])

            if move_type == 0: # Random translation + Gravity
                dx = (np.random.random() - 0.5) * 2 * move_scale * sc
                dy = (np.random.random() - 0.5) * 2 * move_scale * sc
                dx -= cxs[i] * compression * sc
                dy -= cys[i] * compression * sc
                cxs[i] += dx
                cys[i] += dy
                
            elif move_type == 1: # Move towards center
                dx, dy = cx - cxs[i], cy - cys[i]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc
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

            npx, npy = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbb = get_bbox(npx, npy)
            
            if check_overlap_single_cached(i, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i], cangs[i] = ox1, oy1, oa1
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            cached_px[i] = npx
            cached_py[i] = npy
            cached_bboxes[i] = nbb

        elif move_type == 4 and n > 1:
            # Swap
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            while j == i: j = np.random.randint(0, n)
            
            idx1, idx2 = i, j
            ox1, oy1, oa1 = cxs[i], cys[i], cangs[i]
            ox2, oy2, oa2 = cxs[j], cys[j], cangs[j]
            opx1[:] = cached_px[i]
            opy1[:] = cached_py[i]
            obb1[:] = cached_bboxes[i]
            opx2[:] = cached_px[j]
            opy2[:] = cached_py[j]
            obb2[:] = cached_bboxes[j]

            cxs[i], cys[i] = ox2, oy2
            cxs[j], cys[j] = ox1, oy1
            
            npxi, npyi = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbbi = get_bbox(npxi, npyi)
            npxj, npyj = get_poly(cxs[j], cys[j], cangs[j], collision_scale)
            nbbj = get_bbox(npxj, npyj)

            if check_overlap_pair_cached(i, j, npxi, npyi, nbbi, npxj, npyj, nbbj, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i] = ox1, oy1
                cxs[j], cys[j] = ox2, oy2
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            cached_px[i], cached_py[i], cached_bboxes[i] = npxi, npyi, nbbi
            cached_px[j], cached_py[j], cached_bboxes[j] = npxj, npyj, nbbj

        elif move_type == 5:
            # Bbox center move
            i = np.random.randint(0, n)
            idx1 = i
            ox1, oy1, oa1 = cxs[i], cys[i], cangs[i]
            opx1[:] = cached_px[i]
            opy1[:] = cached_py[i]
            obb1[:] = cached_bboxes[i]

            gx0, gy0, gx1, gy1 = get_global_bbox_cached(cached_bboxes, n)
            bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            dx, dy = bcx - cxs[i], bcy - cys[i]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-6:
                step = np.random.random() * move_scale * sc * 0.5
                cxs[i] += dx / d * step
                cxs[i] -= cxs[i] * compression * sc
                cys[i] += dy / d * step
                cys[i] -= cys[i] * compression * sc

            npx, npy = get_poly(cxs[i], cys[i], cangs[i], collision_scale)
            nbb = get_bbox(npx, npy)

            if check_overlap_single_cached(i, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                cxs[i], cys[i] = ox1, oy1
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
            
            cached_px[i], cached_py[i], cached_bboxes[i] = npx, npy, nbb

        elif move_type == 6:
            # Corner tree focus
            corner_indices, count = find_corner_trees_cached(cached_bboxes, n)
            if count > 0:
                idx = corner_indices[np.random.randint(0, count)]
                idx1 = idx
                ox1, oy1, oa1 = cxs[idx], cys[idx], cangs[idx]
                opx1[:] = cached_px[idx]
                opy1[:] = cached_py[idx]
                obb1[:] = cached_bboxes[idx]

                gx0, gy0, gx1, gy1 = get_global_bbox_cached(cached_bboxes, n)
                bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                dx, dy = bcx - cxs[idx], bcy - cys[idx]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc * 0.6
                    cxs[idx] += dx / d * step
                    cys[idx] += dy / d * step
                    cxs[idx] -= cxs[idx] * compression * sc * 1.5
                    cys[idx] -= cys[idx] * compression * sc * 1.5
                    cangs[idx] += (np.random.random() - 0.5) * rot_scale * sc * 0.5
                    cangs[idx] = cangs[idx] % 360

                npx, npy = get_poly(cxs[idx], cys[idx], cangs[idx], collision_scale)
                nbb = get_bbox(npx, npy)

                if check_overlap_single_cached(idx, npx, npy, nbb, cached_px, cached_py, cached_bboxes, n):
                    cxs[idx], cys[idx], cangs[idx] = ox1, oy1, oa1
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
            idx1, idx2 = i, j
            ox1, oy1, oa1 = cxs[i], cys[i], cangs[i]
            ox2, oy2, oa2 = cxs[j], cys[j], cangs[j]
            opx1[:] = cached_px[i]
            opy1[:] = cached_py[i]
            obb1[:] = cached_bboxes[i]
            opx2[:] = cached_px[j]
            opy2[:] = cached_py[j]
            obb2[:] = cached_bboxes[j]

            dx = (np.random.random() - 0.5) * move_scale * sc * 0.5
            dy = (np.random.random() - 0.5) * move_scale * sc * 0.5
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
                cxs[i], cys[i] = ox1, oy1
                cxs[j], cys[j] = ox2, oy2
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
                best_px[:] = cached_px
                best_py[:] = cached_py
                best_bboxes[:] = cached_bboxes
                no_imp = 0
                
                if target_side > 0 and bs <= target_side:
                    break
            else:
                no_imp += 1
        else:
            # Revert to Previous (Correct SA logic)
            if idx1 != -1:
                cxs[idx1], cys[idx1], cangs[idx1] = ox1, oy1, oa1
                cached_px[idx1] = opx1
                cached_py[idx1] = opy1
                cached_bboxes[idx1] = obb1
            if idx2 != -1:
                cxs[idx2], cys[idx2], cangs[idx2] = ox2, oy2, oa2
                cached_px[idx2] = opx2
                cached_py[idx2] = opy2
                cached_bboxes[idx2] = obb2
            
            no_imp += 1

        # Reheat
        if no_imp > 600:
            T = min(T * 3.0, T0 * 0.7)
            no_imp = 0

        T *= alpha
        if T < Tmin:
            T = Tmin

    return bxs, bys, bangs, bs
