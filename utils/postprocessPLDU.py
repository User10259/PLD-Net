# utils/postprocess_utils.py
import numpy as np
import math
from shapely.geometry import LineString, box
import cv2

# ===================== Basic utility functions =====================

def _stable_softmax_prob_1d(logits):

    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    exps = np.exp(x - m)
    return float(exps[1] / (exps[0] + exps[1] + 1e-12))


def denormalize_d_theta_vector(norm_d, sin_theta, cos_theta, max_d):
    """
    Convert (norm_d, sinθ, cosθ) to actual d and θ (in radians)
    """
    d = norm_d * max_d
    theta = np.arctan2(sin_theta, cos_theta)
    return d, theta


def convert_grid_to_global_coords(i, j, grid_size_x, grid_size_y):
    """
    Convert the center coordinates of grid (i,j) to global pixel coordinates
    """
    cx = (j + 0.5) * grid_size_x
    cy = (i + 0.5) * grid_size_y
    return cx, cy


def local_polar_to_line_segment(d, theta, cx, cy, grid_size_x, grid_size_y):
    """
    Convert (d, θ)+(cx, cy) into a small line segment within the grid (global coordinates)
    """
    nx = math.cos(theta)
    ny = math.sin(theta)
    d_global = d + (cx * nx + cy * ny)

    point_on_line = np.array([0.0, 0.0]) + d_global * np.array([nx, ny])

    direction = np.array([-ny, nx])
    pt1 = point_on_line + 1000 * direction
    pt2 = point_on_line - 1000 * direction
    full_line = LineString([pt1, pt2])

    cell_box = box(
        cx - grid_size_x / 2,
        cy - grid_size_y / 2,
        cx + grid_size_x / 2,
        cy + grid_size_y / 2
    )
    segment = full_line.intersection(cell_box)
    if segment.is_empty or segment.geom_type != 'LineString':
        return None
    coords = list(segment.coords)
    return coords[0], coords[-1]

def _calculate_laplacian_variance(image_np, line):

    padding = 10
    h, w, _ = image_np.shape
    x_coords = [line[0][0], line[1][0]]
    y_coords = [line[0][1], line[1][1]]

    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)

    if y_max <= y_min or x_max <= x_min: return 0.0

    patch = image_np[y_min:y_max, x_min:x_max]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray_patch, cv2.CV_64F).var()

def merge_line_cluster_pca(cluster):

    all_points = np.vstack([np.array(line, dtype=np.float64) for line in cluster])
    if all_points.shape[0] < 2:
        return None
    centroid = np.mean(all_points, axis=0)
    centered_points = all_points - centroid
    _, _, Vt = np.linalg.svd(centered_points)
    direction_vector = Vt[0]
    projections = centered_points @ direction_vector
    p_start = centroid + projections.min() * direction_vector
    p_end   = centroid + projections.max() * direction_vector
    return (p_start.tolist(), p_end.tolist())


def cluster_lines_v3(lines,
                     angle_thresh_deg=8.0,
                     dist_thresh_endpoint=24.0,
                     dist_thresh_collinear=6.0,
                     gap_max=32.0):
    if not lines:
        return []

    num = len(lines)
    P = []
    for ln in lines:
        p1 = np.array(ln[0], dtype=np.float64)
        p2 = np.array(ln[1], dtype=np.float64)
        v = p2 - p1
        n = np.linalg.norm(v)
        if n < 1e-6:
            u = np.array([1.0, 0.0])
        else:
            u = v / n
        ang = np.arctan2(u[1], u[0])
        mid = (p1 + p2) * 0.5
        P.append(dict(p1=p1, p2=p2, u=u, ang=ang, mid=mid))

    A = np.zeros((num, num), dtype=bool)
    ang_tol = np.deg2rad(angle_thresh_deg)

    for i in range(num):
        for j in range(i + 1, num):
            pi, pj = P[i], P[j]

            d_ang = abs(pi['ang'] - pj['ang'])
            d_ang = min(d_ang, np.pi - d_ang)
            if d_ang > ang_tol:
                continue

            mind = min(
                np.linalg.norm(pi['p1'] - pj['p1']), np.linalg.norm(pi['p1'] - pj['p2']),
                np.linalg.norm(pi['p2'] - pj['p1']), np.linalg.norm(pi['p2'] - pj['p2'])
            )
            if mind > dist_thresh_endpoint:
                continue

            vi = pi['p2'] - pi['p1']
            denom = np.linalg.norm(vi) + 1e-12
            perp = abs(vi[1] * (pj['mid'][0] - pi['p1'][0]) - vi[0] * (pj['mid'][1] - pi['p1'][1])) / denom
            if perp > dist_thresh_collinear:
                continue

            u_avg = pi['u'] + pj['u']
            if np.linalg.norm(u_avg) < 1e-6:
                u_avg = pi['u']
            else:
                u_avg = u_avg / (np.linalg.norm(u_avg) + 1e-12)

            origin = 0.5 * (pi['mid'] + pj['mid'])

            def proj_interval(p1, p2, o, u):
                t1 = (p1 - o).dot(u)
                t2 = (p2 - o).dot(u)
                return min(t1, t2), max(t1, t2)

            li = proj_interval(pi['p1'], pi['p2'], origin, u_avg)
            lj = proj_interval(pj['p1'], pj['p2'], origin, u_avg)
            gap = max(lj[0] - li[1], li[0] - lj[1])  # >0 表示存在间隙
            if gap > gap_max:
                continue

            A[i, j] = A[j, i] = True

    visited = np.zeros(num, dtype=bool)
    clusters = []
    for i in range(num):
        if visited[i]:
            continue
        q = [i]
        visited[i] = True
        cur = []
        while q:
            u = q.pop(0)
            cur.append(lines[u])
            nbr = np.where(A[u])[0]
            for v in nbr:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        clusters.append(cur)
    return clusters


def _angle_of(line):
    (x1, y1), (x2, y2) = line
    return np.arctan2(y2 - y1, x2 - x1)


def _line_iou(l1, l2, buf=2.0):
    g1 = LineString(l1).buffer(buf)
    g2 = LineString(l2).buffer(buf)
    u = g1.union(g2).area
    if u <= 0:
        return 0.0
    return g1.intersection(g2).area / u


def filter_lines(lines, min_length=20, iou_thresh=0.70, angle_thresh=3.5, buffer_px=4.5):

    if not lines:
        return []

    def L(l):
        a, b = np.array(l[0], dtype=np.float64), np.array(l[1], dtype=np.float64)
        return float(np.linalg.norm(b - a))

    def A(l):
        (x1, y1), (x2, y2) = l
        return math.atan2(y2 - y1, x2 - x1)

    cand = [l for l in lines if L(l) >= float(min_length)]
    if not cand:
        return []

    cand.sort(key=L, reverse=True)
    keep = []
    suppressed = [False] * len(cand)
    ang_tol = np.deg2rad(angle_thresh)

    for i in range(len(cand)):
        if suppressed[i]:
            continue
        keep.append(cand[i])
        ai = A(cand[i])
        bi = LineString(cand[i]).buffer(buffer_px)
        for j in range(i + 1, len(cand)):
            if suppressed[j]:
                continue
            aj = A(cand[j])
            d_ang = abs(ai - aj)
            d_ang = min(d_ang, math.pi - d_ang)
            if d_ang > ang_tol:
                continue
            bj = LineString(cand[j]).buffer(buffer_px)
            iou = bj.intersection(bi).area / (bj.union(bi).area + 1e-12)
            if iou > iou_thresh:
                suppressed[j] = True
    return keep


def stitch_collinear_lines(lines, angle_thresh_deg=3.5, gap_max=24.0, perp_max=6.0):

    if not lines:
        return []

    Ls = [(np.array(a, float), np.array(b, float)) for a, b in lines]
    used = [False] * len(Ls)
    out = []
    ang_tol = np.deg2rad(angle_thresh_deg)

    def angle(l):
        p1, p2 = l
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def unit(l):
        p1, p2 = l
        v = p2 - p1
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1., 0.])

    for i in range(len(Ls)):
        if used[i]:
            continue
        base = Ls[i]
        ui = unit(base)
        ai = angle(base)
        merged = True
        while merged:
            merged = False
            best_j = -1
            for j in range(len(Ls)):
                if i == j or used[j]:
                    continue
                aj = angle(Ls[j])
                d_ang = abs(ai - aj)
                d_ang = min(d_ang, np.pi - d_ang)
                if d_ang > ang_tol:
                    continue

                p1, p2 = base
                q1, q2 = Ls[j]

                o = 0.5 * ((p1 + p2) / 2 + (q1 + q2) / 2)

                def proj(a):
                    return (a - o).dot(ui)

                li = (min(proj(p1), proj(p2)), max(proj(p1), proj(p2)))
                lj = (min(proj(q1), proj(q2)), max(proj(q1), proj(q2)))
                gap = max(lj[0] - li[1], li[0] - lj[1])

                v = base[1] - base[0]
                denom = np.linalg.norm(v) + 1e-12
                midj = 0.5 * (q1 + q2)
                perp = abs(v[1] * (midj[0] - base[0][0]) - v[0] * (midj[1] - base[0][1])) / denom
                if gap <= gap_max and perp <= perp_max:
                    best_j = j
                    break

            if best_j != -1:

                pts = np.vstack([base[0], base[1], Ls[best_j][0], Ls[best_j][1]])
                t = pts @ ui
                p_start = pts[np.argmin(t)]
                p_end = pts[np.argmax(t)]
                base = (p_start, p_end)
                used[best_j] = True
                ui = unit(base)
                ai = np.arctan2(ui[1], ui[0])
                merged = True

        used[i] = True
        out.append((base[0].tolist(), base[1].tolist()))
    return out



def _collect_short_lines_with_prob(pred_cls, pred_reg, grid_size_x, grid_size_y, max_d, conf_thresh):

    H, W, N, _ = pred_cls.shape
    shorts = []
    for i in range(H):
        for j in range(W):
            for n in range(N):
                p1 = _stable_softmax_prob_1d(pred_cls[i, j, n])
                if p1 < conf_thresh: continue
                norm_d, sin_t, cos_t = pred_reg[i, j, n]
                d, theta = denormalize_d_theta_vector(norm_d, sin_t, cos_t, max_d)
                cx, cy = convert_grid_to_global_coords(i, j, grid_size_x, grid_size_y)
                seg = local_polar_to_line_segment(d, theta, cx, cy, grid_size_x, grid_size_y)
                if seg: shorts.append({'seg': seg, 'prob': p1})
    return shorts


def _score_and_filter_clusters(clusters_of_shorts, grid_size,
                               min_support, score_thresh):

    merged_candidates = []
    for cluster in clusters_of_shorts:
        if not cluster: continue

        # Evidence 1: Number of supports
        support_count = len(cluster)
        if support_count < min_support: continue

        lines_only = [item['seg'] for item in cluster]
        merged_line = lines_only[0] if len(lines_only) == 1 else merge_line_cluster_pca(lines_only)
        if not merged_line: continue

        # Evidence 2: Average confidence
        probs = [item['prob'] for item in cluster]
        mean_prob = np.mean(probs)

        # Evidence 3: Continuity/Density
        line_len = np.linalg.norm(np.array(merged_line[1]) - np.array(merged_line[0]))
        expected_cells = max(1.0, line_len / grid_size)
        density = support_count / expected_cells

        density_score = math.log1p(density)
        final_score = mean_prob * density_score

        if final_score >= score_thresh:
            merged_candidates.append({'line': merged_line, 'score': final_score})

    return merged_candidates


def convert_predictions_to_lines(
        pred_cls, pred_reg, image_np, grid_size_x, grid_size_y, max_d,
        conf_thresh=0.4,
        merge=True,
        angle_thresh=2,
        dist_thresh_endpoint=30.0,
        dist_thresh_collinear=3.0,
        # 【新增】最终证据筛选门槛
        score_thresh=0.4,
        min_support=3,
        stitch_angle=3.0,
        stitch_gap=24.0,
        stitch_perp=3.0,
        min_length=10,
        nms_iou_thresh=0.70,
        nms_buffer_px=2.0
):
    if not merge:
        shorts_with_prob = _collect_short_lines_with_prob(pred_cls, pred_reg, grid_size_x, grid_size_y, max_d,
                                                          conf_thresh)
        return [line for line, prob in shorts_with_prob]

    # 1.  Extract all possible short line segments
    shorts = _collect_short_lines_with_prob(pred_cls, pred_reg, grid_size_x, grid_size_y, max_d, conf_thresh)
    if not shorts: return []
    if not merge: return [s['seg'] for s in shorts]

    # 2. clustering
    lines_only = [s['seg'] for s in shorts]
    clusters_of_lines = cluster_lines_v3(lines_only, angle_thresh_deg=angle_thresh,
                                         dist_thresh_endpoint=dist_thresh_endpoint,
                                         dist_thresh_collinear=dist_thresh_collinear)

    line_map = {item['seg']: item for item in shorts}
    clusters_of_shorts = [[line_map.get(line) for line in cluster if line in line_map] for cluster in
                          clusters_of_lines]

    # 3.  Merge, score, filter
    merged_candidates = _score_and_filter_clusters(clusters_of_shorts, grid_size_x,
                                                   min_support, score_thresh)
    if not merged_candidates: return []

    # 4. suture
    lines_to_stitch = [c['line'] for c in merged_candidates]
    stitched = stitch_collinear_lines(lines_to_stitch, angle_thresh_deg=stitch_angle,
                                      gap_max=stitch_gap, perp_max=stitch_perp)

    # 5. NMS
    final_scored_lines = []
    for s_line in stitched:
        constituent_scores = [c['score'] for c in merged_candidates if _line_iou(c['line'], s_line) > 0.5]
        final_score = max(constituent_scores) if constituent_scores else 0.0
        final_scored_lines.append({'line': s_line, 'score': final_score})

    final_scored_lines.sort(key=lambda x: x['score'], reverse=True)

    final_lines, suppressed = [], [False] * len(final_scored_lines)
    for i in range(len(final_scored_lines)):
        if suppressed[i]: continue
        final_lines.append(final_scored_lines[i]['line'])
        for j in range(i + 1, len(final_scored_lines)):
            if suppressed[j]: continue
            if _line_iou(final_scored_lines[i]['line'], final_scored_lines[j]['line'],
                         buf=nms_buffer_px) > nms_iou_thresh:
                suppressed[j] = True

    return [line for line in final_lines if np.linalg.norm(np.array(line[1]) - np.array(line[0])) >= min_length]


def _draw_lines(vis, lines, color=(255, 255, 255), thickness=3, aa=True):

    import numpy as _np
    import cv2 as _cv2
    lt = _cv2.LINE_AA if aa else _cv2.LINE_8
    for (p1, p2) in lines:
        x1, y1 = map(int, _np.rint(p1)); x2, y2 = map(int, _np.rint(p2))
        _cv2.line(vis, (x1, y1), (x2, y2), color, thickness, lt)
    return vis


def _rasterize_lines_to_mask(lines, H, W, thickness=3,
                             blur_ksize=9, blur_sigma=1.5,
                             otsu=True, fixed_thresh=None):

    import numpy as _np
    import cv2 as _cv2

    heat = _np.zeros((H, W), _np.uint8)
    tmp = _np.zeros_like(heat)

    for (p1, p2) in lines:
        x1, y1 = map(int, _np.rint(p1)); x2, y2 = map(int, _np.rint(p2))
        tmp[:] = 0
        _cv2.line(tmp, (x1, y1), (x2, y2), 255, thickness, _cv2.LINE_8)
        heat = _np.maximum(heat, tmp)

    if blur_ksize and blur_ksize >= 3:
        heat = _cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), blur_sigma)

    if otsu:
        _, binary = _cv2.threshold(heat, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
    else:
        thr = 128 if fixed_thresh is None else int(255.0 * float(fixed_thresh))
        _, binary = _cv2.threshold(heat, thr, 255, _cv2.THRESH_BINARY)

    return binary


def lines_to_vis_and_mask(final_lines, H, W,
                          vis_base=None,
                          render_color=(255, 255, 255),
                          render_thickness=3,
                          aa=True,
                          blur_ksize=9, blur_sigma=1.5,
                          otsu=True, fixed_thresh=None):

    import numpy as _np

    vis = (vis_base.copy() if vis_base is not None else _np.zeros((H, W, 3), _np.uint8))
    vis = _draw_lines(vis, final_lines, color=render_color, thickness=render_thickness, aa=aa)
    mask = _rasterize_lines_to_mask(final_lines, H, W, thickness=render_thickness,
                                    blur_ksize=blur_ksize, blur_sigma=blur_sigma,
                                    otsu=otsu, fixed_thresh=fixed_thresh)
    return vis, mask