# utils/postprocess.py
import numpy as np
import math
import cv2
import time

def _stable_softmax_prob_1d(logits):
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    exps = np.exp(x - m)
    denom = exps.sum() + 1e-12
    return float(exps[1] / denom)

def _denormalize_d_theta(norm_d, sin_theta, cos_theta, max_d):
    d = float(norm_d) * float(max_d)
    theta = float(np.arctan2(sin_theta, cos_theta))
    n = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    return d, theta, n

def _grid_cell_center(i, j, grid_size_x, grid_size_y):
    cx = (j + 0.5) * grid_size_x
    cy = (i + 0.5) * grid_size_y
    return cx, cy

def _clip_line_to_cell(d_global, n, i, j, grid_size_x, grid_size_y, H, W, eps=1e-6):
    nx, ny = float(n[0]), float(n[1])
    x0 = j * grid_size_x
    x1 = (j + 1) * grid_size_x
    y0 = i * grid_size_y
    y1 = (i + 1) * grid_size_y

    xs, ys = [], []

    if abs(ny) > eps:
        y = (d_global - nx * x0) / ny
        if y0 - eps <= y <= y1 + eps:
            xs.append(x0); ys.append(y)
        y = (d_global - nx * x1) / ny
        if y0 - eps <= y <= y1 + eps:
            xs.append(x1); ys.append(y)

    if abs(nx) > eps:
        x = (d_global - ny * y0) / nx
        if x0 - eps <= x <= x1 + eps:
            xs.append(x); ys.append(y0)
        x = (d_global - ny * y1) / nx
        if x0 - eps <= x <= x1 + eps:
            xs.append(x); ys.append(y1)

    pts = []
    for xx, yy in zip(xs, ys):
        if -eps <= xx <= W + eps and -eps <= yy <= H + eps:
            pts.append((float(xx), float(yy)))

    uniq = []
    for p in pts:
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 1e-3 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None

    max_d2, best = -1.0, None
    for a in range(len(uniq)):
        for b in range(a+1, len(uniq)):
            dx = uniq[a][0] - uniq[b][0]
            dy = uniq[a][1] - uniq[b][1]
            d2 = dx*dx + dy*dy
            if d2 > max_d2:
                max_d2 = d2; best = (uniq[a], uniq[b])
    return best

def _segment_length(p1, p2):
    return float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))

def _segment_angle(p1, p2):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    ang = math.atan2(dy, dx)
    if ang < 0: ang += math.pi
    if ang >= math.pi: ang -= math.pi
    return ang

def _angle_diff_rad(a, b):
    d = abs(a - b)
    if d > math.pi: d = 2*math.pi - d
    if d > math.pi/2: d = math.pi - d
    return d


def _collect_short_segments_ttpla(pred_cls, pred_reg,
                                  grid_size_x, grid_size_y, max_d,
                                  H, W,
                                  conf_thresh=0.35,
                                  min_seg_len=3.0,
                                  hard_min_len=4.0,
                                  soft_min_len=0.0,
                                  soft_prob=0.0):

    G_h, G_w, N, _ = pred_cls.shape
    segs = []
    sid = 0
    length_thresh = max(hard_min_len, min_seg_len)

    for i in range(G_h):
        for j in range(G_w):
            for k in range(N):
                logits = pred_cls[i, j, k]
                p1_prob = _stable_softmax_prob_1d(logits)
                if p1_prob < conf_thresh:
                    continue

                norm_d, s, c = pred_reg[i, j, k]
                d, _, n = _denormalize_d_theta(norm_d, s, c, max_d)
                cx, cy = _grid_cell_center(i, j, grid_size_x, grid_size_y)
                d_global = d + (cx * n[0] + cy * n[1])

                seg = _clip_line_to_cell(d_global, n, i, j,
                                         grid_size_x, grid_size_y, H, W)
                if seg is None:
                    continue

                p1_xy, p2_xy = seg
                L = _segment_length(p1_xy, p2_xy)

                if L < length_thresh:
                    continue

                ang_t = _segment_angle(p1_xy, p2_xy)
                segs.append({
                    'id': sid,
                    'p1': p1_xy,
                    'p2': p2_xy,
                    'prob': p1_prob,
                    'len': L,
                    'ang': ang_t
                })
                sid += 1
    return segs


def _cluster_segments_by_connectivity(segments,
                                      join_dist=3.0,
                                      join_angle_deg=15.0):
    if not segments:
        return []
    n = len(segments)
    join_angle_rad = math.radians(join_angle_deg)

    endpoints = []
    angles = []
    for seg in segments:
        endpoints.append([np.array(seg['p1'], dtype=np.float64),
                          np.array(seg['p2'], dtype=np.float64)])
        angles.append(seg['ang'])

    adj = [[] for _ in range(n)]
    for i in range(n):
        p1_i, p2_i = endpoints[i]
        ang_i = angles[i]
        for j in range(i+1, n):
            p1_j, p2_j = endpoints[j]
            ang_j = angles[j]
            if _angle_diff_rad(ang_i, ang_j) > join_angle_rad:
                continue
            d_candidates = [
                np.linalg.norm(p1_i - p1_j),
                np.linalg.norm(p1_i - p2_j),
                np.linalg.norm(p2_i - p1_j),
                np.linalg.norm(p2_i - p2_j),
            ]
            dmin = min(d_candidates)
            if dmin > join_dist:
                continue
            adj[i].append(j); adj[j].append(i)

    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        q = [i]; visited[i] = True
        cur = []
        while q:
            u = q.pop()
            cur.append(segments[u])
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        clusters.append(cur)
    return clusters


def _trace_backbone_tracks(segments,
                           prob_hi=0.70,
                           min_seed_len=14,
                           max_turn_deg=10,
                           max_ortho_off=2.5,
                           max_step=10):
    n = len(segments)
    used = [False]*n
    tracks = []
    turn = math.radians(max_turn_deg)

    ends = [(np.array(s['p1'],np.float64), np.array(s['p2'],np.float64)) for s in segments]
    angs = [s['ang'] for s in segments]
    probs= [s['prob'] for s in segments]
    lens = [s['len'] for s in segments]

    neigh = [[] for _ in range(n)]
    for i in range(n):
        p1i,p2i = ends[i]
        for j in range(i+1,n):
            p1j,p2j = ends[j]
            dmin = min(np.linalg.norm(p1i-p1j), np.linalg.norm(p1i-p2j),
                       np.linalg.norm(p2i-p1j), np.linalg.norm(p2i-p2j))
            if dmin <= max_step:
                neigh[i].append(j); neigh[j].append(i)

    def _extend(cur_idx, cur_dir, cur_pt):
        chain=[]; visited=set([cur_idx])
        anchor_dir = np.array([math.cos(cur_dir), math.sin(cur_dir)], np.float64)
        while True:
            best=-1; best_j=-1; best_end=None
            for j in neigh[cur_idx]:
                if j in visited: continue
                pj1,pj2 = ends[j]
                attach, other = (pj1,pj2) if np.linalg.norm(pj1-cur_pt) <= np.linalg.norm(pj2-cur_pt) else (pj2,pj1)
                dang = min(abs(angs[j]-cur_dir), abs(angs[j]-cur_dir+math.pi))
                if dang > turn: continue
                v = other - cur_pt
                ortho = abs(v[0]*(-anchor_dir[1]) + v[1]*anchor_dir[0])
                if ortho > max_ortho_off: continue
                score = probs[j]*lens[j] / (1.0 + ortho + dang*10.0)
                if score > best:
                    best = score; best_j=j; best_end=other
            if best_j==-1: break
            chain.append(best_j); visited.add(best_j)
            cur_idx = best_j; cur_pt = best_end; cur_dir = angs[best_j]
            anchor_dir = np.array([math.cos(cur_dir), math.sin(cur_dir)], np.float64)
        return chain

    for i in range(n):
        if used[i]: continue
        if probs[i] < prob_hi or lens[i] < min_seed_len:
            continue
        p1,p2 = ends[i]; dir0 = angs[i]
        fw = _extend(i, dir0, p2); bw = _extend(i, dir0, p1)
        track = list(reversed(bw)) + [i] + fw
        if len(track) >= 3:
            for j in track:
                used[j]=True
                segments[j]['protected']=True
            tracks.append(track)
    return tracks


def _prune_isolated_shorts(clusters, grid_size_x, grid_size_y,
                           max_cluster_size=4,
                           max_len_px=30.0,
                           max_total_len_px=40.0,
                           max_cells_span=2,
                           max_diameter_px=40.0):
    kept = []
    for cl in clusters:
        if any(s.get('protected', False) for s in cl):
            kept.append(cl); continue

        if len(cl) <= max_cluster_size:
            max_len = max(s['len'] for s in cl)
            total_len = sum(s['len'] for s in cl)
            pts=[]
            for s in cl: pts += [np.array(s['p1']), np.array(s['p2'])]
            if len(pts) >= 2:
                P = np.stack(pts,0)
                dmax = float(np.max(
                    np.linalg.norm(P[None,:,:]-P[:,None,:], axis=-1)
                ))
            else:
                dmax = 0.0
            cells=set()
            for s in cl:
                for p in (s['p1'], s['p2']):
                    cells.add((int(p[0]//grid_size_x), int(p[1]//grid_size_y)))
            cell_span=len(cells)

            if (max_len < max_len_px) and (total_len < max_total_len_px) \
               and (cell_span <= max_cells_span) and (dmax <= max_diameter_px):
                continue
        kept.append(cl)
    return kept


def _filter_cluster_geometry(cluster_indices, segments,
                             grid_size_x, grid_size_y,
                             lambda_min=25.0,
                             min_cells=5,
                             min_len_ratio=0.15,
                             max_ang_q95_deg=10.0,
                             max_hole_ratio=0.30,
                             H=None, W=None):
    if any(segments[idx].get('protected', False) for idx in cluster_indices):
        return True

    n_points = 2 * len(cluster_indices)
    if n_points < 8:
        return False

    pts=[]; angles=[]; total_len=0.0
    cells=set()
    for idx in cluster_indices:
        p1=np.array(segments[idx]['p1']); p2=np.array(segments[idx]['p2'])
        pts.append(p1); pts.append(p2)
        angles.append(segments[idx]['ang'])
        total_len += segments[idx]['len']
        cells.add((int(p1[0]//grid_size_x), int(p1[1]//grid_size_y)))
        cells.add((int(p2[0]//grid_size_x), int(p2[1]//grid_size_y)))
    true_cell_span = len(cells)

    if H is None or W is None:
        H = int(max(p[1] for p in pts)+1); W = int(max(p[0] for p in pts)+1)
    diag = math.hypot(H, W)

    P = np.stack(pts,0)
    Pm = P.mean(0, keepdims=True)
    C = (P-Pm).T @ (P-Pm) / max(len(P)-1,1)
    evals,evecs = np.linalg.eig(C)
    ev = sorted(np.real(evals), reverse=True)
    lin = (ev[0]+1e-6)/(ev[1]+1e-6)
    if lin < lambda_min:
        return False

    if not (true_cell_span >= min_cells or total_len >= min_len_ratio*diag):
        return False

    ang = np.array(angles)
    med = float(np.median(ang))
    d = np.minimum(np.abs(ang-med), np.pi-np.abs(ang-med))
    if np.percentile(d, 95) > math.radians(max_ang_q95_deg):
        return False

    v = np.real(evecs[:, np.argmax(np.real(evals))]); v = v/np.linalg.norm(v)
    t = (P - Pm[0]) @ v
    tmin,tmax = float(t.min()), float(t.max())
    bins = max(10, int((tmax-tmin)/max(grid_size_x,grid_size_y)))
    hist,_ = np.histogram(t, bins=bins)
    holes = np.sum(hist==0)
    if holes/float(bins) > max_hole_ratio:
        return False
    return True

def _fit_cluster_loess_tube(cluster_indices, segments,
                            window_px=18,
                            tube_px=3.5,
                            step_px=6,
                            min_points=12,
                            straight_ang_q95_deg=4.0):
    P=[]; W=[]
    for idx in cluster_indices:
        s=segments[idx]
        p1=np.array(s['p1'],np.float64); p2=np.array(s['p2'],np.float64)
        P.append(p1); P.append(p2)
        w=s['prob']*max(s['len'],1.0)
        W.append(w); W.append(w)
    P=np.stack(P,0); W=np.array(W,dtype=np.float64)

    Pm=P.mean(0,keepdims=True)
    C=(P-Pm).T @ (P-Pm) / max(len(P)-1,1)
    evals,evecs=np.linalg.eig(C)
    v=np.real(evecs[:,np.argmax(np.real(evals))]); v=v/np.linalg.norm(v)
    u=np.array([-v[1], v[0]])
    # 是否近似直线
    angs=[segments[i]['ang'] for i in cluster_indices]
    med=float(np.median(angs))
    d = np.minimum(np.abs(np.array(angs)-med), np.pi-np.abs(np.array(angs)-med))
    if np.percentile(d,95) <= math.radians(straight_ang_q95_deg):
        t=(P-Pm[0])@v; a=P[np.argmin(t)]; b=P[np.argmax(t)]
        return [(tuple(a), tuple(b))]

    t=(P-Pm[0])@v
    order=np.argsort(t); P=P[order]; W=W[order]; t=t[order]
    if len(P) < min_points:
        a=P[0]; b=P[-1]; return [(tuple(a),tuple(b))]

    tmin,tmax=float(t.min()), float(t.max())
    num = max(2, int((tmax - tmin) / max(step_px,1)))
    ts = np.linspace(tmin, tmax, num=num)

    def _tricube(d):
        a = np.clip(1 - np.power(np.abs(d), 3), 0, 1)
        return np.power(a, 3)

    curve=[]
    for tc in ts:
        dwin = (t - tc) / max(window_px, 1e-6)
        kw = _tricube(dwin) * W
        if kw.sum() < 1e-6:
            alpha = (tc - tmin) / max(tmax-tmin,1e-6)
            p = (1-alpha)*P[0] + alpha*P[-1]
            curve.append(p); continue
        T = np.stack([np.ones_like(t), (t - tc)], axis=1)
        Kw = np.diag(kw)
        beta_x = np.linalg.pinv(T.T @ Kw @ T) @ (T.T @ Kw @ P[:,0])
        beta_y = np.linalg.pinv(T.T @ Kw @ T) @ (T.T @ Kw @ P[:,1])
        p = np.array([beta_x[0], beta_y[0]], np.float64)

        tproj = (p - Pm[0]) @ v
        p_axis = Pm[0] + tproj * v
        ortho = (p - p_axis) @ u
        if abs(ortho) > tube_px:
            p = p_axis + np.sign(ortho) * tube_px * u
        curve.append(p)
    curve = np.stack(curve,0)

    segs=[]
    for i in range(len(curve)-1):
        a=tuple(curve[i]); b=tuple(curve[i+1])
        segs.append((a,b))
    return segs

def convert_predictions_to_lines_ttpla(
        pred_cls, pred_reg, image_np,
        grid_size_x, grid_size_y, max_d,
        conf_thresh=0.35,
        min_seg_len=3.0,
        join_dist=0.0,
        join_angle_deg=10.0,
        min_cluster_support=2,
        long_seg_thresh=12.0,
        use_loess=True
):
    H, W = image_np.shape[:2]

    # 1) Collect small segments (with gentle hard cuts)
    segs = _collect_short_segments_ttpla(
        pred_cls, pred_reg,
        grid_size_x, grid_size_y, max_d,
        H, W,
        conf_thresh=conf_thresh,
        min_seg_len=min_seg_len,
    )
    if not segs:
        return []

    # 2') Protect the main trunk
    _trace_backbone_tracks(segs,
                           prob_hi=0.70,
                           min_seed_len=14,
                           max_turn_deg=10,
                           max_ortho_off=2.5,
                           max_step=10)

    # 2'') Clustering
    if not join_dist or join_dist <= 0:
        join_dist = 0.45 * min(grid_size_x, grid_size_y)  # 自适应像素
    clusters = _cluster_segments_by_connectivity(
        segs, join_dist=join_dist, join_angle_deg=join_angle_deg
    )

    # Isolated short clusters are treated uniformly
    clusters = _prune_isolated_shorts(
        clusters, grid_size_x, grid_size_y,
        max_cluster_size=4,
        max_len_px=30.0,
        max_total_len_px=40.0,
        max_cells_span=2,
        max_diameter_px=40.0
    )

    # 3) Cluster-level geometric filtering (preserving straight-through)
    kept = []
    for cluster in clusters:
        idxs = [s['id'] for s in cluster]
        if _filter_cluster_geometry(
                idxs, segs, grid_size_x, grid_size_y,
                lambda_min=25.0, min_cells=5,
                min_len_ratio=0.15, max_ang_q95_deg=10.0,
                max_hole_ratio=0.30, H=H, W=W):
            kept.append(idxs)

    # 4) Soft merging (LOESS)
    final_segments = []
    for idxs in kept:
        merged = _fit_cluster_loess_tube(
            idxs, segs,
            window_px=int(1.0 * max(grid_size_x, grid_size_y)),
            tube_px=10 if use_loess else 3.5, # tube_px 用来限制 LOESS 曲线允许偏离主轴多少像素
            step_px=int(0.9 * min(grid_size_x, grid_size_y)), #取点采样步长
            min_points=12,
            straight_ang_q95_deg=2.0
        )
        final_segments.extend(merged)

    return final_segments

def _draw_lines(vis, lines, color=(0, 0, 255), thickness=2, aa=True):
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    for (p1, p2) in lines:
        x1, y1 = map(int, np.rint(p1))
        x2, y2 = map(int, np.rint(p2))
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness, lt)
    return vis

def _rasterize_lines_to_mask(lines, H, W, thickness=2,
                             blur_ksize=7, blur_sigma=1.0,
                             otsu=True, fixed_thresh=None):
    heat = np.zeros((H, W), np.uint8)
    for (p1, p2) in lines:
        x1, y1 = map(int, np.rint(p1))
        x2, y2 = map(int, np.rint(p2))
        cv2.line(heat, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)

    if blur_ksize is not None and blur_ksize > 1:
        heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), blur_sigma)

    if fixed_thresh is not None:
        _, mask = cv2.threshold(heat, fixed_thresh, 255, cv2.THRESH_BINARY)
    elif otsu:
        _, mask = cv2.threshold(heat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(heat, 127, 255, cv2.THRESH_BINARY)
    return mask

def lines_to_vis_and_mask_ttpla(final_lines, image_np,
                                vis_base=None,
                                render_color=(0, 0, 255),
                                render_thickness=2,
                                aa=True,
                                blur_ksize=7, blur_sigma=1.0,
                                otsu=True, fixed_thresh=None):
    H, W = image_np.shape[:2]
    vis = image_np.copy() if vis_base is None else vis_base.copy()
    vis = _draw_lines(vis, final_lines, color=render_color,
                      thickness=render_thickness, aa=aa)
    mask = _rasterize_lines_to_mask(
        final_lines, H, W, thickness=render_thickness,
        blur_ksize=blur_ksize, blur_sigma=blur_sigma,
        otsu=otsu, fixed_thresh=fixed_thresh
    )
    return vis, mask
