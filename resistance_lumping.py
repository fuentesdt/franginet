"""
resistance_lumping.py — Hagen-Poiseuille resistance lumping on a 1-D skeleton
                        centerline mesh.

Reads a VTP produced by skelcenterline.py, derives vessel radii, solves the
sparse conductance system K·p = q, and writes the pressure / flow / radius
solution back to a new VTP.

Radius sources (pick one)
-------------------------
  --label LABEL.nii[.gz]   Label NIfTI; the distance transform of the binary
                            vessel mask (label == --label-val) is computed
                            internally with scipy.ndimage.distance_transform_edt
                            using the true anisotropic voxel spacing.
                            This mirrors what resistanceLumping.m does:
                              binary = label_vol == label_val
                              dt_mm  = bwdist(~binary) * mean(vox)

  --dt DT.nii[.gz]         Pre-computed distance-transform NIfTI (radii in mm).
                            Use this when you have already saved the DT volume.

  --default-radius MM       Uniform radius in mm (fallback when neither
                            --label nor --dt is given; default 1.0 mm).

Phantom gap edges
-----------------
  mst  (default) — one phantom edge per MST-minimal inter-component connection.
                   Guarantees connectivity with the fewest, shortest phantom
                   edges so spurious parallel conductance paths are avoided.

  knn  (fallback) — all cross-component leaf pairs within --gap-max.
                    Matches MATLAB resistanceLumping.m behaviour; can produce
                    redundant phantom paths when many leaf nodes are nearby.

Usage
-----
    python resistance_lumping.py <centerline.vtp> [output.vtp]
        --label  vessel_label.nii.gz          # recommended
      OR
        --dt     dt_mm.nii.gz                 # if DT already computed
      OR
        --default-radius 1.0                  # fallback

        [--label-val 2]
        [--p-in 100] [--p-out 5] [--mu 3.5e-3]
        [--gap-max 15] [--alpha 10] [--gap-mode mst|knn]

If output path is omitted, <stem>_pressure.vtp is written beside the input VTP.

Dependencies
------------
    pip install nibabel vtk numpy scipy
"""

import argparse
import os
import sys
import types

import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree


def log(msg):
    print(f"[resistance_lumping] {msg}", flush=True)


# ── VTP I/O ──────────────────────────────────────────────────────────────────

def read_fcsv_seed(fcsv_path):
    """Read the first fiducial from a Slicer .fcsv file; return [x,y,z] RAS mm."""
    with open(fcsv_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            lps = [float(parts[1]), float(parts[2]), float(parts[3])]
            return [-lps[0], -lps[1], lps[2]]   # LPS → RAS
    return None


def load_vtp(path):
    """
    Read a VTP file.  Returns
      nodes  : N×3 float64 world coordinates (mm)
      edges  : E×2 int64  node-index pairs (0-based)
      pd     : vtkPolyData (kept alive for write-back)
    Handles both the legacy SetCells format from skelcenterline.py and
    the VTK-9+ offset/connectivity format.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    pd = reader.GetOutput()

    nodes = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)

    lines = pd.GetLines()
    n_cells = lines.GetNumberOfCells()
    if n_cells == 0:
        return nodes, np.zeros((0, 2), dtype=np.int64), pd

    # VTK 9+ stores offsets + connectivity separately
    if hasattr(lines, 'GetConnectivityArray'):
        conn = vtk_to_numpy(lines.GetConnectivityArray()).astype(np.int64)
        edges = conn.reshape(-1, 2)          # all 2-point cells
    else:
        # Legacy: [2, src, dst, 2, src, dst, ...]
        raw   = vtk_to_numpy(lines.GetData()).astype(np.int64)
        edges = raw.reshape(-1, 3)[:, 1:]

    return nodes, edges, pd


def _build_cell_array(edges):
    """Build a vtkCellArray from an E×2 edge array using the legacy format."""
    E = len(edges)
    if E == 0:
        return vtk.vtkCellArray()
    legacy       = np.empty(3 * E, dtype=np.int64)
    legacy[0::3] = 2
    legacy[1::3] = edges[:, 0]
    legacy[2::3] = edges[:, 1]
    id_arr = numpy_to_vtk(legacy, deep=True, array_type=vtk.VTK_ID_TYPE)
    ca     = vtk.vtkCellArray()
    ca.SetCells(E, id_arr)
    return ca


def write_vtp(path, pd, nodes, edges, pressure_mmhg, radii,
              G_real, flows_mm3s, r_edge, L_edge,
              ph_edges, G_ph, pressure_pa):
    """
    Append solution arrays to pd (modifies in place) and write VTP.

    Point data  : pressure_mmhg, radius_mm
    Cell data   : flow_mm3s, conductance_SI, radius_mm_edge, length_mm,
                  is_phantom
    """
    n_real = len(edges)
    n_ph   = len(ph_edges)

    # Rebuild cell array to include phantom edges
    all_edges = np.vstack([edges, ph_edges]) if n_ph > 0 else edges
    pd.SetLines(_build_cell_array(all_edges))

    def _add_pt(name, data, vtk_type, active=False):
        arr = numpy_to_vtk(np.asarray(data, dtype=np.float64),
                           deep=True, array_type=vtk_type)
        arr.SetName(name)
        pd.GetPointData().AddArray(arr)
        if active:
            pd.GetPointData().SetActiveScalars(name)

    _add_pt('pressure_mmhg', pressure_mmhg, vtk.VTK_DOUBLE, active=True)
    _add_pt('radius_mm',     radii,         vtk.VTK_DOUBLE)

    def _add_cell(name, real_data, ph_data, vtk_type, active=False):
        combined = (np.concatenate([real_data, ph_data])
                    if n_ph > 0 else real_data)
        arr = numpy_to_vtk(np.asarray(combined, dtype=np.float64),
                           deep=True, array_type=vtk_type)
        arr.SetName(name)
        pd.GetCellData().AddArray(arr)
        if active:
            pd.GetCellData().SetActiveScalars(name)

    # Phantom edge physical quantities
    if n_ph > 0:
        ph_r_e  = (radii[ph_edges[:, 0]] + radii[ph_edges[:, 1]]) / 2.0
        ph_L_e  = np.linalg.norm(
                      nodes[ph_edges[:, 0]] - nodes[ph_edges[:, 1]], axis=1)
        ph_flow = (G_ph * (pressure_pa[ph_edges[:, 0]]
                           - pressure_pa[ph_edges[:, 1]]) * 1e9)
    else:
        ph_r_e  = np.empty(0)
        ph_L_e  = np.empty(0)
        ph_flow = np.empty(0)

    _add_cell('flow_mm3s',      flows_mm3s, ph_flow,          vtk.VTK_DOUBLE, active=True)
    _add_cell('conductance_SI', G_real,     G_ph,             vtk.VTK_DOUBLE)
    _add_cell('radius_mm_edge', r_edge,     ph_r_e,           vtk.VTK_DOUBLE)
    _add_cell('length_mm',      L_edge,     ph_L_e,           vtk.VTK_DOUBLE)

    is_ph = np.array([0] * n_real + [1] * n_ph, dtype=np.int8)
    ph_flag = numpy_to_vtk(is_ph, deep=True, array_type=vtk.VTK_SIGNED_CHAR)
    ph_flag.SetName('is_phantom')
    pd.GetCellData().AddArray(ph_flag)

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetDataModeToAscii()
    w.Write()


# ── Radius lookup ─────────────────────────────────────────────────────────────

def _lookup_volume(nodes, affine, vol):
    """
    Look up values from a 3-D volume at node world positions.
    Uses the NIfTI affine inverse to map world → nearest voxel.
    Returns a float64 array floored at 0.1 mm.
    """
    inv = np.linalg.inv(affine)
    N   = len(nodes)
    ijk = (inv @ np.hstack([nodes, np.ones((N, 1))]).T).T[:, :3]
    idx = np.clip(np.round(ijk).astype(int), 0,
                  np.array(vol.shape) - 1)
    r   = np.asarray(vol[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=np.float64)
    return np.maximum(r, 0.1)


def radii_from_label(nodes, label_path, label_val=2):
    """
    Compute the Euclidean distance transform of the vessel binary mask
    (label == label_val) and look up the result at each node position.

    Equivalent to MATLAB:
        binary = label_vol == label_val;
        dt_mm  = bwdist(~binary) .* mean(vox);

    Uses scipy.ndimage.distance_transform_edt with per-axis voxel spacing
    so anisotropic volumes are handled correctly.
    """
    nii     = nib.load(label_path)
    affine  = nii.affine.astype(np.float64)
    vol     = np.asarray(nii.dataobj)
    binary  = (vol == label_val)
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))  # mm per voxel, each axis
    log(f"  Voxel spacing : {spacing.round(4).tolist()} mm")
    log(f"  Vessel voxels : {binary.sum()}")
    dt_mm   = distance_transform_edt(~binary, sampling=spacing)
    radii   = _lookup_volume(nodes, affine, dt_mm)
    log(f"  Radius range  : {radii.min():.2f} – {radii.max():.2f} mm")
    return radii


def radii_from_dt(nodes, dt_path):
    """
    Load a pre-computed distance-transform NIfTI (radii in mm) and look up
    the value at each node world position.
    """
    nii    = nib.load(dt_path)
    affine = nii.affine.astype(np.float64)
    dt_mm  = np.asarray(nii.dataobj, dtype=np.float64)
    radii  = _lookup_volume(nodes, affine, dt_mm)
    log(f"  Radius range  : {radii.min():.2f} – {radii.max():.2f} mm")
    return radii


# ── Edge properties ───────────────────────────────────────────────────────────

def edge_properties(nodes, edges, radii, mu_pas):
    """
    Returns
      r_edge  : E  mean radius per edge (mm)
      L_edge  : E  Euclidean length per edge (mm)
      G_real  : E  Hagen-Poiseuille conductance (m³ Pa⁻¹ s⁻¹)
    """
    r_edge  = (radii[edges[:, 0]] + radii[edges[:, 1]]) / 2.0          # mm
    diff    = nodes[edges[:, 0]] - nodes[edges[:, 1]]
    L_edge  = np.maximum(np.linalg.norm(diff, axis=1), 1e-3)           # mm
    r_m     = r_edge * 1e-3
    L_m     = L_edge * 1e-3
    G_real  = np.pi * r_m ** 4 / (8.0 * mu_pas * L_m)
    return r_edge, L_edge, G_real


# ── Phantom edge detection ────────────────────────────────────────────────────

def _leaf_candidates(edges, comp_labels, n_comp, N):
    """
    For each component, return candidate gap-endpoint nodes.
    Prefers degree-1 nodes (skeleton tips); falls back to all component nodes
    if a component has no leaves (e.g. a single isolated node).
    """
    degree = np.bincount(edges.ravel(), minlength=N) if len(edges) > 0 \
             else np.zeros(N, dtype=np.int64)
    cands  = {}
    for c in range(n_comp):
        mask   = comp_labels == c
        cnodes = np.where(mask)[0]
        leaves = cnodes[degree[cnodes] <= 1]
        cands[c] = leaves if len(leaves) > 0 else cnodes
    return cands


def _conductance(radii, nodes, ni_arr, nj_arr, mu_pas, alpha):
    """Hagen-Poiseuille conductance for phantom edges (with alpha penalty)."""
    r_m = (radii[ni_arr] + radii[nj_arr]) / 2.0 * 1e-3
    L_m = np.maximum(
            np.linalg.norm(nodes[ni_arr] - nodes[nj_arr], axis=1), 1e-6) * 1e-3
    return np.pi * r_m ** 4 / (8.0 * mu_pas * alpha * L_m)


def phantom_edges_mst(nodes, edges, radii, comp_labels, n_comp,
                      gap_max_mm, mu_pas, alpha):
    """
    Algorithm 3 — MST inter-component phantom edges.

    Builds a global KDTree on all candidate (leaf) nodes, finds every
    cross-component pair within gap_max_mm, keeps only the closest pair per
    component-pair as the representative, then runs a minimum spanning tree on
    the resulting component-level distance graph.  The MST edges become phantom
    connections, guaranteeing one bridge per gap at minimum total phantom
    length.

    Components with no neighbour within gap_max_mm remain disconnected here;
    assemble_and_solve pins them to mean pressure.
    """
    if n_comp == 1:
        return np.zeros((0, 2), dtype=np.int64), np.array([])

    N     = len(nodes)
    cands = _leaf_candidates(edges, comp_labels, n_comp, N)

    all_cand_ids   = np.concatenate([cands[c] for c in range(n_comp)])
    all_cand_comps = comp_labels[all_cand_ids]

    tree  = KDTree(nodes[all_cand_ids])
    pairs = tree.query_pairs(r=gap_max_mm)          # pairs of indices into all_cand_ids

    best_dist  = {}   # (ci,cj) → float
    best_nodes = {}   # (ci,cj) → (ni, nj)

    for a, b in pairs:
        ci = int(all_cand_comps[a])
        cj = int(all_cand_comps[b])
        if ci == cj:
            continue
        key = (min(ci, cj), max(ci, cj))
        d   = float(np.linalg.norm(nodes[all_cand_ids[a]] - nodes[all_cand_ids[b]]))
        if key not in best_dist or d < best_dist[key]:
            best_dist[key]  = d
            ni = int(all_cand_ids[a])
            nj = int(all_cand_ids[b])
            best_nodes[key] = (ni, nj) if ci < cj else (nj, ni)

    if not best_dist:
        log(f"  No cross-component pairs within {gap_max_mm} mm — "
            f"isolated components will be pinned to mean pressure")
        return np.zeros((0, 2), dtype=np.int64), np.array([])

    keys   = list(best_dist.keys())
    ci_arr = np.array([k[0] for k in keys], dtype=np.int64)
    cj_arr = np.array([k[1] for k in keys], dtype=np.int64)
    dvals  = np.array([best_dist[k] for k in keys])

    sym_rows = np.concatenate([ci_arr, cj_arr])
    sym_cols = np.concatenate([cj_arr, ci_arr])
    sym_vals = np.concatenate([dvals,  dvals])
    comp_mat = csr_matrix((sym_vals, (sym_rows, sym_cols)),
                          shape=(n_comp, n_comp))

    mst     = minimum_spanning_tree(comp_mat)
    cx, cy  = mst.nonzero()

    ph_ni, ph_nj = [], []
    for ci, cj in zip(cx.tolist(), cy.tolist()):
        key = (min(ci, cj), max(ci, cj))
        if key in best_nodes:
            ni, nj = best_nodes[key]
            ph_ni.append(ni)
            ph_nj.append(nj)

    if not ph_ni:
        return np.zeros((0, 2), dtype=np.int64), np.array([])

    ph_edges = np.array(list(zip(ph_ni, ph_nj)), dtype=np.int64)
    G_ph     = _conductance(radii, nodes,
                             ph_edges[:, 0], ph_edges[:, 1], mu_pas, alpha)
    log(f"  MST phantom edges : {len(ph_ni)}")
    return ph_edges, G_ph


def phantom_edges_knn(nodes, edges, radii, comp_labels, n_comp,
                      gap_max_mm, mu_pas, alpha):
    """
    Algorithm 2 — k-NN all cross-component leaf pairs within gap_max_mm.

    Matches the MATLAB resistanceLumping.m behaviour: every leaf pair from
    different components closer than gap_max_mm receives a phantom edge.
    May produce redundant parallel phantom paths when many tips are nearby.
    """
    if n_comp == 1:
        return np.zeros((0, 2), dtype=np.int64), np.array([])

    N     = len(nodes)
    cands = _leaf_candidates(edges, comp_labels, n_comp, N)

    all_cand_ids   = np.concatenate([cands[c] for c in range(n_comp)])
    all_cand_comps = comp_labels[all_cand_ids]

    tree  = KDTree(nodes[all_cand_ids])
    pairs = tree.query_pairs(r=gap_max_mm)

    ph_ni, ph_nj = [], []
    for a, b in pairs:
        if all_cand_comps[a] != all_cand_comps[b]:
            ph_ni.append(int(all_cand_ids[a]))
            ph_nj.append(int(all_cand_ids[b]))

    if not ph_ni:
        log(f"  No cross-component pairs within {gap_max_mm} mm — "
            f"isolated components will be pinned to mean pressure")
        return np.zeros((0, 2), dtype=np.int64), np.array([])

    ph_edges = np.array(list(zip(ph_ni, ph_nj)), dtype=np.int64)
    G_ph     = _conductance(radii, nodes,
                             ph_edges[:, 0], ph_edges[:, 1], mu_pas, alpha)
    log(f"  k-NN phantom edges : {len(ph_ni)}")
    return ph_edges, G_ph


# ── Linear solve ──────────────────────────────────────────────────────────────

def assemble_and_solve(N, nodes, edges, G_real, ph_edges, G_ph,
                       radii, p_in_mmhg, p_out_mmhg, extra_bc=None):
    """
    1. Assembles the symmetric sparse Laplacian K from real + phantom edges.
    2. Auto-detects inlet (leaf with largest radius in real graph) and outlets
       (all other real-graph leaves).
    3. Detects floating components in the combined graph and pins each to mean
       pressure so K remains non-singular.
    4. Solves via Schur-complement elimination (partition free / constrained
       nodes), which is efficient for large sparse systems.

    Returns
      pressure_mmhg : N  nodal pressures
      pressure_pa   : N  (for phantom-flow computation in write_vtp)
      flows_mm3s    : E  signed volumetric flow per real edge
      inlet_node    : int
      outlet_nodes  : int array
    """
    p_in_pa   = p_in_mmhg  * 133.322
    p_out_pa  = p_out_mmhg * 133.322
    p_mean_pa = (p_in_pa + p_out_pa) / 2.0

    # Combined edge list
    has_ph   = len(ph_edges) > 0
    all_e    = np.vstack([edges, ph_edges]) if has_ph else edges
    all_G    = np.concatenate([G_real, G_ph]) if has_ph else G_real

    # Sparse Laplacian  K = Σ G_ij (e_i - e_j)(e_i - e_j)ᵀ
    e0, e1 = all_e[:, 0], all_e[:, 1]
    rows   = np.concatenate([e0, e1, e0, e1])
    cols   = np.concatenate([e0, e1, e1, e0])
    vals   = np.concatenate([all_G, all_G, -all_G, -all_G])
    K      = csr_matrix((vals, (rows, cols)), shape=(N, N))

    # BC detection: use real-graph degree only
    deg_real  = (np.bincount(edges.ravel(), minlength=N)
                 if len(edges) > 0 else np.zeros(N, dtype=np.int64))
    leaf_idx  = np.where(deg_real <= 1)[0]
    if len(leaf_idx) == 0:
        # No topological leaves — fall back to extremal z nodes
        leaf_idx = np.array([np.argmax(nodes[:, 2]),
                              np.argmin(nodes[:, 2])], dtype=np.int64)

    inlet_node   = int(leaf_idx[np.argmax(radii[leaf_idx])])
    outlet_nodes = leaf_idx[leaf_idx != inlet_node]

    bc_nodes_list = [int(inlet_node)] + [int(n) for n in outlet_nodes]
    bc_vals_list  = [p_in_pa] + [p_out_pa] * len(outlet_nodes)
    if extra_bc:
        existing = set(bc_nodes_list)
        for ni, pi_mmhg in extra_bc.items():
            ni = int(ni)
            if ni not in existing:
                bc_nodes_list.append(ni)
                bc_vals_list.append(float(pi_mmhg) * 133.322)
                existing.add(ni)
    bc_nodes = np.array(bc_nodes_list, dtype=int)
    bc_vals  = np.array(bc_vals_list)

    # Pin each floating component (no BC reachable in combined graph)
    adj_all = csr_matrix(
        (np.ones(2 * len(all_e)),
         (np.concatenate([all_e[:, 0], all_e[:, 1]]),
          np.concatenate([all_e[:, 1], all_e[:, 0]]))),
        shape=(N, N))
    n_comp_all, comp_all = connected_components(adj_all, directed=False)

    bc_set = set(bc_nodes.tolist())
    pinned = 0
    for c in range(n_comp_all):
        comp_nodes = np.where(comp_all == c)[0]
        if not any(int(n) in bc_set for n in comp_nodes):
            anchor = int(comp_nodes[0])
            bc_nodes = np.append(bc_nodes, anchor)
            bc_vals  = np.append(bc_vals,  p_mean_pa)
            bc_set.add(anchor)
            pinned  += 1
    if pinned:
        log(f"  Pinned {pinned} floating component(s) at "
            f"{p_mean_pa / 133.322:.1f} mmHg")

    # Schur-complement solve: partition into free (F) and constrained (C) nodes
    free_mask         = np.ones(N, bool)
    free_mask[bc_nodes] = False
    free_idx          = np.where(free_mask)[0]

    K_FF = K[np.ix_(free_idx, free_idx)]
    rhs  = -(K[free_idx, :][:, bc_nodes] @ bc_vals)

    # Regularise any zero-diagonal rows left after partitioning
    zero_rows = np.where(K_FF.diagonal() == 0)[0]
    if len(zero_rows) > 0:
        K_FF = K_FF.tolil()
        for r in zero_rows:
            K_FF[r, r] = 1.0
            rhs[r]     = p_mean_pa
        K_FF = K_FF.tocsr()
        log(f"  Regularised {len(zero_rows)} zero-diagonal free node(s)")

    p_free = spsolve(K_FF, rhs)

    pressure_pa          = np.zeros(N)
    pressure_pa[free_idx] = p_free
    pressure_pa[bc_nodes] = bc_vals
    pressure_mmhg        = pressure_pa / 133.322

    log(f"  Pressure range : {pressure_mmhg.min():.1f} – "
        f"{pressure_mmhg.max():.1f} mmHg")

    flows_mm3s = G_real * (pressure_pa[edges[:, 0]]
                           - pressure_pa[edges[:, 1]]) * 1e9
    log(f"  |Flow| range   : {np.abs(flows_mm3s).min():.3e} – "
        f"{np.abs(flows_mm3s).max():.3e} mm³/s")

    return pressure_mmhg, pressure_pa, flows_mm3s, inlet_node, outlet_nodes


# ── Main pipeline ─────────────────────────────────────────────────────────────

def solve(vtp_in, vtp_out, opts=None):
    """
    Full pipeline: load VTP → radii → conductances → phantom edges →
    sparse solve → write VTP with solution.

    opts is a namespace or dict with fields:
        label_path, label_val, dt_path,
        p_in_mmhg, p_out_mmhg, mu_pas, gap_max_mm, alpha, gap_mode,
        default_radius
    Missing fields fall back to the defaults defined in __main__.
    Radius source priority: label_path > dt_path > default_radius.
    """
    if opts is None:
        opts = types.SimpleNamespace()

    def _get(name, default):
        v = (getattr(opts, name, None)
             if not isinstance(opts, dict)
             else opts.get(name, None))
        return v if v is not None else default

    label_path  = _get('label_path',    None)
    label_val   = _get('label_val',        2)
    dt_path     = _get('dt_path',       None)
    p_in        = _get('p_in_mmhg',    100.0)
    p_out       = _get('p_out_mmhg',     5.0)
    mu          = _get('mu_pas',       3.5e-3)
    gap_max     = _get('gap_max_mm',    15.0)
    alpha       = _get('alpha',         10.0)
    gap_mode    = _get('gap_mode',     'mst')
    def_radius  = _get('default_radius', 1.0)
    seed_coord  = _get('seed_coord_mm', None)
    p_seed      = float(_get('p_seed_mmhg', 882.0))

    # ── 1. Load VTP ──────────────────────────────────────────────────────────
    log(f"[1/5] Loading VTP : {vtp_in}")
    nodes, edges, pd = load_vtp(vtp_in)
    N, E = len(nodes), len(edges)
    log(f"  Nodes: {N}  Edges: {E}")
    if E == 0:
        sys.exit("Error: VTP has no edges — run skelcenterline.py first")

    # Optional seed Dirichlet BC (e.g. label==5 injection site)
    extra_bc = None
    if seed_coord is not None:
        seed_pt  = np.array(seed_coord, dtype=np.float64)
        seed_idx = int(np.argmin(np.linalg.norm(nodes - seed_pt, axis=1)))
        extra_bc = {seed_idx: p_seed}
        log(f"  Seed node {seed_idx} at {nodes[seed_idx].round(2).tolist()} "
            f"pinned to {p_seed:.0f} mmHg")

    # ── 2. Radii ─────────────────────────────────────────────────────────────
    if label_path is not None:
        log(f"[2/5] Computing DT from label NIfTI (label={label_val}) : {label_path}")
        radii = radii_from_label(nodes, label_path, label_val=label_val)
    elif dt_path is not None:
        log(f"[2/5] Loading pre-computed DT : {dt_path}")
        radii = radii_from_dt(nodes, dt_path)
    else:
        log(f"[2/5] No radius source — using uniform {def_radius:.2f} mm")
        radii = np.full(N, float(def_radius))

    # ── 3. Edge conductances ─────────────────────────────────────────────────
    log("[3/5] Computing Hagen-Poiseuille conductances ...")
    r_edge, L_edge, G_real = edge_properties(nodes, edges, radii, mu)
    log(f"  Radius range (edges) : {r_edge.min():.2f} – {r_edge.max():.2f} mm")
    log(f"  Length range         : {L_edge.min():.3f} – {L_edge.max():.3f} mm")

    # ── 4. Phantom gap edges ─────────────────────────────────────────────────
    log(f"[4/5] Phantom edge detection  [mode={gap_mode}]  "
        f"gap_max={gap_max} mm  alpha={alpha} ...")

    adj_real = csr_matrix(
        (np.ones(2 * E),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(N, N))
    n_comp, comp_labels = connected_components(adj_real, directed=False)
    log(f"  Connected components (real graph) : {n_comp}")

    if gap_mode == 'knn':
        ph_edges, G_ph = phantom_edges_knn(
            nodes, edges, radii, comp_labels, n_comp, gap_max, mu, alpha)
    else:
        ph_edges, G_ph = phantom_edges_mst(
            nodes, edges, radii, comp_labels, n_comp, gap_max, mu, alpha)

    # ── 5. Solve ─────────────────────────────────────────────────────────────
    log("[5/5] Assembling K and solving ...")
    log(f"  p_in={p_in:.0f} mmHg  p_out={p_out:.0f} mmHg  "
        f"mu={mu:.2e} Pa·s")
    pressure_mmhg, pressure_pa, flows_mm3s, inlet, outlets = \
        assemble_and_solve(N, nodes, edges, G_real, ph_edges, G_ph,
                           radii, p_in, p_out, extra_bc=extra_bc)
    log(f"  Inlet node  : {inlet}")
    log(f"  Outlet nodes: {len(outlets)}")

    # ── Write VTP ────────────────────────────────────────────────────────────
    log(f"Writing  : {vtp_out}")
    write_vtp(vtp_out, pd, nodes, edges, pressure_mmhg, radii,
              G_real, flows_mm3s, r_edge, L_edge,
              ph_edges, G_ph, pressure_pa)
    log("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _stem(path, suffixes):
    for s in suffixes:
        if path.endswith(s):
            return path[:-len(s)]
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Hagen-Poiseuille resistance lumping on a skeleton VTP',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('vtp_in',  help='Input centerline .vtp (from skelcenterline.py)')
    ap.add_argument('vtp_out', nargs='?', default=None,
                    help='Output .vtp (default: <stem>_pressure.vtp)')

    src = ap.add_argument_group('radius source (pick one)')
    src.add_argument('--label', default=None, dest='label_path',
                     metavar='LABEL.nii[.gz]',
                     help='Label NIfTI; DT computed internally from vessel mask '
                          '(recommended — matches resistanceLumping.m)')
    src.add_argument('--label-val', type=int, default=2, dest='label_val',
                     metavar='INT',
                     help='Label value for vessel voxels (default 2)')
    src.add_argument('--dt', default=None, dest='dt_path',
                     metavar='DT.nii[.gz]',
                     help='Pre-computed distance-transform NIfTI (radii in mm)')
    src.add_argument('--default-radius', type=float, default=1.0,
                     dest='default_radius', metavar='MM',
                     help='Uniform radius [mm] when --label / --dt omitted (default 1.0)')

    ph = ap.add_argument_group('phantom gap edges')
    ph.add_argument('--gap-max', type=float, default=15.0, dest='gap_max_mm',
                    metavar='MM', help='Max gap distance [mm] (default 15)')
    ph.add_argument('--alpha',   type=float, default=10.0,
                    metavar='X',  help='Phantom conductance penalty (default 10)')
    ph.add_argument('--gap-mode', choices=['mst', 'knn'], default='mst',
                    dest='gap_mode',
                    help='mst = MST minimal bridges (default);  knn = all pairs')

    ph2 = ap.add_argument_group('solver')
    ph2.add_argument('--p-in',  type=float, default=100.0, dest='p_in_mmhg',
                     metavar='MMHG', help='Inlet pressure  [mmHg] (default 100)')
    ph2.add_argument('--p-out', type=float, default=5.0,   dest='p_out_mmhg',
                     metavar='MMHG', help='Outlet pressure [mmHg] (default 5)')
    ph2.add_argument('--mu',    type=float, default=3.5e-3, dest='mu_pas',
                     metavar='PA_S', help='Dynamic viscosity [Pa·s] (default 3.5e-3)')

    args = ap.parse_args()

    vtp_in = os.path.abspath(args.vtp_in)
    if not os.path.exists(vtp_in):
        sys.exit(f"Error: {vtp_in} not found")

    if args.label_path:
        args.label_path = os.path.abspath(args.label_path)
        if not os.path.exists(args.label_path):
            sys.exit(f"Error: {args.label_path} not found")

    if args.dt_path:
        args.dt_path = os.path.abspath(args.dt_path)
        if not os.path.exists(args.dt_path):
            sys.exit(f"Error: {args.dt_path} not found")

    vtp_out = (os.path.abspath(args.vtp_out)
               if args.vtp_out
               else _stem(vtp_in, ('.vtp',)) + '_pressure.vtp')

    solve(vtp_in, vtp_out, args)
