import numpy as np
import cupy as cp
from typing import List

eps = 1e-6

def compatibility(edges: List[tuple], pos: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise edge compatibility for a set of edges and node positions.

    Args:
        edges: List of (source, target) edge tuples (size: n_e x 2)
        pos: Node positions, shape (n_n, 2)
    Returns:
        C: Edge compatibility matrix, shape (n_e, n_e)
    """
    edge_pos = [_get_edge_positions(e, pos) for e in edges]
    M = len(edges)
    C = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            C[i, j] = _compatibility_score(edge_pos[i], edge_pos[j])
    # Symmetrize the matrix
    C += C.T
    return C

def _get_edge_positions(edge: tuple, pos: np.ndarray) -> np.ndarray:
    i1, i2 = edge
    return np.array([pos[i1], pos[i2]])

def _euc_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.linalg.norm(B - A)

def _point_on_line(q: np.ndarray, P: np.ndarray) -> np.ndarray:
    L = _euc_distance(P[0], P[1])
    if L == 0:
        return P[0]
    r = np.dot(P[1] - P[0], q - P[0]) / L**2
    return P[0] + r * (P[1] - P[0])

def _edge_visibility(P: np.ndarray, Q: np.ndarray) -> float:
    I0 = _point_on_line(Q[0], P)
    I1 = _point_on_line(Q[1], P)
    mid_I = (I0 + I1) / 2
    mid_P = (P[0] + P[1]) / 2
    denom = _euc_distance(I0, I1)
    temp = 1 - 2 * _euc_distance(mid_P, mid_I) / denom if denom > eps else eps
    return max(temp, eps)

def _Ca(P: np.ndarray, Q: np.ndarray) -> float:
    dot = np.dot(P[1] - P[0], Q[1] - Q[0])
    norm = _euc_distance(P[0], P[1]) * _euc_distance(Q[0], Q[1])
    return np.abs(dot / norm) if norm > eps else eps

def _Cs(P: np.ndarray, Q: np.ndarray) -> float:
    euc_P = _euc_distance(P[0], P[1])
    euc_Q = _euc_distance(Q[0], Q[1])
    l_ave = (euc_P + euc_Q) / 2
    denom = l_ave / min(euc_P, euc_Q) + max(euc_P, euc_Q) / l_ave if l_ave != 0 else 1
    return 2 / denom if denom > eps else eps

def _Cp(P: np.ndarray, Q: np.ndarray) -> float:
    euc_P = _euc_distance(P[0], P[1])
    euc_Q = _euc_distance(Q[0], Q[1])
    l_ave = (euc_P + euc_Q) / 2
    mid_P = (P[0] + P[1]) / 2
    mid_Q = (Q[0] + Q[1]) / 2
    euc_mid = _euc_distance(mid_P, mid_Q)
    denom = l_ave + euc_mid
    return l_ave / denom if denom > eps else eps

def _Cv(P: np.ndarray, Q: np.ndarray) -> float:
    return min(_edge_visibility(P, Q), _edge_visibility(Q, P))

def _compatibility_score(P: np.ndarray, Q: np.ndarray) -> float:
    return _Ca(P, Q) * _Cs(P, Q) * _Cp(P, Q) * _Cv(P, Q)

# ---------------------------------------------------------------------------
# GPU implementation

compatibility_kernel = cp.RawKernel(r'''
extern "C" __device__ float euc_distance(float ax,float ay,float bx,float by){
    float dx = bx - ax;
    float dy = by - ay;
    return sqrtf(dx*dx + dy*dy);
}

extern "C" __device__ void point_on_line(
    float qx,float qy,
    float px0,float py0,float px1,float py1,
    float* rx,float* ry){
    float lx = px1 - px0;
    float ly = py1 - py0;
    float l2 = lx*lx + ly*ly;
    float r = ((qx-px0)*lx + (qy-py0)*ly) / l2;
    *rx = px0 + r*lx;
    *ry = py0 + r*ly;
}

extern "C" __device__ float edge_visibility(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    float I0x,I0y,I1x,I1y;
    point_on_line(qx0,qy0,px0,py0,px1,py1,&I0x,&I0y);
    point_on_line(qx1,qy1,px0,py0,px1,py1,&I1x,&I1y);
    float midIx = 0.5f*(I0x + I1x);
    float midIy = 0.5f*(I0y + I1y);
    float midPx = 0.5f*(px0 + px1);
    float midPy = 0.5f*(py0 + py1);
    float denom = euc_distance(I0x,I0y,I1x,I1y);
    if(denom == 0.0f) return 0.0f;
    float temp = 1.0f - 2.0f*euc_distance(midPx,midPy,midIx,midIy)/denom;
    return temp > 0.0f ? temp : 0.0f;
}

extern "C" __device__ float Ca(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    float dot = (px1-px0)*(qx1-qx0) + (py1-py0)*(qy1-qy0);
    float norm = euc_distance(px0,py0,px1,py1) * euc_distance(qx0,qy0,qx1,qy1);
    if(norm == 0.0f) return 0.0f;
    float cosv = dot / norm;
    return fabsf(cosv);
}

extern "C" __device__ float Cs(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    float eucP = euc_distance(px0,py0,px1,py1);
    float eucQ = euc_distance(qx0,qy0,qx1,qy1);
    float lAve = 0.5f*(eucP + eucQ);
    float minL = eucP < eucQ ? eucP : eucQ;
    float maxL = eucP > eucQ ? eucP : eucQ;
    float denom = lAve/minL + maxL/lAve;
    return denom != 0.0f ? 2.0f/denom : 0.0f;
}

extern "C" __device__ float Cp(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    float eucP = euc_distance(px0,py0,px1,py1);
    float eucQ = euc_distance(qx0,qy0,qx1,qy1);
    float lAve = 0.5f*(eucP + eucQ);
    float midPx = 0.5f*(px0 + px1);
    float midPy = 0.5f*(py0 + py1);
    float midQx = 0.5f*(qx0 + qx1);
    float midQy = 0.5f*(qy0 + qy1);
    float eucMid = euc_distance(midPx,midPy,midQx,midQy);
    float denom = lAve + eucMid;
    return denom != 0.0f ? lAve/denom : 0.0f;
}

extern "C" __device__ float Cv(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    float v1 = edge_visibility(px0,py0,px1,py1,qx0,qy0,qx1,qy1);
    float v2 = edge_visibility(qx0,qy0,qx1,qy1,px0,py0,px1,py1);
    return v1 < v2 ? v1 : v2;
}

extern "C" __device__ float compatibility_score(
    float px0,float py0,float px1,float py1,
    float qx0,float qy0,float qx1,float qy1){
    return Ca(px0,py0,px1,py1,qx0,qy0,qx1,qy1) *
           Cs(px0,py0,px1,py1,qx0,qy0,qx1,qy1) *
           Cp(px0,py0,px1,py1,qx0,qy0,qx1,qy1) *
           Cv(px0,py0,px1,py1,qx0,qy0,qx1,qy1);
}

extern "C" __global__ void compatibility_kernel(
    const float* edges,int n_e,float* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_e) return;
    for(int j=i+1;j<n_e;++j){
        float px0 = edges[i*4];
        float py0 = edges[i*4+1];
        float px1 = edges[i*4+2];
        float py1 = edges[i*4+3];
        float qx0 = edges[j*4];
        float qy0 = edges[j*4+1];
        float qx1 = edges[j*4+2];
        float qy1 = edges[j*4+3];
        float val = compatibility_score(px0,py0,px1,py1,qx0,qy0,qx1,qy1);
        out[i*n_e + j] = val;
        out[j*n_e + i] = val;
    }
}
''', 'compatibility_kernel')


def compatibility_gpu(edges: list, pos: list) -> np.ndarray:
    """Compute compatibility matrix using a CUDA kernel.

    Parameters
    ----------
    edges : list
        List of (source, target) index tuples (size ``n_e`` x 2).
    pos : list
        Node coordinates array of shape ``(n_n, 2)``.

    Returns
    -------
    np.ndarray
        Symmetric compatibility matrix of shape ``(n_e, n_e)``.
    """
    m = len(edges)
    # prepare edge endpoints in a single array on the GPU
    edge_pos = np.zeros((m, 4), dtype=np.float32)
    for idx, (s, t) in enumerate(edges):
        edge_pos[idx, 0:2] = pos[s]
        edge_pos[idx, 2:4] = pos[t]
    edge_pos_gpu = cp.asarray(edge_pos)
    out_gpu = cp.zeros((m, m), dtype=cp.float32)
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    compatibility_kernel((blocks,), (threads_per_block,),
                         (edge_pos_gpu.ravel(), m, out_gpu.ravel()))
    return cp.asnumpy(out_gpu)
