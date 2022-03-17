"""
Code modified from Kaolin.
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
Author: Bharat
"""
import torch

def np2tensor(x, device=None):
    return torch.tensor(x, device=device)


def tensor2np(x):
    return x.detach().cpu().numpy()


def closest_index(src_points: torch.Tensor, tgt_points: torch.Tensor, K=1):
    """
    Given two point clouds, finds closest point id
    :param src_points: B x N x 3
    :param tgt_points: B x M x 3
    :return B x N
    """
    from pytorch3d.ops import knn_points
    closest_index_in_tgt = knn_points(src_points, tgt_points, K=K)
    return closest_index_in_tgt.idx.squeeze(-1)


def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out


def batch_sparse_dense_matmul(S, D):
    """
    Batch sparse-dense matrix multiplication

    :param torch.SparseTensor S: a sparse tensor of size (batch_size, p, q)
    :param torch.Tensor D: a dense tensor of size (batch_size, q, r)
    :return: a dense tensor of size (batch_size, p, r)
    :rtype: torch.Tensor
    """

    num_b = D.shape[0]
    S_shape = S.shape
    if not S.is_coalesced():
        S = S.coalesce()

    indices = S.indices().view(3, num_b, -1)
    values = S.values().view(num_b, -1)
    ret = torch.stack([
        torch.sparse.mm(
            torch.sparse_coo_tensor(indices[1:, i], values[i], S_shape[1:], device=D.device),
            D[i]
        )
        for i in range(num_b)
    ])
    return ret


def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    """
    from pytorch3d.ops import knn_points

    assert s1.is_cuda and s2.is_cuda
    closest_dist_in_s2 = knn_points(s1, s2, K=1)
    closest_dist_in_s1 = knn_points(s2, s1, K=1)

    return (closest_dist_in_s2.dists**0.5 * w1).mean(axis=1).squeeze(-1) + (closest_dist_in_s1.dists**0.5 * w2).mean(axis=1).squeeze(-1)


def batch_chamfer(pc_list, verts, reduction='mean', reverse=False, bidirectional=False):
    """
    simple implementation to batchify pc with different number of points
    verts: (B, N, 3) tensor, where len(pc_list) == B
    pc_list: a list of point clouds with varying number of points. the mean is weighted by the number of points in each point cloud
    if reverse: compute chamfer from pc to verts
    default direction: verts to kinect pc
    """
    # from chamferdist import ChamferDistance
    assert len(pc_list) == verts.shape[0], 'the size of pc list does not match verts batch size'
    batch_size = verts.shape[0]
    # chamferDist = ChamferDistance()
    if bidirectional:
        w1, w2 = 1.0, 1.0
    else:
        w1, w2 = 1.0, 0.  # only compute distance from s1 to s2
    distances = []
    points_num = []
    for i, pc in enumerate(pc_list):
        if reverse:
            chamf = chamfer_distance(pc.unsqueeze(0), verts[i].unsqueeze(0), w1, w2)
            # chamf = chamferDist(pc.unsqueeze(0), verts[i].unsqueeze(0), bidirectional=bidirectional)  # unidirection: pc to SMPL
        else:
            # verts to pc distance
            chamf = chamfer_distance(verts[i].unsqueeze(0), pc.unsqueeze(0), w1, w2)
            # chamf = chamferDist(verts[i].unsqueeze(0), pc.unsqueeze(0), bidirectional=bidirectional) # unidirection: SMPL to pc
        distances.append(chamf)
        points_num.append(pc.shape[0])
    points_num = torch.tensor(points_num, device=verts.device, dtype=verts.dtype)
    distances = torch.stack(distances).reshape((batch_size, 1))
    if reduction == 'mean':
        total_num = torch.sum(points_num)
        weights = points_num.reshape((1, batch_size)) / total_num
        return torch.mean(torch.matmul(weights, distances))
    elif reduction == None:
        return distances
    else:
        raise NotImplemented


def get_closest_face(points, meshes):
    """
    NOT WORKING
    :param points: List of points
    :param mesh: pytorch3d meshes
    :return:
    """
    from pytorch3d.structures.pointclouds import Pointclouds
    from pytorch3d import _C

    pcls = Pointclouds(points)

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    dist, face_idx = _C.point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    # above provides squared distance
    return dist**0.5, face_idx

if __name__ == "__main__":
    from psbody.mesh import Mesh
    from pytorch3d.io import load_obj, load_objs_as_meshes
    import numpy as np
    from pytorch3d.structures import Meshes

    pts = np.zeros((1,3)).astype('float32')   #np.random.rand(1, 3).astype('float32') *2 -1
    # pts = np.array([[0,0,1], [0.5, 0.5, 0.5], [1, 0, -1]]).astype('float32')
    temp = Mesh(filename='/BS/bharat-2/static00/renderings/renderpeople_rigged/rp_eric_rigged_005_zup_a/rp_eric_rigged_005_zup_a_smpl.obj')
    # temp = Mesh(v=[[0,0,0], [0,1,0], [1,0,0], [0, 0, 1]], f=[[0,1,2], [0, 1, 3]])
    closest_face, closest_points = temp.closest_faces_and_points(pts)
    dist = np.linalg.norm(pts - closest_points, axis=1)


    # pytorch
    temp2 = load_objs_as_meshes([
                                    '/BS/bharat-2/static00/renderings/renderpeople_rigged/rp_eric_rigged_005_zup_a/rp_eric_rigged_005_zup_a_smpl.obj'])
    # temp2 = Meshes([np2tensor(temp.v.astype('float32'))], [np2tensor(temp.f.astype('float32'))])
    dist2, closest_face2 = get_closest_face([np2tensor(pts)], temp2)

    print('done')

