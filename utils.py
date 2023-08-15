import torch
import numpy as np

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

def dcg_at_k(r, k):
    r = r[:, :k]
    dcg = (r / torch.log2(torch.arange(2, r.size()[-1] + 2, dtype=torch.float32)).to(r.device)).sum(1)
    return dcg

def ndcg_at_k(r, k):
    best_r, _ = torch.sort(r, dim=1, descending=True)
    dcg_max = dcg_at_k(best_r, k)
    dcg = dcg_at_k(r, k)
    ndcg = torch.where(dcg_max > 0, dcg / dcg_max, torch.zeros_like(dcg, device=dcg.device))
    return ndcg

@torch.no_grad()
def euclidean_distance(x, y, topk=2):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return torch.topk(dist, topk, largest=False)

@torch.no_grad()
def get_metric(
        query: torch.Tensor,
        query_label: list,
        gallery: torch.Tensor = None,
        gallery_label: list = None,
        metric: int = 5):

    if gallery is None:
        query = query.cuda()
        gallery_label = query_label.cpu().numpy()
        list_pred = []
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], query,topk=metric+1)
            index_np = index_pt.cpu().numpy()[:, 1:metric+1]
            list_pred.append(index_np)
            idx += 128
        query_label = np.repeat(np.array(query_label)[:,np.newaxis],repeats=metric,axis=1)
        pred = np.concatenate(list_pred).reshape(num_feat,metric)
        exact_match_relevance = query_label == gallery_label[pred]
        rank_1 = np.sum(np.any(exact_match_relevance[:,:1], axis=1)) / num_feat
        rank_5 = np.sum(np.any(exact_match_relevance[:,:5], axis=1)) / num_feat
        ndcg_1 = ndcg_at_k(torch.from_numpy(exact_match_relevance), 1)
        ndcg_5 = ndcg_at_k(torch.from_numpy(exact_match_relevance), 5)

        mAP = 0
        for q_res in torch.from_numpy(exact_match_relevance):
            AP = 0
            for k in range(5):
                if q_res[k] == 1:
                    right = torch.sum(q_res[:k+1])
                    precision = right * 1.0 / (k+1)
                    AP += precision / 5
            mAP += AP
        mAP = mAP / len(exact_match_relevance)

        rank_1 = float(rank_1)
        rank_5 = float(rank_5)
        ndcg_1 = float(ndcg_1.mean().item())
        ndcg_5 = float(ndcg_5.mean().item())
        try:
            mAP = float(mAP.item())
        except:
            mAP = float(mAP)


        return {"R@1": rank_1 * 100, "R@5": rank_5 * 100, "ndcg@1": ndcg_1 * 100, "ndcg@5": ndcg_5 * 100, "mAP": mAP * 100,}
    else:
        query = query.cuda()
        query_label = query_label.cpu().numpy()
        gallery = gallery.cuda()
        gallery_label = gallery_label.cpu().numpy()
        list_pred = []
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], gallery,topk=metric+1)
            index_np = index_pt.cpu().numpy()[:, 1:metric+1]
            list_pred.append(index_np)
            idx += 128
        query_label = np.repeat(np.array(query_label)[:,np.newaxis],repeats=metric,axis=1)
        pred = np.concatenate(list_pred).reshape(num_feat,metric)
        exact_match_relevance = query_label == gallery_label[pred]
        rank_1 = np.sum(np.any(exact_match_relevance[:,:1], axis=1)) / num_feat
        rank_5 = np.sum(np.any(exact_match_relevance[:,:5], axis=1)) / num_feat
        ndcg_1 = ndcg_at_k(torch.from_numpy(exact_match_relevance), 1)
        ndcg_5 = ndcg_at_k(torch.from_numpy(exact_match_relevance), 5)

        mAP = 0
        for q_res in torch.from_numpy(exact_match_relevance):
            AP = 0
            for k in range(5):
                if q_res[k] == 1:
                    right = torch.sum(q_res[:k+1])
                    precision = right * 1.0 / (k+1)
                    AP += precision / 5
            mAP += AP
        mAP = mAP / len(exact_match_relevance)

        rank_1 = float(rank_1)
        rank_5 = float(rank_5)
        ndcg_1 = float(ndcg_1.mean().item())
        ndcg_5 = float(ndcg_5.mean().item())
        try:
            mAP = float(mAP.item())
        except:
            mAP = float(mAP)

        return {"R@1": rank_1 * 100, "R@5": rank_5 * 100, "ndcg@1": ndcg_1 * 100, "ndcg@5": ndcg_5 * 100, "mAP": mAP * 100,}