import os
import torch
from torch.nn import functional as F
import numpy as np
import pandas
import pickle
from time import time

from vehicle_reid_pytorch.metrics import eval_func, eval_func_mp
from vehicle_reid_pytorch.loss.triplet_loss import normalize, euclidean_dist
from functools import reduce

from vehicle_reid_pytorch.metrics.rerank import re_ranking
from vehicle_reid_pytorch.utils.tools import set_diag_to_zreo, constr_views_mask

from vehicle_reid_pytorch.utils.visualizer_from_fastreid import Visualizer
from sklearn import metrics
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from vehicle_reid_pytorch.utils.file_io import PathManager

# def calc_dist_split(qf, gf, split=0):
#     qf = qf
#     m = qf.shape[0]
#     n = gf.shape[0]
#     distmat = gf.new(m, n)

#     if split == 0:
#         distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                 torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         distmat.addmm_(x, y.t(), beta=1, alpha=-2)

#     # 用于测试时控制显存
#     else:
#         start = 0
#         while start < n:
#             end = start + split if (start + split) < n else n
#             num = end - start

#             sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
#                     torch.pow(gf[start:end], 2).sum(dim=1, keepdim=True).expand(num, m).t()
#             # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
#             sub_distmat.addmm_(qf, gf[start:end].t(), beta=1, alpha=-2)
#             distmat[:, start:end] = sub_distmat.cpu()
#             start += num

#     return distmat


def clck_dist(feat1, feat2, vis_score1, vis_score2, split=0):
    """
    计算vpm论文中的clck距离

    :param torch.Tensor feat1: [B1, C, 3]
    :param torch.Tensor feat2: [B2, C, 3]
    :param torch.Tensor vis_score: [B, 3]
    :rtype torch.Tensor
    :return: clck distance. [B1, B2]
    """

    B, C, N = feat1.shape
    dist_mat = 0
    ckcl = 0
    for i in range(N):
        parse_feat1 = feat1[:, :, i]
        parse_feat2 = feat2[:, :, i]
        ckcl_ = torch.mm(vis_score1[:, i].view(-1, 1), vis_score2[:, i].view(1, -1))  # [N, N]
        ckcl += ckcl_
        dist_mat += euclidean_dist(parse_feat1, parse_feat2, split=split).sqrt() * ckcl_

    return dist_mat / ckcl


class Clck_R1_mAP:
    def __init__(self, num_query, *, max_rank=50, feat_norm=True, output_path='', rerank=False,
                 remove_junk=True,
                 lambda_=0.5):
        """
        计算VPM中的可见性距离并计算性能

        :param num_query:
        :param max_rank:
        :param feat_norm:
        :param output_path:
        :param rerank:
        :param remove_junk:
        :param lambda_: distmat = global_dist + lambda_ * local_dist, default 0.5
        """
        super(Clck_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.output_path = output_path
        self.rerank = rerank
        self.remove_junk = remove_junk
        self.lambda_ = lambda_
        self.reset()


    def reset(self):
        self.s_feats = []
        self.d_feats = []
        self.views = []
        self.global_feats = []
        self.local_feats = []
        self.vis_scores = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):
        global_feat, local_feat, vis_score, pid, camid, paths = output
        self.global_feats += global_feat
        self.local_feats += local_feat
        self.vis_scores += vis_score
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths += paths

    def update_vanet(self, output):
        s_feats, d_feats, views, pid, camid, paths = output
        self.s_feats += s_feats
        self.d_feats += d_feats
        self.views.extend(np.asarray(views))
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths += paths

    def resplit_for_vehicleid(self):
        """每个ID随机选择一辆车组成gallery，剩下的为query。
        """
        # 采样
        indexes = range(len(self.pids))
        df = pandas.DataFrame(dict(index=indexes, pid=self.pids))
        query_idxs = []
        gallery_idxs = []
        for idx, group in df.groupby('pid'):
            gallery = group.sample(1)['index'][0]
            gallery_idxs.append(gallery)
            for index in group.indexes:
                if index != gallery:
                    query_idxs.append(index)
        re_idxs = query_idxs + gallery_idxs
        # 重排序
        self.global_feats = [self.global_feats[i] for i in re_idxs]
        self.local_feats = [self.local_feats[i] for i in re_idxs]
        self.vis_scores = [self.vis_scores[i] for i in re_idxs]
        self.pids = [self.pids[i] for i in re_idxs]
        self.camids = [self.camids[i] for i in re_idxs]
        self.paths = [self.paths[i] for i in re_idxs]

    def cal_dist_mat(self, split=0, views_feats=None):
        """
        :param split:
        :param feats:
        :param views_mask:
        :return:
        """
        # query
        qf = views_feats[:self.num_query]
        # gallery
        gf = views_feats[self.num_query:]

        qf = qf
        m, n = qf.shape[0], gf.shape[0]
        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            # qf: M, F
            # gf: N, F
            if split == 0:
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)

            else:
                distmat = gf.new(m, n)

                start = 0
                while start < n:
                    end = start + split if (start + split) < n else n
                    num = end - start
                    sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                                  torch.pow(gf[start:end], 2).sum(dim=1, keepdim=True).expand(num,
                                                                                              m).t()
                    # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
                    sub_distmat.addmm_(qf, gf[start:end].t(), beta=1, alpha=-2)
                    distmat[:, start:end] = sub_distmat

                    start += num

            distmat = distmat.detach().numpy()

        return distmat

    # remove same id in same camera.
    def get_matched_result(self, indices, matches, q_pids, q_camids, g_pids, g_camids, q_index):
        q_pid = q_pids[q_index]
        q_camid = q_camids[q_index]
        order = indices[q_index]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def plot_roc_curve(self, fpr, tpr, name='model', fig=None):
        if fig is None:
            fig = plt.figure()
            plt.semilogx(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'r', linestyle='--', label='Random guess')
        plt.semilogx(fpr, tpr, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
                     label='ROC curve with {}'.format(name))
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        return fig

    def save_roc_info(self, output, fpr, tpr, pos, neg):
        results = {
            "fpr": np.asarray(fpr),
            "tpr": np.asarray(tpr),
            "pos": np.asarray(pos),
            "neg": np.asarray(neg),
        }
        with open(os.path.join(output, "roc_info.pickle"), "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def vis_roc_curve(self, output, dist, q_pids, q_camids, g_pids, g_camids):
        print("calculate roc curve...")
        start_time = time()
        PathManager.mkdirs(output)
        pos, neg = [], []
        self.indices = np.argsort(dist, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)
        for i, query in enumerate(tqdm(q_pids)):
            cmc, sort_idx = self.get_matched_result(self.indices, self.matches, q_pids, q_camids, g_pids, g_camids, i)  # remove same id in same camera
            ind_pos = np.where(cmc == 1)[0]
            q_dist = dist[i]
            pos.extend(q_dist[sort_idx[ind_pos]])

            ind_neg = np.where(cmc == 0)[0]
            neg.extend(q_dist[sort_idx[ind_neg]])

        scores = np.hstack((pos, neg))
        labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        print("time used: ", time() - start_time)
        # self.plot_roc_curve(fpr, tpr)
        # filepath = os.path.join(output, "roc.jpg")
        # plt.savefig(filepath)

        # self.plot_distribution(pos, neg)
        # filepath = os.path.join(output, "pos_neg_dist.jpg")
        # plt.savefig(filepath)
        return fpr, tpr, pos, neg

    def plot_distribution(self, vis_save_path, t_name, s_name, t_pos, t_neg, s_pos, s_neg, fig=None):
        """
        :param vis_save_path:
        :param pos:
        :param neg:
        :param name:
        :param fig:
        :return:
        """
        print("plot hist...")
        print("length of 'pos': ", len(t_pos))
        print("length of 'neg': ", len(t_neg))
        if fig is None:
            fig = plt.figure()
        ###################two branch###################
        # two branch pos
        t_pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(t_pos, bins=80, alpha=0.7, density=True,
                              color=t_pos_color,
                              label='positive with {}'.format(t_name))
        mu = np.mean(t_pos)
        sigma = np.std(t_pos)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=t_pos_color)  # plot y curve

        # two branch neg
        t_neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(t_neg, bins=80, alpha=0.5, density=True,
                              color=t_neg_color,
                              label='negative with {}'.format(t_name))
        mu = np.mean(t_neg)
        sigma = np.std(t_neg)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=t_neg_color)  # plot y curve

        ###################single branch##################
        # for single branch positive samples.
        s_pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(s_pos, bins=80, alpha=0.7, density=True,
                              color=s_pos_color,
                              label='positive with {}'.format(s_name))
        mu = np.mean(s_pos)
        sigma = np.std(s_pos)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=s_pos_color)  # plot y curve

        # for single branch negative samples.
        s_neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(s_neg, bins=80, alpha=0.5, density=True,
                              color=s_neg_color,
                              label='negative with {}'.format(s_name))
        mu = np.mean(s_neg)
        sigma = np.std(s_neg)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=s_neg_color)  # plot y curve


        # plt.xticks(np.arange(0, 2.5, 0.1))  # (0, 1.5, 0.1)
        plt.xlim(0, 2.5)
        plt.title('positive and negative pairs distribution')
        plt.legend(loc='best')
        # plt.savefig('{}.jpg'.format(name))
        plt.savefig('vis_t.jpg')
        return fig

    def draw_distri(self, vis_save_path, fig_name, distmat, q_pids, q_camids, g_pids, g_camids):
        """
        :param vis_save_path:
        :param distmat:
        :param q_pids:
        :param q_camids:
        :param g_pids:
        :param g_camids:
        :return:
        """
        fpr, tpr, pos, neg = self.vis_roc_curve(vis_save_path, distmat, q_pids, q_camids, g_pids,
                                                g_camids)

        # save pos, neg to pickle. Just for data analysis.
        # pkl_dict = {}
        # pkl_list = []
        # pkl_dict['pos'] = pos #[:500]
        # pkl_dict['neg'] = neg
        # pkl_list.append(pkl_dict)
        # pkl_file = open('single-branch_s-view_infer.pkl', 'wb')
        # pickle.dump(pkl_list, pkl_file)
        # print("pkl file saved!!!")

        # save roc curve
        # self.save_roc_info(vis_out, fpr, tpr, pos, neg)

        # plot distribuction prediction res
        # print("prepare to plot distribution...")
        # self.plot_distribution(vis_save_path=vis_save_path, name=fig_name, pos=pos, neg=neg, fig=None)

        # save distribuction
        # plt.savefig('dist.jpg')

    def compute_vanet_two_branch(self, save_path, split=0):
        """
        split: When the CUDA memory is not sufficient, we can split the dataset into different parts
               for the computing of distance.
        """
        s_feats = torch.stack((self.s_feats), dim=0)
        d_feats = torch.stack((self.d_feats), dim=0)

        if self.feat_norm:
            print("The test feature is normalized")
            s_feats = F.normalize(s_feats, dim=1, p=2)
            d_feats = F.normalize(d_feats, dim=1, p=2)

        print('Calculate distance matrixs...')
        # query
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_views = torch.from_numpy(np.asarray(self.views[:self.num_query]))
        # gallery
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_views = torch.from_numpy(np.asarray(self.views[self.num_query:]))
        # pdb.set_trace()
        s_branch_distmat = self.cal_dist_mat(split=split, views_feats=s_feats)  # np.array(1677, 11578)
        d_branch_distmat = self.cal_dist_mat(split=split, views_feats=d_feats)  # np.array(1677, 11578)

        # convert ndarray to tensor
        s_branch_distmat = torch.from_numpy(s_branch_distmat)
        d_branch_distmat = torch.from_numpy(d_branch_distmat)
        # generate q_views mask and g_views mask respectively.
        # make two dummy views matrixs of same dimension.(dim1=q_size, dim2=g_size)
        dummy_q_views = q_views.repeat(g_views.shape[0], 1).permute(1, 0)  # (q_size, g_size)
        dummy_g_views = g_views.repeat(q_views.shape[0], 1)  # (q_size, g_size)
        s_views_mask = [dummy_q_views == dummy_g_views]  # boolean (q_size, g_size)
        d_views_mask = [dummy_q_views != dummy_g_views]  # boolean (q_size, g_size)
        s_views_mask = s_views_mask[0].long()   # Convert bool value to int. Tensor type,(q_size, g_size)
        d_views_mask = d_views_mask[0].long()   # Convert bool value to int. Tensor type,(q_size, g_size)

        distmat = torch.where(s_views_mask > 0, s_branch_distmat, d_branch_distmat)  # integrate two dist matrixs (coordinate-wise).
        print('Eval...')
        cmc, mAP, all_AP = eval_func_mp(distmat, q_pids, g_pids, q_camids, g_camids,
                                        remove_junk=self.remove_junk)

        ###### add visualize results from fast-reid
        # draw total distribuction hist.
        # print("draw two branch distribution fig...")

        ###################################
        # get specific pos and neg.
        q_pids = torch.from_numpy(q_pids)
        g_pids = torch.from_numpy(g_pids)
        dummy_q_pid = q_pids.repeat(g_pids.shape[0], 1).permute(1, 0)
        dummy_g_pid = g_pids.repeat(q_pids.shape[0], 1)
        s_id_mask = [dummy_q_pid == dummy_g_pid]
        d_id_mask = [dummy_q_pid != dummy_g_pid]
        s_id_mask = s_id_mask[0].long()
        d_id_mask = d_id_mask[0].long()
        zero = torch.zeros_like(s_views_mask, dtype=torch.float32)

        s_view_s_id_mask = torch.mul(s_id_mask, s_views_mask)
        s_view_d_id_mask = torch.mul(d_id_mask, s_views_mask)

        sim_pos_distmat = distmat[s_view_s_id_mask == 1]
        sim_neg_distmat = distmat[s_view_d_id_mask == 1]

        d_view_s_id_mask = torch.mul(s_id_mask, d_views_mask)
        d_view_d_id_mask = torch.mul(d_id_mask, d_views_mask)

        diff_pos_distmat = distmat[d_view_s_id_mask == 1]
        diff_neg_distmat = distmat[d_view_d_id_mask == 1]

        # Save pos, neg to pickle file.
        # if needed, you can open this following operations for further analysis.
        # s_pkl_dict = {}
        # d_pkl_dict = {}
        # s_pkl_list = []
        # d_pkl_list = []
        # s_pkl_dict['pos'] = sim_pos_distmat  # get nearest distance for positive samples.
        # s_pkl_dict['neg'] = sim_neg_distmat[torch.argsort(sim_neg_distmat)[:len(sim_pos_distmat)]]
        # d_pkl_dict['pos'] = diff_pos_distmat  # get furthest distance for negative samples.
        # d_pkl_dict['neg'] = diff_neg_distmat[torch.argsort(diff_neg_distmat)[:len(diff_pos_distmat)]]

        # s_pkl_list.append(s_pkl_dict)
        # d_pkl_list.append(d_pkl_dict)
        
        # s_pkl_file = open('two-branch_s-view_pos-&-neg_infer_part-neg.pkl', 'wb')
        # d_pkl_file = open('two-branch_d-view_pos-&-neg_infer_part-neg.pkl', 'wb')
        # pickle.dump(s_pkl_list, s_pkl_file)
        # pickle.dump(d_pkl_list, d_pkl_file)
        # print("pkl file saved.")
        #####################################

        # self.draw_distri(vis_save_path, 'two_branch', distmat, q_pids, q_camids, g_pids, g_camids)

        # draw distribuction hist of similar views.
        # print("draw s branch distribution fig...")
        # self.draw_distri(vis_save_path, 's_branch', s_branch_distmat, q_pids, q_camids, g_pids, g_camids)

        # draw distribuction hist of different views.
        # print("draw d branch distribution fig...")
        # self.draw_distri(vis_save_path, 'd_branch', d_branch_distmat, q_pids, q_camids, g_pids, g_camids)

        return {
            "cmc": cmc,
            "mAP": mAP,
            "distmat": s_branch_distmat,
            "all_AP": all_AP
        }

    def compute_vanet_single_branch(self, save_path, split=0):
        """
        @Modified 
        split: When the CUDA memory is not sufficient, we can split the dataset into different parts
               for the computing of distance.
        """
        s_feats = torch.stack((self.s_feats), dim=0)  # [13255, 2048]
        d_feats = torch.stack((self.d_feats), dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            s_feats = F.normalize(s_feats, dim=1, p=2)
            d_feats = F.normalize(d_feats, dim=1, p=2)

        print('Calculate distance matrixs...')
        # query
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_views = torch.from_numpy(np.asarray(self.views[:self.num_query]))

        # gallery
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_views = torch.from_numpy(np.asarray(self.views[self.num_query:]))


        # # just to save info.
        # # pdb.set_trace()
        # g_pids = self.pids[self.num_query:]
        # g_camids = self.camids[self.num_query:]
        # g_paths = self.paths[self.num_query:]
        # g_frames = [i.split('/')[-1].split('_')[-2] for i in g_paths]
        # g_frames = list(map(int, g_frames))
        # g_features = s_feats.numpy()[self.num_query :]
        
        # q_pids = self.pids[:self.num_query]
        # q_camids = self.camids[:self.num_query]
        # q_paths = self.paths[:self.num_query]
        # q_frames = [i.split('/')[-1].split('_')[-2] for i in q_paths]
        # q_frames = list(map(int, q_frames))
        # q_features = s_feats.numpy()[: self.num_query]

        # res = {}
        # res['gallery_f'] = g_features
        # res['gallery_label'] = g_pids
        # res['gallery_cam'] = g_camids
        # res['gallery_frames'] = g_frames

        # res['query_f'] = q_features
        # res['query_label'] = q_pids
        # res['query_cam'] = q_camids
        # res['query_frames'] = q_frames
        # from scipy import io
        # io.savemat('./pytorch_results.mat', res)
        # print("################# ok ###############")

        s_distmat = self.cal_dist_mat(split=split, views_feats=s_feats)  # np.array(1677, 11578)
        d_distmat = self.cal_dist_mat(split=split, views_feats=d_feats)  # np.array(1677, 11578)

        s_distmat = torch.from_numpy(s_distmat)
        d_distmat = torch.from_numpy(d_distmat)

        # generate q_views mask and g_views mask respectively.
        # make two dummy views matrixs of same dimension.(dim1=q_size, dim2=g_size)
        dummy_q_views = q_views.repeat(g_views.shape[0], 1).permute(1, 0)  # (q_size, g_size)
        dummy_g_views = g_views.repeat(q_views.shape[0], 1)  # (q_size, g_size)
        s_views_mask = [dummy_q_views == dummy_g_views]  # boolean (q_size, g_size)
        d_views_mask = [dummy_q_views != dummy_g_views]  # boolean (q_size, g_size)
        s_views_mask = s_views_mask[0].long()   # Convert bool value to int. Tensor type,(q_size, g_size)
        d_views_mask = d_views_mask[0].long()   # Convert bool value to int. Tensor type,(q_size, g_size)

        # distmat = torch.where(s_views_mask > 0, s_distmat, d_distmat)  # integrate two dist matrixs (coordinate-wise).
        print('Eval...')
        cmc, mAP, all_AP = eval_func_mp(s_distmat, q_pids, g_pids, q_camids, g_camids,
                                        remove_junk=self.remove_junk)

        ###### add visualization results from fast-reid
        vis_save_path = save_path

        # get specific pos and neg.
        q_pids = torch.from_numpy(q_pids)
        g_pids = torch.from_numpy(g_pids)
        dummy_q_pid = q_pids.repeat(g_pids.shape[0], 1).permute(1, 0)
        dummy_g_pid = g_pids.repeat(q_pids.shape[0], 1)
        s_id_mask = [dummy_q_pid == dummy_g_pid]
        d_id_mask = [dummy_q_pid != dummy_g_pid]
        s_id_mask = s_id_mask[0].long()
        d_id_mask = d_id_mask[0].long()
        # pdb.set_trace()
        zero = torch.zeros_like(s_views_mask, dtype=torch.float32)

        s_view_s_id_mask = torch.mul(s_id_mask, s_views_mask)
        s_view_d_id_mask = torch.mul(d_id_mask, s_views_mask)

        sim_pos_distmat = s_distmat[s_view_s_id_mask == 1]
        sim_neg_distmat = s_distmat[s_view_d_id_mask == 1]

        d_view_s_id_mask = torch.mul(s_id_mask, d_views_mask)
        d_view_d_id_mask = torch.mul(d_id_mask, d_views_mask)

        diff_pos_distmat = s_distmat[d_view_s_id_mask == 1]
        diff_neg_distmat = s_distmat[d_view_d_id_mask == 1]

        # save pos, neg to pickle.
        s_pkl_dict = {}
        d_pkl_dict = {}
        s_pkl_list = []
        d_pkl_list = []
        s_pkl_dict['pos'] = sim_pos_distmat  # get positive nearest distance
        s_pkl_dict['neg'] = sim_neg_distmat[torch.argsort(sim_neg_distmat)[:len(sim_pos_distmat)]]
        d_pkl_dict['pos'] = diff_pos_distmat # get negative furthest distance
        d_pkl_dict['neg'] = diff_neg_distmat[torch.argsort(diff_neg_distmat)[:len(diff_pos_distmat)]]

        s_pkl_list.append(s_pkl_dict)
        d_pkl_list.append(d_pkl_dict)
        s_pkl_file = open('single-branch_s-view_pos-&-neg_infer_part-neg.pkl', 'wb')
        d_pkl_file = open('single-branch_d-view_pos-&-neg_infer_part-neg.pkl', 'wb')
        pickle.dump(s_pkl_list, s_pkl_file)
        pickle.dump(d_pkl_list, d_pkl_file)
        print("pkl file saved.")

        # print("draw single branch distribution fig...")
        # self.draw_distri(vis_save_path, 'single_branch', sim_s_distmat, q_pids, q_camids, g_pids, g_camids)

        return {
            "cmc": cmc,
            "mAP": mAP,
            "distmat": s_distmat,
            "all_AP": all_AP
        }

    def compute(self, split=0):
        """
        split: When the CUDA memory is not sufficient, we can split the dataset into different parts
               for the computing of distance.
        """
        global_feats = torch.stack(self.global_feats, dim=0) # torch.Size([13255, 2048])
        local_feats = torch.stack(self.local_feats, dim=0)  # torch.Size([13255, 2048, 4])
        vis_scores = torch.stack(self.vis_scores)  # torch.Size([13255, 4])
        if self.feat_norm:
            print("The test feature is normalized")
            global_feats = F.normalize(global_feats, dim=1, p=2)
            local_feats = F.normalize(local_feats, dim=1, p=2)
        # 全局距离
        print('Calculate distance matrixs...')
        # query
        qf = global_feats[:self.num_query]  # torch.Size([1677, 2048])
        q_pids = np.asarray(self.pids[:self.num_query])  # (1677, )
        q_camids = np.asarray(self.camids[:self.num_query])  # (1677, )
        # gallery
        gf = global_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        qf = qf
        m, n = qf.shape[0], gf.shape[0]  # 1677 + 11578 = 13255

        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            # qf: M, F
            # gf: N, F
            if split == 0:
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            else:
                distmat = gf.new(m, n)
                start = 0
                while start < n:
                    end = start + split if (start + split) < n else n
                    num = end - start

                    sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                                  torch.pow(gf[start:end], 2).sum(dim=1, keepdim=True).expand(num,
                                                                                              m).t()
                    # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
                    sub_distmat.addmm_(qf, gf[start:end].t(), beta=1, alpha=-2)
                    distmat[:, start:end] = sub_distmat

                    start += num

            distmat = distmat.detach().numpy()

        # 局部距离
        print('Calculate local distances...')
        local_distmat = clck_dist(local_feats[:self.num_query], local_feats[self.num_query:],
                                  vis_scores[:self.num_query], vis_scores[self.num_query:],
                                  split=split)

        local_feats = local_feats
        local_distmat = local_distmat.detach().cpu().numpy()

        if self.output_path:
            print('Saving results...')
            outputs = {
                "global_feats": global_feats,
                "vis_scores": vis_scores,
                "local_feats": local_feats,
                "pids": self.pids,
                "camids": self.camids,
                "paths": self.paths,
                "num_query": self.num_query,
                "distmat": distmat,
                "local_distmat": local_distmat,
            }
            torch.save(outputs, os.path.join(self.output_path, 'test_output.pkl'),
                       pickle_protocol=4)

        print('Eval...')
        cmc, mAP, all_AP = eval_func_mp(distmat + self.lambda_ * (local_distmat ** 2), q_pids,
                                        g_pids, q_camids, g_camids,
                                        remove_junk=self.remove_junk)

        return {
            "cmc": cmc,
            "mAP": mAP,
            "distmat": distmat,
            "all_AP": all_AP
        }

if __name__ == '__main__':

    def plot_distribution(name, s_pos, s_neg, d_pos, d_neg, fig=None):
        """
        :param vis_save_path:
        :param pos:
        :param neg:
        :param name:
        :param fig:
        :return:
        """
        print("plot hist...")
        if fig is None:
            fig = plt.figure()
        ###################two branch
        # two branch pos
        # pdb.set_trace()
        s_pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(s_pos, bins=80, alpha=0.7, density=True,
                              color=s_pos_color,
                              label='positive with {}-s-view'.format(name))
        # mu = np.mean(s_pos)
        # sigma = np.std(s_pos)
        mu = torch.mean(s_pos)
        sigma = torch.std(s_pos)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=s_pos_color)  # plot y curve

        # two branch neg
        s_neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(s_neg, bins=80, alpha=0.5, density=True,
                              color=s_neg_color,
                              label='negative with {}-s-view'.format(name))
        # mu = np.mean(s_neg)
        # sigma = np.std(s_neg)
        mu = torch.mean(s_neg)
        sigma = torch.std(s_neg)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=s_neg_color)  # plot y curve

        ###################single branch
        # single branch pos
        d_pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(d_pos, bins=80, alpha=0.7, density=True,
                              color=d_pos_color,
                              label='positive with {}-d-view'.format(name))
        # mu = np.mean(d_pos)
        # sigma = np.std(d_pos)
        mu = torch.mean(d_pos)
        sigma = torch.std(d_pos)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=d_pos_color)  # plot y curve

        # single branch neg
        d_neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(d_neg, bins=80, alpha=0.5, density=True,
                              color=d_neg_color,
                              label='negative with {}-d-view'.format(name))
        # mu = np.mean(d_neg)
        # sigma = np.std(d_neg)
        mu = torch.mean(d_neg)
        sigma = torch.std(d_neg)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=d_neg_color)  # plot y curve


        # plt.xticks(np.arange(0, 2.5, 0.1))  # (0, 1.5, 0.1)
        plt.xlim(0, 2.8)
        plt.title('views-aware positive and negative pairs distribution')
        plt.legend(loc='best')
        # plt.savefig('{}.jpg'.format(name))
        plt.savefig('vis_neg-part_{}_branch.jpg'.format(name))
        return fig

    # draw total hist graph

    # two branch
    two_s_view_path = '/liushichao/VANet/two-branch_s-view_pos-&-neg_infer_part-neg.pkl'
    two_d_view_path = '/liushichao/VANet/two-branch_d-view_pos-&-neg_infer_part-neg.pkl'

    two_s_data = pickle.load(open(two_s_view_path, 'rb'))
    two_d_data = pickle.load(open(two_d_view_path, 'rb'))

    two_s_pos = two_s_data[0]['pos']
    two_s_neg = two_s_data[0]['neg']
    two_d_pos = two_d_data[0]['pos']
    two_d_neg = two_d_data[0]['neg']
    # pdb.set_trace()
    plot_distribution(name='two', s_pos=two_s_pos, s_neg=two_s_neg,
                      d_pos=two_d_pos, d_neg=two_d_neg, fig=None)

    # single branch
    single_s_view_path = '/liushichao/VANet/single-branch_s-view_pos-&-neg_infer_part-neg.pkl'
    single_d_view_path = '/liushichao/VANet/single-branch_d-view_pos-&-neg_infer_part-neg.pkl'

    single_s_data = pickle.load(open(single_s_view_path, 'rb'))
    single_d_data = pickle.load(open(single_d_view_path, 'rb'))

    single_s_pos = single_s_data[0]['pos']
    single_s_neg = single_s_data[0]['neg']
    single_d_pos = single_d_data[0]['pos']
    single_d_neg = single_d_data[0]['neg']

    plot_distribution(name='single', s_pos=single_s_pos, s_neg=single_s_neg,
                      d_pos=single_d_pos, d_neg=single_d_neg, fig=None)

