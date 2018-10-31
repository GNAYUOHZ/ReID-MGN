import os
import numpy as np
from scipy.spatial.distance import cdist

import torch
from torch.optim import Adam, lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from functions import mean_ap, cmc, re_ranking
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, loader):
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = self.get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def test(self):

        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.model.eval()
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        print('epoch:{:d} lr:{:.6f} [   re_rank] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(epoch, lr, m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        print('epoch:{:d} lr:{:.6f} [no re_rank] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(epoch, lr, m_ap, r[0], r[2], r[4],r[9]))

    def get_optimizer(self, net):

        if opt.freeze:

            for p in net.parameters():
                p.requires_grad = True
            for q in net.backbone.parameters():
                q.requires_grad = False

            optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, weight_decay=5e-4,amsgrad=True)

        else:

            optimizer = Adam(net.parameters(), lr=opt.lr, weight_decay=5e-4, amsgrad=True)

        return optimizer

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to('cuda')
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
        return features


if __name__ == '__main__':

    loader = Data()
    model = MGN()
    loss = Loss()
    reid = Main(model, loss, loader)

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch+1):
            print('\nepoch', epoch)
            reid.train()
            if epoch % 50 == 0:
                print('\nstart evaluate')
                reid.test()
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        reid.test()
