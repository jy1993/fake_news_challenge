import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision.models import DenseNet
from collections import OrderedDict
from torchvision.models.densenet import _DenseLayer, _DenseBlock, _Transition
from torchvision.models.inception import InceptionA

class HighWay(nn.Module):
        """docstring for HighWay"""
        def __init__(self, input_size):
            super(HighWay, self).__init__()
            self.input_size = input_size
            self.wh = nn.Parameter(torch.Tensor(input_size, input_size))
            self.bh = nn.Parameter(torch.Tensor(input_size))
            self.wt = nn.Parameter(torch.Tensor(input_size, input_size))
            self.bt = nn.Parameter(torch.Tensor(input_size))
            self.reset_parameters()

        def forward(self, x):
            '''
            x: B * T * H  ==>  B * T * H
            '''
            T = F.sigmoid(torch.matmul(x, self.wt) + self.bt)
            C = 1 - T
            out = T * F.relu(torch.matmul(x, self.wh) + self.bh) + C * x
            return out

        def reset_parameters(self):
            stdv = 1.0 / np.sqrt(self.input_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

class Esim(nn.Module):
        """docstring for Esim"""
        def __init__(self, pretrained, hid_size=100, dropout=0.5):
            super(Esim, self).__init__()
            vocab_size, emb_size = pretrained.shape
            self.dropout = dropout
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float())
            self.gru_enc1 = nn.LSTM(input_size=emb_size, hidden_size=hid_size, 
                batch_first=True, bidirectional=True)
            self.gru_comp1 = nn.LSTM(input_size=hid_size * 8, hidden_size=hid_size, 
                batch_first=True, bidirectional=True)

            # self.hn = HighWay(hid_size)
            self.fc1 = nn.Linear(hid_size * 8, hid_size)
            self.last_layer = nn.Linear(hid_size, 4)

        def forward(self, x1, x2):
            # to shape B * T1 * T2
            x1_mask = x1.eq(0).unsqueeze(2)
            x2_mask = x2.eq(0).unsqueeze(1)
            
            x1 = self.embedding(x1)
            x2 = self.embedding(x2)

            # x1 = F.dropout(self.hn(x1), p=self.dropout, training=self.training)
            # x2 = F.dropout(self.hn(x2), p=self.dropout, training=self.training)

            x1, h1 = self.gru_enc1(x1)
            x2, h2 = self.gru_enc1(x2)

            # attention
            x2_t = torch.transpose(x2, 1, 2)
            attention = torch.bmm(x1, x2_t)
            
            attention.masked_fill_(x1_mask, 1e-6).masked_fill_(x2_mask, 1e-6)
            w1_attn = F.softmax(attention, dim=2)
            w2_attn = F.softmax(attention, dim=1)
            w2_attn_t = torch.transpose(w2_attn, 1, 2)

            x1_attn = torch.matmul(w1_attn, x2)
            x2_attn = torch.matmul(w2_attn_t, x1)

            x1_sub = x1 - x1_attn
            x2_sub = x2 - x2_attn

            x1_mul = x1 * x1_attn
            x2_mul = x2 * x2_attn

            x1_concat = torch.cat((x1, x1_attn, x1_sub, x1_mul), 2)
            x2_concat = torch.cat((x2, x2_attn, x2_sub, x2_mul), 2)

            x1_concat, _ = self.gru_comp1(x1_concat)
            x2_concat, _ = self.gru_comp1(x2_concat)

            x1_max_pooling, _ = torch.max(x1_concat, 1)
            x2_max_pooling, _ = torch.max(x2_concat, 1)
            x1_avg_pooling = torch.mean(x1_concat, 1)
            x2_avg_pooling = torch.mean(x2_concat, 1)

            x = torch.cat((x1_max_pooling, x1_avg_pooling, x2_max_pooling, x2_avg_pooling), 1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.softmax(self.last_layer(x))
            return x, w1_attn, w2_attn
        
class MATCHING(nn.Module):
    """docstring for MATCHING"""
    def __init__(self, l, hid_size):
        super(MATCHING, self).__init__()
        self.hid_size = hid_size
        self.w = nn.Parameter(torch.Tensor(l, hid_size))
        self.reset_parameters()

    def forward(self, v1, v2, mode='others'):
        # if mode == 'full':
        # v1: B * T * H, v2: B * T * H
        v1 = v1.unsqueeze(2)
        v1 = self.w * v1
        v2 = v2.unsqueeze(2)
        v2 = self.w * v2
        dim = 3
        if mode == 'max_pooling':
            norm1 = v1.norm(p=2, dim=3, keepdim=True)
            norm2 = v2.norm(p=2, dim=3, keepdim=True)
            d = norm1.transpose(1, 2) * norm2.permute(0, 2, 3, 1)
            # B * l * T * H
            v1 = v1.transpose(1, 2)
            # B * l * H * T
            v2 = v2.permute(0, 2, 3, 1)
            # B * T * T * l
            out = (torch.matmul(v1, v2) / (d + 1e-8)).permute(0, 2, 3, 1)
        else:
            out = F.cosine_similarity(v1, v2, dim=dim)
        return out

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hid_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)   

class BIMPM(nn.Module):
    """docstring for BIMPM"""
    def __init__(self, pretrained, hid_size=100, l=20, dropout=0.2):
        super(BIMPM, self).__init__()
        self.dropout = dropout
        vocab_size, emb_size = pretrained.shape
        self.hid_size = hid_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float())
        self.gru_enc = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
            batch_first=True, bidirectional=True)
        self.gru_comp = nn.LSTM(input_size=l * 10, hidden_size=hid_size, 
            batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(hid_size * 8, hid_size)
        self.bn1 = nn.BatchNorm1d(hid_size * 8)
        # self.fc2 = nn.Linear(300, 100)
        # self.bn2 = nn.BatchNorm1d(300)
        self.last_layer = nn.Linear(hid_size, 4)
        self.dp = nn.Dropout(dropout)
        self.match_fun = MATCHING(l, hid_size)
        self.hn = HighWay(hid_size)
            
    def forward(self, x1, x2):
        bsz, max_len_h = x1.shape
        bsz, max_len_b = x2.shape

        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        x1 = self.dp(self.hn(x1))
        x2 = self.dp(self.hn(x2))

        x1, h1 = self.gru_enc(x1)
        x2, h2 = self.gru_enc(x2)

        # attention
        x2_t = torch.transpose(x2, 1, 2)
        attention = torch.bmm(x1, x2_t)
        w1_attn = F.softmax(attention, dim=2)
        w2_attn = F.softmax(attention, dim=1)
        w2_attn_t = torch.transpose(w2_attn, 1, 2)
        
        # get the max attention 
        w1_attn_max = attention - attention.max(dim=2, keepdim=True)[0]
        w2_attn_max = attention - attention.max(dim=1, keepdim=True)[0]
        
        w1_attn_max[w1_attn_max < 0] = 0
        w1_attn_max[w1_attn_max >=0] = 1

        w2_attn_max[w2_attn_max < 0] = 0
        w2_attn_max[w2_attn_max >=0] = 1
        
        w1_attn_max = w1_attn_max / w1_attn_max.sum(dim=2, keepdim=True)
        w2_attn_max = w2_attn_max / w2_attn_max.sum(dim=2, keepdim=True)

        x1_attn = torch.matmul(w1_attn, x2)
        x2_attn = torch.matmul(w2_attn_t, x1)
        
        x1_max_attn = torch.matmul(w1_attn_max, x2)
        x2_max_attn = torch.matmul(w2_attn_max.transpose(1, 2), x1)

        # full-matching
        # B * T * H
        x1_fw, x1_bw = x1[:, :, 0:self.hid_size], x1[:, :, self.hid_size:]
        x2_fw, x2_bw = x2[:, :, 0:self.hid_size], x2[:, :, self.hid_size:]
        full1 = self.match_fun(x1_fw, 
            x2[:, max_len_h - 1, 0:self.hid_size].unsqueeze(1).repeat(1, max_len_h, 1))
        full2 = self.match_fun(x1_bw, 
            x2[:, max_len_b - 1, self.hid_size:].unsqueeze(1).repeat(1, max_len_b, 1))
        full3 = self.match_fun(x2_fw, 
            x1[:, max_len_h - 1, 0:self.hid_size].unsqueeze(1).repeat(1, max_len_h, 1))
        full4 = self.match_fun(x2_bw, 
            x1[:, max_len_b - 1, self.hid_size:].unsqueeze(1).repeat(1, max_len_b, 1))

        # print(full1.size())
        # print(full2.size())

        # max-pooling-matching
        tmp1 = self.match_fun(x1_fw, x2_fw, mode='max_pooling')
        tmp2 = self.match_fun(x1_bw, x2_bw, mode='max_pooling')
        max_pooling1, _ = torch.max(tmp1, dim=1)
        max_pooling2, _ = torch.max(tmp1, dim=2)
        max_pooling3, _ = torch.max(tmp2, dim=1)
        max_pooling4, _ = torch.max(tmp2, dim=2)
        
        # avg-pooling-matching
        avg_pooling1 = torch.mean(tmp1, dim=1)
        avg_pooling2 = torch.mean(tmp1, dim=2)
        avg_pooling3 = torch.mean(tmp2, dim=1)
        avg_pooling4 = torch.mean(tmp2, dim=2)
        # print(max_pooling1.size())
        # print(max_pooling2.size())

        # attentive-matching
        attentive1 = self.match_fun(x1_fw, x1_attn[:, :, 0:self.hid_size])
        attentive2 = self.match_fun(x1_bw, x1_attn[:, :, self.hid_size:])
        attentive3 = self.match_fun(x2_fw, x2_attn[:, :, 0:self.hid_size])
        attentive4 = self.match_fun(x2_bw, x2_attn[:, :, self.hid_size:])
        # print(attentive1.size())
        # print(attentive2.size())
        # print(attentive3.size())
        # print(attentive4.size())

        # max-attentive-matching
        max_attentive1 = self.match_fun(x1_fw, x1_max_attn[:, :, 0:self.hid_size])
        max_attentive2 = self.match_fun(x1_bw, x1_max_attn[:, :, self.hid_size:])
        max_attentive3 = self.match_fun(x2_fw, x2_max_attn[:, :, 0:self.hid_size])
        max_attentive4 = self.match_fun(x2_bw, x2_max_attn[:, :, self.hid_size:])
        # # print(max_attentive1.size())
        # print(max_attentive2.size())
        # print(max_attentive3.size())
        # print(max_attentive4.size())
        
        x1_concat = torch.cat((full1, full2, attentive1, attentive2, 
            max_pooling1, max_pooling2, avg_pooling1, avg_pooling2, max_attentive1, max_attentive2), 2)
        x2_concat = torch.cat((full3, full4, attentive3, attentive4, 
            max_pooling3, max_pooling4, avg_pooling3, avg_pooling4, max_attentive3, max_attentive4), 2)
        # print('after concating...')
        # print(x1_concat.size(), x2_concat.size())
        x1_concat, _ = self.gru_comp(x1_concat)
        x2_concat, _ = self.gru_comp(x2_concat)
        # print('after gru..')
        # print(x1_concat.size(), x2_concat.size())

        x1_max_pooling, _ = torch.max(x1_concat, 1)
        x2_max_pooling, _ = torch.max(x2_concat, 1)
        x1_avg_pooling = torch.mean(x1_concat, 1)
        x2_avg_pooling = torch.mean(x2_concat, 1)

        x = torch.cat((x1_max_pooling, x1_avg_pooling, x2_max_pooling, x2_avg_pooling), 1)
        # print('after pooling...')
        # print(x.size())
        x = F.relu(self.dp(self.fc1(x)))
        # x = F.relu(self.dp(self.fc2(self.bn2(x)))
        x = F.softmax(self.last_layer(x))
        return x

class DIIN(nn.Module):
    """docstring for DIIN"""
    def __init__(self, pretrained, model_name, hid_size=100, dropout=0.5):
        super(DIIN, self).__init__()
        vocab_size, emb_size = pretrained.shape
        self.hid_size = hid_size
        self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float())
        self.gru_enc = nn.LSTM(input_size=emb_size, hidden_size=hid_size, 
            batch_first=True, bidirectional=False)
        
        if model_name == 'diin_my':
            self.features = nn.Sequential(
                nn.Conv2d(hid_size, 64, 1), 

                nn.Conv2d(64, 128, 7, 1, 3, bias=False), 
                nn.BatchNorm2d(128), 
                nn.ReLU(), 
                nn.AdaptiveMaxPool2d((32, 32)),

                nn.Conv2d(128, 128, 3, 1, 1, bias=False), 
                nn.BatchNorm2d(128), 
                nn.ReLU(), 
                nn.AdaptiveMaxPool2d((16, 16)), 
                
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), 
                nn.BatchNorm2d(256), 
                nn.ReLU(), 
                nn.AdaptiveMaxPool2d((8, 8)), 

                nn.Conv2d(256, 256, 3, 1, 1, bias=False), 
                nn.BatchNorm2d(256), 
                nn.ReLU(), 
                nn.AdaptiveMaxPool2d((4, 4)),

                nn.Conv2d(256, 512, 3, 1, 1, bias=False), 
                nn.BatchNorm2d(512), 
                nn.ReLU(), 
                nn.MaxPool2d(4),
                ) 

            self.fc1 = nn.Linear(512, 100)

        elif model_name == 'diin_densenet':
            #self.features = DNet(args)
            self.features = DNet(hid_size, block_config=(4, 4, 4), drop_rate=0.5, num_init_features=200)
            
            #self.features = DNet(args, block_config=(2, 2, 2), drop_rate=0.3, num_init_features=100)
            #self.fc1 = nn.Linear(2016, 100)
            self.fc1 = nn.Linear(2466, 100)
            #self.fc1 = nn.Linear(3425, 100)
        
        elif model_name == 'diin_inception':
            # input: 16 * 16
            self.features = nn.Sequential(
                nn.Conv2d(hid_size, 64, 1),

                InceptionA(64, 32), 
                nn.MaxPool2d(2),

                InceptionA(256, 32), 
                nn.MaxPool2d(2),

                nn.Conv2d(256, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512), 
                nn.ReLU(), 
                nn.AvgPool2d(4))

            self.fc1 = nn.Linear(512, 100)
        elif model_name == 'diin_inceptionB':
            # input: 24 * 24
            self.features = nn.Sequential(
                nn.Conv2d(hid_size, 100, 1),
                nn.BatchNorm2d(100),
                nn.ReLU(),

                InceptionA(100, 32), 
                nn.MaxPool2d(2),

                InceptionA(256, 64), 
                nn.MaxPool2d(2),
                
                InceptionA(288, 32),
                nn.MaxPool2d(2), 

                nn.Conv2d(288, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512), 
                nn.ReLU(), 
                nn.AvgPool2d(3))

            self.fc1 = nn.Linear(512, 100)
        self.last_layer = nn.Linear(100, 1)
        self.dp = nn.Dropout(self.dropout)
        self.hn = HighWay(hid_size)
            
    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        
        x1 = self.dp(self.hn(x1))
        x2 = self.dp(self.hn(x2))

        x1, _ = self.gru_enc(x1)
        x2, _ = self.gru_enc(x2)
        
        x1 = x1.unsqueeze(2) # B * T * 1 * H
        x2 = x2.unsqueeze(1) # B * 1 * T * H
        x = x1 * x2 # B * T * T * H
        #x_sub = x1 - x2
        #x = torch.cat((x, x_sub), dim=3)
        x = x.permute(0, 3, 1, 2) # B * H * T * T

        x = self.features(x)
        #print(x.size())
        x = x.squeeze()
        x = F.relu(self.dp(self.fc1(x)))
        x = F.sigmoid(self.last_layer(x))
        return x, x, x

class DNet(nn.Module):
    def __init__(self, hid_size=100, growth_rate=32, block_config=(2, 4, 4), num_init_features=64, bn_size=4, drop_rate=0.2, num_directions=1):
        super(DNet, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(hid_size * num_directions, num_init_features, 1))]))
        
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm4', nn.BatchNorm2d(num_features)) 
        
        #self.classifier = nn.Linear(num_features, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=2, stride=1).view(features.size(0), -1)
        #out = F.sigmoid(self.classifier(out))
        return out  