import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pdb
import math
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        x = f.view(f.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return f, x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    # # print (kwargs['num_classes'])
    # if pretrained:
    #     # model_path = '/mnt/lustre/wangzhouxia/Data_t1/rl_exposure_fusion_v1/368/decision_ps_v2_401_r1_ssim_fc99_b16_lr5/policy_decision_ps_v2_401_r1_ssim_fc99_b16_lr5_checkpoint_19.path.tar'
    #     model_path = '/mnt/lustre/wangzhouxia/Data_t1/rl_exposure_fusion_v1/0327/decision_gff_v2_438_ssim_lr4_c99/policy_decision_gff_v2_438_ssim_lr4_c99_checkpoint_29.path.tar'
    #     # model_path = '/data1/models/rl_exposure_fusion/policy_decision_ps_v2_401_r1_ssim_fc99_b16_lr5_checkpoint_19.path.tar'
    #     pre_model = torch.load(model_path)['state_dict']
    #     # pdb.set_trace()
    #     model_dict = model.state_dict()
    #     for k in pre_model.keys():
    #         if not model_dict.has_key(k):
    #             del pre_model[k]
    #     # pre_model['classifier.6.weight'] = torch.empty((kwargs['num_classes'], 4096)).normal_(0.0, 0.01)
    #     # pre_model['classifier.6.bias'] = torch.zeros(kwargs['num_classes'])

    #     model.load_state_dict(pre_model)

    #     model = AlexNet(**kwargs)
    # print (kwargs['num_classes'])
    if pretrained:
        pre_model = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        for k in pre_model.keys():
            if not model_dict.has_key(k):
                del pre_model[k]
        model.load_state_dict(pre_model)
        
    return model

class HistogramSpacialNet(nn.Module):

    def __init__(self, num_classes=1000, bins_num=32):
        super(HistogramSpacialNet, self).__init__()
        
        self.bins_num = bins_num

        self.features = nn.Sequential(
            nn.Conv2d(self.bins_num*3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*4*4, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        f = self.features(x)
        x = f.view(f.size(0), 128*4*4)
        x = self.classifier(x)
        return f, x

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 3:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class HistogramNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(HistogramNet, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(32*21, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 3:
                    m.weight.data.normal_(0, 0.1)
                else:
                    m.weight.data.normal_(0, 0.01)

def histogramspacialnet(pretrained=False, bins_num=32):
    # weight_path = '/mnt/lustre/wangzhouxia/Data_t1/rl_exposure_fusion_v1/0312/decision_ps_v2_401_r1_ssim_hist32_fc199_spatial_lr3/policy_decision_ps_v2_401_r1_ssim_hist32_fc199_spatial_lr3_checkpoint_199.path.tar'
    # weight_path = '/mnt/lustre/wangzhouxia/Data_t1/rl_exposure_fusion_v1/0327/decision_gff_v2_438_ssim_lr4_c99_hist/policy_decision_gff_v2_438_ssim_lr4_c99_hist_checkpoint_399.path.tar'
    # weight_path = '/data1/models/rl_exposure_fusion/policy_decision_ps_v2_401_r1_ssim_hist32_fc199_spatial_lr3_checkpoint_199.path.tar'
    model = HistogramSpacialNet(bins_num=bins_num)
    # model= HistogramNet()
    # if pretrained:
    #     pre_model = torch.load(weight_path)['state_dict']
    #     model_dict = model.state_dict()
    #     for k in pre_model.keys():
    #         if not model_dict.has_key(k):
    #             del pre_model[k]
    #     model.load_state_dict(pre_model)
    return model

class l2norm(nn.Module):
    def __init__(self):
        super(l2norm,self).__init__()

    def forward(self,input,epsilon = 1e-7):
        assert len(input.size()) == 2,"Input dimension requires 2,but get {}".format(len(input.size()))
        
        norm = torch.norm(input,p = 2,dim = 1,keepdim = True)
        output = torch.div(input,norm+epsilon)
        return output

class DecisionNet(nn.Module):
    def __init__(self, pretrained, num_classes=1000, bins_num=32):
        super(DecisionNet, self).__init__()

        self.semantic = alexnet(pretrained)
        self.histogram = histogramspacialnet(pretrained, bins_num=32)
        self.l2norm = l2norm()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024+4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, image, hist):
        sem_conv, fsem = self.semantic(image)
        his_conv, fhis = self.histogram(hist)

        # fhis = self.histogram(hist)
        # print(fsem.shape, fhis.shape)
        fsem = self.l2norm(fsem)
        fhis = self.l2norm(fhis)
        fcat = torch.cat([fsem, fhis], dim=1)
        output = self.classifier(fcat)
        # return sem_conv, his_conv, fsem, fhis, output
        return output

    def _initialize_weights(self):
        cnt = 0
        for m in self.classifier.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 2:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


from torch.autograd import Variable
import numpy as np
if __name__=='__main__':
    a = DecisionNet(num_classes=10)
    seg = torch.empty((1, 3, 224, 224)).normal_(0., 0.01)
    his = torch.empty((1, 32*3, 4, 4)).normal_(0., 0.01)
    # print(input.shape)

    output = a(seg, his)
    print(a)