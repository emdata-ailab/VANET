"""
Appends two convolutional branches to transform all images to 2N features.
@ Author: liushichao
"""
import torch
import copy
from torch import Tensor, nn
from torch.jit.annotations import Optional
from torchvision.models import GoogLeNet, googlenet
from torchvision.models.resnet import ResNet, _resnet, Bottleneck
import torch.utils.model_zoo as model_zoo


def set_diag_to_zreo(metrix):
    """
    :param metrix: tensor
    :return:
    """
    diag = torch.diag(metrix)  # get diag value
    embed_diag = torch.diag_embed(diag)  # reshape to views mask's metrix dimension.
    final_metrix = metrix - embed_diag  # set diag value to zero, what real views mask we need.

    return final_metrix

class vanet_googlenet_wrapper(GoogLeNet):
    """
    Wrapped from 'GoogleNet'.
    """

    def __init__(self):
        super(vanet_googlenet_wrapper, self).__init__(num_classes=1000,
                                                      aux_logits=True,
                                                      transform_input=False,
                                                      init_weights=None,
                                                      blocks=None)
        # if pre_trained:
        model = googlenet(pretrained=True)
        print(f'Loaded pretrained parameters from GoogleNet.')

        self.in_planes = 1024
        self.num_classes = 575

        # define shared_conv
        self.shared_conv = nn.Sequential(*list(model.children())[:8])

        # define branch_convs
        self.branch_conv = nn.Sequential(*list(model.children())[8:-2])

        # define classifier
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.s_branch_conv = copy.deepcopy(self.branch_conv)
        self.d_branch_conv = copy.deepcopy(self.branch_conv)

        self.s_classifier = copy.deepcopy(self.classifier)
        self.d_classifier = copy.deepcopy(self.classifier)

    def forward(self, x):  # type: (Tensor) -> GoogLeNetOutputs
        """
        :param x: tensor (N * 3 * 224 * 224)
        :return:
        """

        batch_images = x['image']
        shared_feat = self.shared_conv(batch_images)

        s_feat = self.s_branch_conv(shared_feat)
        s_feat = s_feat.view(s_feat.shape[0], -1)  # Flatten to (B, 1024)
        s_cls_score = self.s_classifier(s_feat)

        d_feat = self.d_branch_conv(shared_feat)
        d_feat = d_feat.view(d_feat.shape[0], -1)  # Flatten to (B, 1024)
        d_cls_score = self.d_classifier(d_feat)

        # return
        res = {
            's_feat': s_feat,
            's_cls_score': s_cls_score,
            'd_feat': d_feat,
            'd_cls_score': d_cls_score,
        }

        return res

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class vanet_resnet50_wrapper(ResNet):
    """
    Wrapped from resnet50.
    Note: there have three version of network constructions. 1 version of single-branch and 2 versions of two-branch. 
          You can try this following three backbone respectively for performance comparasion. 

          prefix s: similar views
          prefix d: different views
    """
    # shared_conv and branch_conv network version 1.
    # def __init__(self):
    #     super(vanet_resnet50_wrapper, self).__init__(
    #                                                 block=Bottleneck,
    #                                                 layers=[3, 4, 6, 3],
    #                                                 num_classes=1000,
    #                                                 zero_init_residual=False,
    #                                                 groups=1,
    #                                                 width_per_group=64,
    #                                                 replace_stride_with_dilation=None,
    #                                                 norm_layer=None
    #                                                 )

    #     # load pre_trained resnet model.
    #     model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True)
    #     print("Load pretrained model from resnet50.")

    #     # add BNNeck
    #     self.neck = 'bnneck'
    #     self.neck_feat = 'after'

    #     self.in_planes = 2048
    #     self.num_classes = 575

    #     # initial networks.
    #     # construct shared_conv: get first 4 layers from class ResNet.
    #     self.shared_conv = nn.Sequential(*list(model.children())[:4])

    #     # construct branch_conv: get remain layers of class ResNet,
    #     # except for the last fully connection layer.
    #     self.branch_conv = nn.Sequential(*list(model.children())[4:-1])


    #     # add intial BNNeck
    #     self.bnneck = nn.BatchNorm1d(self.in_planes)
    #     self.bnneck.bias.requires_grad_(False)  # no shift
    #     self.bnneck.apply(weights_init_kaiming)

    #     # initial classifier.
    #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
    #     self.classifier.apply(weights_init_classifier)

    #     # construct two seperate branch network space.
    #     self.s_branch_conv = copy.deepcopy(self.branch_conv)  # similar view branch CNN.
    #     self.s_bnneck = copy.deepcopy(self.bnneck)
    #     self.s_classifier = copy.deepcopy(self.classifier)

    #     self.d_branch_conv = copy.deepcopy(self.branch_conv)  # different view branch CNN.
    #     self.d_bnneck = copy.deepcopy(self.bnneck)
    #     self.d_classifier = copy.deepcopy(self.classifier)

    # shared_conv and branch_conv network version 2: branch-conv layer4
    # def __init__(self):
    #     super(vanet_resnet50_wrapper, self).__init__(
    #         block=Bottleneck,
    #         layers=[3, 4, 6, 3],
    #         num_classes=1000,
    #         zero_init_residual=False,
    #         groups=1,
    #         width_per_group=64,
    #         replace_stride_with_dilation=None,
    #         norm_layer=None
    #     )
    #
    #     # load pre_trained resnet model.
    #     model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True)
    #     print("Load pretrained model from resnet50.")
    #
    #     # add BNNeck
    #     self.neck = 'bnneck'
    #     self.neck_feat = 'after'
    #
    #     self.in_planes = 2048
    #     self.num_classes = 575
    #
    #     # initial networks.
    #     # construct shared_conv: get first 7 layers from class ResNet.
    #     self.shared_conv = nn.Sequential(*list(model.children())[:7])
    #
    #     # construct branch_conv: get remain one layer of class ResNet,
    #     # except for the last fully connection layer.
    #     self.branch_conv = nn.Sequential(*list(model.children())[7:-1])
    #
    #     # add intial BNNeck
    #     self.bnneck = nn.BatchNorm1d(self.in_planes)
    #     self.bnneck.bias.requires_grad_(False)  # no shift
    #     self.bnneck.apply(weights_init_kaiming)
    #
    #     # initial classifier.
    #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
    #     self.classifier.apply(weights_init_classifier)
    #
    #     # construct two seperate branch network space.
    #     self.s_branch_conv = copy.deepcopy(self.branch_conv)
    #     self.s_bnneck = copy.deepcopy(self.bnneck)
    #     self.s_classifier = copy.deepcopy(self.classifier)
    #
    #     self.d_branch_conv = copy.deepcopy(self.branch_conv)
    #     self.d_bnneck = copy.deepcopy(self.bnneck)
    #     self.d_classifier = copy.deepcopy(self.classifier)

    # shared_conv and branch_conv network version 2: branch-conv layer 3 and layer 4.  For best mAP(80.1) of two branch.
    def __init__(self):
        super(vanet_resnet50_wrapper, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
        )
    
        # load pre_trained resnet model.
        model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True)
        print("Load pretrained model from resnet50.")
    
        # add BNNeck
        self.neck = 'bnneck'
        self.neck_feat = 'after'
    
        self.in_planes = 2048
        self.num_classes = 575
    
        # initial networks.
        # construct shared_conv: get first 6 layers from class ResNet.
        self.shared_conv = nn.Sequential(*list(model.children())[:6])
    
        # construct branch_conv: get remain two layers of class ResNet,
        # except for the last fully connection layer.
        self.branch_conv = nn.Sequential(*list(model.children())[6:-1])
    
        # add intial BNNeck
        self.bnneck = nn.BatchNorm1d(self.in_planes)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)
    
        # initial classifier.
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
    
        # construct two seperate branch network space.
        self.s_branch_conv = copy.deepcopy(self.branch_conv)
        self.s_bnneck = copy.deepcopy(self.bnneck)
        self.s_classifier = copy.deepcopy(self.classifier)
    
        self.d_branch_conv = copy.deepcopy(self.branch_conv)
        self.d_bnneck = copy.deepcopy(self.bnneck)
        self.d_classifier = copy.deepcopy(self.classifier)


    def forward(self, x):
        """
        :param x: tensor (N * 3 * 224 * 224)
        :return:
        """

        # input
        batch_images = x['image']
        shared_feature = self.shared_conv(batch_images)

        # feature extraction
        s_feat = self.s_branch_conv(shared_feature)
        s_feat = s_feat.view(s_feat.shape[0], -1)  # Flatten to (B, 2048)
        s_feat_bn = self.s_bnneck(s_feat)  # normalize for angular softmax
        # s_cls_score = self.s_classifier(s_feat)  # torch.Size([64, 575])

        d_feat = self.d_branch_conv(shared_feature)
        d_feat = d_feat.view(d_feat.shape[0], -1)  # Flatten to (B, 2048)
        d_feat_bn = self.d_bnneck(d_feat)  # normalize for angular softmax
        # d_cls_score = self.d_classifier(d_feat)  # torch.Size([64, 575])

        if self.training:
            s_cls_score = self.s_classifier(s_feat_bn)  # torch.Size([64, 575])
            d_cls_score = self.d_classifier(d_feat_bn)  # torch.Size([64, 575])
            # return
            res = {
                's_feat': s_feat,
                's_cls_score': s_cls_score,
                'd_feat': d_feat,
                'd_cls_score': d_cls_score,
            }
            return res
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return {'s_feat': s_feat_bn, 'd_feat': d_feat_bn}
            else:
                # print("Test with feature before BN")
                return {'s_feat': s_feat, 'd_feat': d_feat}
