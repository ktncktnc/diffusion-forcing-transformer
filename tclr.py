import sys
sys.path.append('..')
import numpy as np 
import torch.nn as nn
import torch
import torchvision
from torch.cuda.amp import autocast
from algorithms.contrastive_dfot.losses import InfoNCELoss, global_local_temporal_contrastive


#This file is taken from Official PyTorch repository for video models: https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py with change in forward function


import torch
import torch.nn as nn

# from torchvision.models.utils import load_state_dict_from_url
from torch.cuda.amp import autocast

__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        with autocast():

            x, clip_type = x

            x = self.stem(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)

            
            return x, clip_type

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
   


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def r2plus1d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)

class mlp(nn.Module):

    def __init__(self, final_embedding_size = 128, use_normalization = True):
        
        super(mlp, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(512,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):
        with autocast():

            x, clip_type = x

            if clip_type == 'd':
                # Global Dense Representation, this will be used for IC, LL and GL losses
                # ('input x', x.shape)
                x = self.temp_avg(x)
                x = x.flatten(1)
                # print('x flatten', x.shape)
                
                x = self.relu(self.bn1(self.fc1(x)))
                # print('x relu', x.shape)

                x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
                return x
            
            elif clip_type == 's':
                # Global Sparse Representation, this will be used for the IC loss
                print('x', x.shape)
                gsr = self.temp_avg(x)
                gsr = gsr.flatten(1)
                print('gsr flatten', gsr.shape)
                gsr = self.relu(self.bn1(self.fc1(gsr)))
                print('gsr relu', gsr.shape)
                gsr = nn.functional.normalize(self.bn2(self.fc2(gsr)), p=2, dim=1)
                print('gsr', gsr.shape)
                # Local Sparse Representations, These will be used for the GL losses
                x1, x2, x3, x4 = [nn.functional.normalize(self.bn2(self.fc2(\
                                    self.relu(self.bn1(self.fc1(x[:,:,i,:,:].flatten(1))))))) for i in range(4)]
                print('x1', x1.shape)
                return gsr, x1, x2, x3, x4
            
            else:
                return None, None, None, None, None


def build_r3d_backbone(): #Official PyTorch R3D-18 model taken from https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
    
    model = r3d_18(pretrained = False, progress = False)
    #Expanding temporal dimension of the final layer by replacing temporal stride with temporal dilated convolution, this doesn't cost any additional parameters!

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

def build_r3d_mlp():
    f = build_r3d_backbone()
    g = mlp()
    model = nn.Sequential(f,g)
    return model
    

if __name__ == '__main__':
    
    input = torch.rand(5, 3, 16, 64, 64).cuda() # 5 clips of 16 frames, 3 channels, 112x112 resolution
    model = build_r3d_mlp()
    # print(model)
    # print()
    model.eval()
    model.cuda()

    sparse_clip = torch.randn(5, 3, 15, 64, 64).to('cuda')
    dense_clip0 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    dense_clip1 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    dense_clip2 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    dense_clip3 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    a_sparse_clip = torch.randn(5, 3, 19, 64, 64).to('cuda')
    a_dense_clip0 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    a_dense_clip1 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    a_dense_clip2 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    a_dense_clip3 = torch.randn(5, 3, 17, 64, 64).to('cuda')
    
    out_sparse = []
    # out_dense will have output in this order : [d0,d1,d2,d3,a_d0,...]
    out_dense = [[],[]]

    out_sparse.append(model((sparse_clip.cuda(),'s'))) # (5, 128)
    exit(0)
    out_sparse.append(model((a_sparse_clip.cuda(),'s')))

    out_dense[0].append(model((dense_clip0.cuda(),'d')))
    out_dense[0].append(model((dense_clip1.cuda(),'d')))
    out_dense[0].append(model((dense_clip2.cuda(),'d')))
    out_dense[0].append(model((dense_clip3.cuda(),'d')))

    out_dense[1].append(model((a_dense_clip0.cuda(),'d')))
    out_dense[1].append(model((a_dense_clip1.cuda(),'d')))
    out_dense[1].append(model((a_dense_clip2.cuda(),'d')))
    out_dense[1].append(model((a_dense_clip3.cuda(),'d')))

    criterion = InfoNCELoss(device = 'cuda', batch_size=out_sparse[0][0].shape[0], temperature=0.1, use_cosine_similarity = False).cuda()
    criterion_local_local = InfoNCELoss(device = 'cuda', batch_size=4, temperature=0.1, use_cosine_similarity = False).cuda()

    print('out_sparse[0][0]', out_sparse[0][0].shape)
    print('out_sparse[1][0]', out_sparse[1][0].shape)

    print('torch.stack(out_dense[0],dim=1)[ii],', torch.stack(out_dense[0],dim=1).shape)

    print('model((sparse_clip.cuda(),"s"))', model((sparse_clip.cuda(),'s')).shape)
    print('torch.stack(out_sparse[ii][1:],dim=1)', torch.stack(out_sparse[0][1:],dim=1).shape)
    print('torch.stack(out_sparse[ii][0:],dim=1)', torch.stack(out_sparse[0][0:],dim=1).shape)
    print('torch.stack(out_dense[jj],dim=1)', torch.stack(out_dense[0],dim=1).shape)
