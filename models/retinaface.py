import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        # 对，这是conv, NCHW->NHWC->N-H*W-C; 无bn relu; anchor_num=2说明每个像素点预设anchor为2，然后分类需要输出2，bbox需要4，landmarks需要10，所以与论文源码的anchor预设是一致的；
        # 但是没有 将ssh的结果级联，源码是将ssh的结果上采样再“+”，即为 cascade
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

# 此类为人脸姿态估计(yaw-pitch-roll)共有的分类 类-66类
class PoseHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*66,kernel_size=(1,1),stride=1,padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()  # 一般view前面，需要设置contiguous，将数据底层连续；直接reshape是不太好的！

        return out.view(out.shape[0], -1, 66)

class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        # 获取中间层输出
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        # 参考口罩分支，ssh单独新建
        self.ssh1_pose = SSH(out_channels, out_channels)
        self.ssh2_pose = SSH(out_channels, out_channels)
        self.ssh3_pose = SSH(out_channels, out_channels)
        # 对应ssh 的 框-关键点-2分类
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        # 对应ssh_pose 的 yaw-pitch-roll 的 3个66分类
        self.Pose_yaw_Head = self._make_pose_yaw_pitch_roll_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.Pose_pitch_Head = self._make_pose_yaw_pitch_roll_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.Pose_roll_Head = self._make_pose_yaw_pitch_roll_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def _make_pose_yaw_pitch_roll_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        pose_head = nn.ModuleList()
        for i in range(fpn_num):
            pose_head.append(PoseHead(inchannels, anchor_num))  # append添加Module
        return pose_head


    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # SSH Pose
        feature1_pose = self.ssh1_pose(fpn[0])
        feature2_pose = self.ssh2_pose(fpn[1])
        feature3_pose = self.ssh3_pose(fpn[2])
        features_pose = [feature1_pose, feature2_pose, feature3_pose]
        # 与RetinaFace细节区别，没有P5最后新建1层，只有最后3层；然后3层ssh输出是独立的，没有cascade做逐级相加“+”输出，且fpn的inchannel有差异；原始retinaface也没有cascade的ssh

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        yaw_preds = torch.cat([self.Pose_yaw_Head[i](feature) for i, feature in enumerate(features_pose)], dim=1)
        pitch_preds = torch.cat([self.Pose_pitch_Head[i](feature) for i, feature in enumerate(features_pose)], dim=1)
        roll_preds = torch.cat([self.Pose_roll_Head[i](feature) for i, feature in enumerate(features_pose)], dim=1)

        if self.phase == 'train':
            # 建议 采用 dict 返回，包括数据读取返回
            output = (bbox_regressions, classifications, ldm_regressions, yaw_preds, pitch_preds, roll_preds)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions, yaw_preds, pitch_preds, roll_preds)  # 测试ypr需要进行softmax
        return output

