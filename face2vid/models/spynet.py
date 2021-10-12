import getopt
import math
import sys
import torch
import torchfile
# import torch.utils.serialization
from torch.nn.functional import upsample

arguments_strModel = 'sintel-final'

if arguments_strModel == 'chairs-clean':
    arguments_strModel = '4'

elif arguments_strModel == 'chairs-final':
    arguments_strModel = '3'

elif arguments_strModel == 'sintel-clean':
    arguments_strModel = 'C'

elif arguments_strModel == 'sintel-final':
    arguments_strModel = 'F'

elif arguments_strModel == 'kitti-final':
    arguments_strModel = 'K'

# end

##########################################################

class SpyNetwork(torch.nn.Module):
    def __init__(self, optical_flow_model='F'):
        super(SpyNetwork, self).__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()
            # end

            def forward(self, tensorInput):
                tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

                return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

                if intLevel == 5:
                    if arguments_strModel == '3' or arguments_strModel == '4':
                        intLevel = 4 # the models trained on the flying chairs dataset do not come with weights for the sixth layer
                    # end
                # end

                for intConv in range(5):
                    # torchfile.load
                    # self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.t7'))
                    # self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.t7'))
                    aa = torchfile.load('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.t7')
                    bb = torchfile.load('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.t7')
                    self.moduleBasic[intConv * 2].weight.data.copy_(torch.tensor(aa))
                    self.moduleBasic[intConv * 2].bias.data.copy_(torch.tensor(bb))
                # end
            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)
            # end
        # end

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow):
                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    # self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
                    self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1)
                # end

                tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
                # print(tensorInput)
                # print(self.tensorGrid)
                # print(tensorFlow)
                return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid.cuda() + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
            # end
        # end

        self.modulePreprocess = Preprocess()

        self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        self.moduleBackward = Backward()
    # end

    def estimate(self, tensorFirst, tensorSecond):
        tensorFlow = []
        # print('Before preprocessing ',tensorFirst[0].shape)
        tensorFirst = [ self.modulePreprocess(tensorFirst) ]
        tensorSecond = [ self.modulePreprocess(tensorSecond) ]
        # print('After ', tensorFirst[0].shape)
        for intLevel in range(5):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
            # end
        # end

        # print('After ', tensorFirst[0].shape)
        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.upsample(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=(0, 0, 0, 1), mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=(0, 1, 0, 0), mode='replicate')

            tensorFlow = self.moduleBasic[intLevel](torch.cat(( tensorFirst[intLevel], self.moduleBackward(tensorSecond[intLevel], tensorUpsampled), tensorUpsampled ), 1)) + tensorUpsampled
        # end

        return tensorFlow

    def forward(self, tensorFirst, tensorSecond):
        assert(tensorFirst.size(2) == tensorSecond.size(2))
        assert(tensorFirst.size(3) == tensorSecond.size(3))

        intWidth = tensorFirst.size(3)
        intHeight = tensorFirst.size(2)

        if intWidth % 32 == 0 and intHeight % 32 == 0:
            tensorFlow = self.estimate(tensorFirst, tensorSecond)
        else:
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

            tensorPreprocessedFirst = torch.nn.functional.upsample(input=tensorFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tensorPreprocessedSecond = torch.nn.functional.upsample(input=tensorSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

            tensorFlow = self.estimate(tensorPreprocessedFirst, tensorPreprocessedSecond)
            tensorFlow = torch.nn.functional.upsample(input=tensorFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow[0]

