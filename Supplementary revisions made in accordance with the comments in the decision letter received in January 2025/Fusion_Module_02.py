
import torch
import torch.nn as nn

class Fusion_Module_02(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Fusion_Module_02, self).__init__()

        self.vertical_conv  =  nn.Conv2d(in_channels, out_channels,
                                         kernel_size = (3, 1),
                                         padding = (1, 0),
                                         bias=False)

        self.horizontal_conv = nn.Conv2d(in_channels, out_channels,
                                         kernel_size = (1, 3),
                                         padding = (0, 1),
                                         bias=False)

        self.sigmoid = nn.Sigmoid()

        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input_low, input_high):  # (1, 32, 64, 64)


        low_level = self.sigmoid(self.vertical_conv(self.horizontal_conv(input_low)))

        high_level = self.conv_1x1(input_high)                        # (1, 32, 64, 64)

        # c1 = input_low_w * high_level    # (1, 32, 64, 64)
        # c2 = input_low_h * high_level    # (1, 32, 64, 64)

        output = low_level + high_level

        return output

# 输入张量形状 (1, 32, 64, 64)
input_tensor_1 = torch.randn(1, 8, 128, 128)
input_tensor_2 = torch.randn(1, 8, 128, 128)

model = Fusion_Module_02(8, 8)

output = model(input_tensor_1, input_tensor_2)

# print(output.size())
