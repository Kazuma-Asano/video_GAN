import os
import torch
import torch.nn as nn
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#
# def get_norm_layer(norm_type):
#     if norm_type == 'batch':
#         norm_layer = nn.BatchNorm2d
#     elif norm_type == 'instance':
#         norm_layer = nn.InstanceNorm2d
#     else:
#         print('normalization layer [%s] is not found' % norm_type)
#     return norm_layer

def define_G(gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netG = Generator()

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

def define_D(gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator()

    if len(gpu_ids) > 0:
        netD.cuda()
    netD.apply(weights_init)
    return netD

def print_network(net):
    num_parms = 0
    for parm in net.parameters():
        num_parms += parm.numel()
    print(net)
    print('Total number of parameters:{}'.format(num_parms))


############# compornent #######################################################
def conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def deconv2d_first(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (4,4))

def deconv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def deconv3d_first(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,4,4))

def deconv3d_video(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,1,1))

def deconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def batchNorm4d(num_features, eps = 1e-5): #input: N, C, H, W
    return nn.BatchNorm2d(num_features, eps = eps)

def batchNorm5d(num_features, eps = 1e-5): #input: N, C, D, H, W
    return nn.BatchNorm3d(num_features, eps = eps)

def relu(inplace = True):
    return nn.ReLU(inplace)

def lrelu(negative_slope = 0.2, inplace = True):
    return nn.LeakyReLU(negative_slope, inplace)
################################################################################

class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        # layer1~4
        self.model = nn.Sequential(
                conv2d(3,128),
                relu(),
                conv2d(128,256),
                batchNorm4d(256),
                relu(),
                conv2d(256,512),
                batchNorm4d(512),
                relu(),
                conv2d(512,1024),
                batchNorm4d(1024),
                relu(),
                )
    def forward(self,x):
        print('G_encode Input =', x.size()) # [Batch, channel=3, width, height]
        out = self.model(x)
        print('G_encode Output =', out.size()) # [Batch, 1024, 4, 4]
        return out

class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                deconv2d(1024,512), #[-1,512,4,4]
                batchNorm4d(512),
                relu(),
                deconv2d(512,256),
                batchNorm4d(256),
                relu(),
                deconv2d(256,128),
                batchNorm4d(128),
                relu(),
                deconv2d(128,3),
                nn.Tanh()
                )

    def forward(self, x):
        print('G_background Input =', x.size())
        out = self.model(x)
        print('G_background Output =', out.size())
        return out

class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
                deconv3d_video(1024,1024), #[-1,512,4,4]
                batchNorm5d(1024),
                relu(),
                deconv3d(1024,512),
                batchNorm5d(512),
                relu(),
                deconv3d(512,256),
                batchNorm5d(256),
                relu(),
                deconv3d(256,128),
                batchNorm5d(128),
                relu(),
                )
    def forward(self, x):
        print('G_video input =', x.size())
        out = self.model(x)
        print('G_video output =', out.size())
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode = G_encode()
        self.background = G_background()
        self.video = G_video()

        self.gen_net = nn.Sequential(deconv3d(128,3), nn.Tanh())
        self.mask_net = nn.Sequential(deconv3d(128,1), nn.Sigmoid())

    def forward(self,x):
        print('Generator input =',x.size()) # [Batch, channel=3, frame=1, width, height] : image
        x = x.squeeze(2) # [Batch, channel=3, width, height]
        encoded = self.encode(x)
        # print(encoded.size())  # [batch, 1024, 4, 4]
        encoded = encoded.unsqueeze(2)
        # print(encoded.size())  # [batch, 1024, 1, 4, 4]
        video = self.video(encoded) #[-1, 128, 16, 32, 32], which will be used for generating the mask and the foreground
        print('Video size = ', video.size()) #[batch, 128, 16, 32, 32]

        foreground = self.gen_net(video) #[-1,3,32,64,64]
        #print('Foreground size =', foreground.size())

        mask = self.mask_net(video) #[-1,1,32,64,64]
        #print('Mask size = ', mask.size())
        mask_repeated = mask.repeat(1,3,1,1,1) # repeat for each color channel. [-1, 3, 32, 64, 64]
        #print('Mask repeated size = ', mask_repeated.size())

        x = encoded.view((-1,1024,4,4))
        background = self.background(x) # [-1,3,64,64]
        #print('Background size = ', background.size())
        background_frames = background.unsqueeze(2).repeat(1,1,32,1,1) # [-1,3,32,64,64]
        out = torch.mul(mask,foreground) + torch.mul(1-mask, background_frames)
        #print('Generator out = ', out.size())
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                conv3d(3, 128), #[-1, 64, 16, 32, 32]
                lrelu(0.2),
                conv3d(128,256), #[-1, 126,8,16,16]
                batchNorm5d(256, 1e-3),
                lrelu(0.2),
                conv3d(256,512), #[-1,256,4,8,8]
                batchNorm5d(512, 1e-3),
                lrelu(0.2),
                conv3d(512,1024), #[-1,512,2,4,4]
                batchNorm5d(1024,1e-3),
                lrelu(0.2),
                conv3d(1024,2, (2,4,4), (1,1,1), (0,0,0)) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        #self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])

    def forward(self, x):
        out = self.model(x).squeeze()
        #out = self.mymodules[0](out)
        return out


if __name__ == '__main__':
    for i in range(1):
        """Check Discriminator"""
        netD = define_D(gpu_ids=[0])
        print_network(netD)
        x_D = Variable(torch.rand([20, 3, 32, 64, 64]).cuda()) # Batch, channel, frame, width, height
        print('Discriminator input', x_D.size())
        out_D = netD(x_D).squeeze()
        print('Discriminator out ', out_D.size())

        print('-'*50)

        """Check Generator"""
        netG = define_G(gpu_ids=[0])
        print_network(netG)
        x_G = Variable(torch.rand([20,3,1,64,64]).cuda()) # Batch, channel, frame, width, height
        print('Generator input', x_G.size())
        out_G = netG(x_G)
        print('Generator out ', out_G.size())
        print(type(out_G.data[0]))
        print(out_G.data[0].size())
        #
        # """Check Generator"""
        # x = Variable(torch.rand([13,3,64,64])).cuda()
        # #x = Variable(torch.rand([13,3,1,64,64]))
        # print('Generator input', x.size())
        # model = Generator().cuda()
        # out = model(x)
        # print('Generator out ', out.size())
