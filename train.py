#coding:utf-8
from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from   torch.autograd         import Variable
from   torch.utils.data       import DataLoader
from   torchvision.utils      import save_image, make_grid
from   networks               import define_G, define_D, print_network
from   loss                   import GANLoss
from   dataloader             import get_training_set, get_test_set
from   util                   import progress_bar
import torch.backends.cudnn as cudnn

import pandas as pd
import matplotlib.pyplot as plt


################################################################################
def train(epoch):
    netD.train()
    netG.train()
    total_D_loss = 0
    total_G_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        fake_b = netG(real_a)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        optimizerD.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g = loss_g_gan + loss_g_l1 ##
        loss_g.backward()
        optimizerG.step()

        total_D_loss += loss_d.item()
        total_G_loss += loss_g.item()

        # display
        # progress_bar(epoch, iteration, len(training_data_loader)+1, ': Loss_D: {:.4f}, Loss_G: {:.4f}'.format(loss_d.item(), loss_g.item()) )
        print('Epoch: {} Ite: {}/{} Loss_D: {:.4f}, Loss_G: {:.4f}'.format(epoch, iteration, len(training_data_loader)+1, loss_d.item(), loss_g.item()))
    avg_D_loss = total_D_loss / len(training_data_loader)
    avg_G_loss = total_G_loss / len(training_data_loader)

    return avg_D_loss, avg_G_loss
#
# def test(epoch):
#     netG.eval()
#     total_psnr = 0
#     total_loss = 0
#     for iteration, batch in enumerate(test_data_loader, 1):
#
#         with torch.no_grad():
#             input, target = Variable(batch[0]), Variable(batch[1])
#             if opt.cuda:
#                 input = input.cuda()
#                 target = target.cuda()
#
#             prediction = netG(input)
#             mse = criterionMSE(prediction, target)
#             total_loss += mse.item()
#             psnr = 10 * log10(1 / mse.item())
#             total_psnr += psnr
#             # display
#             progress_bar(epoch, iteration, len(test_data_loader)+1, ': Avg.Loss: {:.4f}, Avg.PSNR: {:.4f} dB'.format(mse.item(), psnr)  )
#             # print('Epoch: {} Ite: {}/{} Avg.Loss: {:.4f}, Avg.PSNR: {:.4f} dB'.format(epoch, iteration, len(test_data_loader)+1, mse.item(), psnr))
#
#
#     if epoch % 10 == 0 or epoch == opt.nEpochs:
#         # testDir = 'Test_Prediction/'
#         # os.makedirs(testDir, exist_ok=True)
#         # imgList = [ input[0], prediction[0], target[0] ]
#         # grid_img = make_grid(imgList)
#         # save_image(grid_img, testDir + 'epoch_{}.png'.format(epoch))
#
#     # avg_loss = total_loss / len(test_data_loader)
#     # avg_psnr = total_psnr / len(test_data_loader)
#     # print('Test Avg.Loss: {:.4f}, Avg.PSNR: {:.4f} dB'.format(avg_loss, avg_psnr))
#
#
#     # return avg_loss, avg_psnr

################################################################################

def checkpoint(epoch):
    checkpointDir = 'checkpoint/'
    os.makedirs(checkpointDir, exist_ok=True)
    net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
    net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
    torch.save(netG.state_dict(), net_g_model_out_path)
    torch.save(netD.state_dict(), net_d_model_out_path)
    cprint('---Checkpoint saved---\n', 'green')

def save_train_graph(logs, epoch):
    ## save csv ##
    graphPath = './graph/Train/'
    os.makedirs(graphPath, exist_ok=True)
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(graphPath + 'train_log_epoch_{}.csv'.format(epoch), index = False)

    ## save graph ##
    x = df_logs['epoch'].values
    y0 = df_logs['train_D_Loss'].values
    y1 = df_logs['train_G_Loss'].values
    fig, ax = plt.subplots(1,2, figsize=(20,8)) # 1x2 のグラフを生成
    ax[0].plot(x, y0, label='train_D_Loss', color = "red")
    ax[0].legend()
    ax[1].plot(x, y1, label='train_G_Loss', color = "blue")
    ax[1].legend()
    plt.savefig(graphPath + 'graph_epoch_{}.png'.format(epoch))
    cprint('---Training Graph saved---\n', 'green')

# def save_test_graph(logs, epoch):
#     ## save csv ##
#     graphPath = './graph/Test/'
#     os.makedirs(graphPath, exist_ok=True)
#     df_logs = pd.DataFrame(logs)
#     df_logs.to_csv(graphPath + 'test_log_epoch_{}.csv'.format(epoch), index = False)
#
#     ## save graph ##
#     x = df_logs['epoch'].values
#     y0 = df_logs['test_Loss'].values
#     y1 = df_logs['test_PSNR'].values
#     fig, ax = plt.subplots(1,2, figsize=(20,8)) # 1x2 のグラフを生成
#     ax[0].plot(x, y0, label='test_Loss', color = "red")
#     ax[0].legend()
#     ax[1].plot(x, y1, label='test_PSNR [dB]', color = "blue")
#     ax[1].legend()
#     plt.savefig(graphPath + 'graph_epoch_{}.png'.format(epoch))
#     cprint('---Testing Graph saved---\n', 'green')
#

################################################################################
if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch pix2pix')
    parser.add_argument('--batchSize', '-b', type=int, default=64, help='training batch size')
    # parser.add_argument('--testBatchSize', '-tb', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', '-e', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    # parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    # parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.00015')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    # parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=100, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    opt = parser.parse_args()

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    gpu_ids = []

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        gpu_ids = [0]

    ################################################################################
    print('==> Preparing Data Set')
    root_path = './dataset/'

    train_set = get_training_set(root_path)
    test_set = get_test_set(root_path)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                        batch_size=opt.batchSize, shuffle=True)

    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                                        batch_size=opt.testBatchSize, shuffle=False)

    print('==> Preparing Data Set: Complete')
    print('-'*50)
    print()

    ################################################################################
    print('==> Building Models')
    netG = define_G(gpu_ids=gpu_ids)
    netD = define_D(gpu_ids=gpu_ids)

    print('---------- Networks initialized -------------')
    print_network(netG)
    print('-'*50)
    print_network(netD)
    print('-----------------------------------------------\n')
    print('==> Building Models: Complete\n')
    
    ################################################################################
    # criterionGAN = GANLoss()
    # criterionL1 = nn.L1Loss()
    # criterionMSE = nn.MSELoss()
    # setup Loss
    criterionLoss = nn.CrossEntropyLoss()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 640, 256)
    real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 640, 256)

    if opt.cuda:
        netD = netD.cuda()
        netG = netG.cuda()
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)

    # Logging
    train_logs = []
    test_logs = []

    cprint('------ Start Training ------', 'yellow')
    for epoch in range(1, opt.nEpochs + 1):
        cprint('\n Epoch:{}'.format(epoch), 'cyan')
        train_D_loss, train_G_loss = train(epoch)
        test_loss, test_psnr = test(epoch)

        # Logging for epoch
        train_log = {'epoch':epoch, 'train_D_Loss':train_D_loss, 'train_G_Loss':train_G_loss}
        test_log = {'epoch':epoch, 'test_Loss':test_loss, 'test_PSNR':test_psnr}
        train_logs.append(train_log)
        test_logs.append(test_log)

        if epoch % 20 == 0 or epoch == opt.nEpochs:
            checkpoint(epoch)
            save_train_graph(train_logs, epoch)
            save_test_graph(test_logs, epoch)

    cprint('---Training Finished---\n', 'green')
