import  torch
from    torch import optim, nn, autograd
from    torch.utils.data import DataLoader
from    model import TfaceGAN, NLayerDiscriminator
from    dataset102 import Facial_Dataset
import argparse
import os, glob
import random, csv
import numpy as np

parser = argparse.ArgumentParser(description='Train_setting')
parser.add_argument('--audiopath', type=str, default='/content/FACIAL/examples/audio_preprocessed/train1.pkl')
parser.add_argument('--npzpath', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz')
parser.add_argument('--cvspath', type=str, default = '/content/FACIAL/video_preprocess/train1_openface/train1_512_audio.csv')
parser.add_argument('--pretainpath_gen', type=str, default = '/content/FACIAL/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl')
parser.add_argument('--savepath', type=str, default = './checkpoint/train1')
opt = parser.parse_args()

if not os.path.exists(opt.savepath):
    os.mkdir(opt.savepath)

audio_paths = []
audio_paths.append(opt.audiopath)
npz_paths = []
npz_paths.append(opt.npzpath)
cvs_paths = []
cvs_paths.append(opt.cvspath)

batchsz = 16
epochs = 11

device = torch.device('cuda')
torch.manual_seed(1234)

training_set = Facial_Dataset(audio_paths,npz_paths,cvs_paths)


train_loader = DataLoader(training_set,
                            batch_size=batchsz,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def main():

    lr = 1e-4
    modelgen = TfaceGAN().to(device)

    modeldis = NLayerDiscriminator().to(device)

    modelgen.load_state_dict(torch.load(opt.pretainpath_gen))
    print(modelgen)
    print(modeldis)

    optimG = optim.Adam(modelgen.parameters(), lr=lr*0.1)

    optimD = optim.Adam(modeldis.parameters(), lr=lr*0.1)

    
    criteon1 = nn.L1Loss()
    criteon = nn.MSELoss()

    for epoch in range(0, epochs):

        if epoch % 5 == 0:
            torch.save(modelgen.state_dict(), opt.savepath+'/Gen-'+str(epoch)+'.mdl')
            torch.save(modeldis.state_dict(), opt.savepath+'/Dis-'+str(epoch)+'.mdl')


        for step, (x,y) in enumerate(train_loader):
            modelgen.train()
            #x(64, 128, 29) y(64, 128, 70)
            x, y = x.to(device), y.to(device)
            motiony = y[:,1:,:]-y[:,:-1,:]

            # #dis
            set_requires_grad(modeldis, True)

            predr = modeldis(torch.cat([y, motiony], 1))
            lossr = criteon(torch.ones_like(predr),predr)

            yf = modelgen(x,y[:,:1,:])
            motionlogits = yf[:,1:,:]-yf[:,:-1,:]
            
            predf = modeldis(torch.cat([yf, motionlogits], 1).detach())
            lossf = criteon(torch.zeros_like(predf),predf)

            lossD = lossr + lossf
            optimD.zero_grad()
            lossD.backward()
            optimD.step()
            

            # generator
            set_requires_grad(modeldis, False)
            loss_s = 10*(criteon1(yf[:,:1,:6], y[:,:1,:6])+criteon1(yf[:,:1,6], y[:,:1,6])+criteon1(yf[:,:1,6:], y[:,:1,6:]))
            lossg_e = 20*criteon(yf[:,:,7:], y[:,:,7:])
            lossg_em = 200*criteon(motionlogits[:,:,7:], motiony[:,:,7:])
            
            loss_au = 0.5*criteon(yf[:,:,6],y[:,:,6])
            loss_aum = 1*criteon(motionlogits[:,:,6], motiony[:,:,6])
            loss_pose = 1*criteon(yf[:,:,:6],y[:,:,:6])
            loss_posem = 10*criteon(motionlogits[:,:,:6], motiony[:,:,:6])
            predf2 = modeldis(torch.cat([yf, motionlogits], 1))
  
            lossg_gan = criteon(torch.ones_like(predf2),predf2)

            lossG =  loss_s + lossg_e +  lossg_em + loss_au + loss_aum + loss_pose + loss_posem + 0.1*lossg_gan
            # lossG =   loss_pose_1 + loss_pose_2
            optimG.zero_grad()
            lossG.backward()
            optimG.step()


            if step % 60 == 0:
                print('epoch: ',epoch,' loss_s: ',loss_s.item(),' lossg_e: ',lossg_e.item(), ' lossg_em: ',lossg_em.item())
                print(' loss_au: ',loss_au.item(),' loss_aum: ',loss_aum.item()) 
                print(' loss_pose: ',loss_pose.item(),' loss_posem: ',loss_posem.item()) 
if __name__ == '__main__':
    main()

