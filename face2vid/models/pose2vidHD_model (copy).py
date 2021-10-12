### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
# from . import networks
from . import networks_modified as networks


# TODO: modify upon pix2pixHDModel, adding last-frame reference
#   TODO: generator, input two images now
#   TODO: discriminator, using conv2D but adding PWC-net for optical flow alignment as well
#   TODO: loss, using both feature maching and VGG, also implement PWC-loss in the future.

class Pose2VidHDModel(BaseModel):
    def name(self):
        return 'Pose2VidHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_flow_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_flow_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_flow, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,g_flow,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain                                             # isTrain = True
        self.use_features = opt.instance_feat or opt.label_feat                # use_features = false 
        self.gen_features = self.use_features and not self.opt.load_features   # gen_features = false
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc     # input_nc =3 and lable_nc = 0

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc                                           # netG_input_nc = 3
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                                  # netG_input_nc = 3 and feat_num =3

        # TODO: 20180929: Generator Input contains two images...
        netG_input_nc += opt.output_nc  # also contains the previous frame   netG_input_nc = 6
        netG_input_nc = 27
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)       

# (input_nc = 27, output_nc = 3, ngf = 32, netG = local, n_downsample_global = 4, n_blocks_global = 9, n_local_enhancers = 1, n_blocks_local =3, instance, 0) 

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan          # false
            netD_input_nc = input_nc + opt.output_nc           # 6
            if not opt.no_instance:
                netD_input_nc += 1

            # TODO: 20180929: Generator Input contains two images...
            netD_input_nc *= 2  # two pairs of pose/frame      # 12
            netD_input_nc = 60
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            # (input=60,ndf=32,n_layers_D=3,instance,false,num_D=3, True, 0)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.old_lr = opt.lr                                                            #lr=0.0001

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_flow_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            if not opt.no_flow_loss:
                # 20181013 Flow L1 needs averaging
                self.nelem = 288*512 if opt.dataroot.find('512') != -1 else 576*1024
                # print(self.nelem / 512) 
                self.criterionFlow = networks.FlowLoss(self.gpu_ids)
                
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_flow', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map

    # TODO: 20180930: Ignore features for now.
    def forward(self, label, real_image):

        label = label.data.cuda()
        real_image = real_image.data.cuda()
        # label = label.data
        # real_image = real_image.data
        gt1 = real_image[:, 5, ...]
        gt2 = real_image[:, 6, ...]

        ### Generator Forward, predict two frames
        x1 = torch.cat((label[:,0, ...], label[:,1, ...], label[:,2, ...], label[:,3, ...], label[:,4, ...], label[:,5, ...], label[:,6, ...], label[:,7, ...], label[:,8, ...]), dim=1)
        x2 = torch.cat((label[:,1, ...], label[:,2, ...], label[:,3, ...], label[:,4, ...], label[:,5, ...], label[:,6, ...], label[:,7, ...], label[:,8, ...], label[:,9, ...]), dim=1)

        y1 = self.netG.forward(x1)
        y2 = self.netG.forward(x2)

        # Fake Detection and Loss
        pred_fake_pool = self.netD.forward(torch.cat((x1, x2, y1.detach(), y2.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.netD.forward(torch.cat((x1, x2, gt1.detach(), gt2.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Possibility Loss)
        pred_fake = self.netD.forward(torch.cat((x1, x2, y1, y2), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = (self.criterionVGG(y1, gt1) + self.criterionVGG(y2, gt2))\
                            * self.opt.lambda_feat

        # 20181012: pwc-flow matching loss
        loss_G_flow = 0
        if not self.opt.no_flow_loss:
            # loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow / self.nelem
            loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow

        # 20180930: Always return fake_B now, let super function decide whether to save it
        y1_clean = torch.squeeze(y1.detach())
        y2_clean = torch.squeeze(y2.detach())
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_flow, loss_D_real, loss_D_fake),
                torch.cat((y1_clean, y2_clean), dim=2)]
    # def forward(self, label, real_image):

    #     label = label.data.cuda()
    #     real_image = real_image.data.cuda()
    #     # label = label.data
    #     # real_image = real_image.data
    #     gt1 = real_image[:, 0, ...]
    #     gt2 = real_image[:, 1, ...]

    #     ### Generator Forward, predict two frames
    #     x1 = label[:, 0, ...]
    #     x2 = label[:, 1, ...]
    #     zero = torch.zeros_like(gt1)
    #     # print(x1.shape)
    #     y1 = self.netG.forward(torch.cat((x1, zero), 1))
    #     y2 = self.netG.forward(torch.cat((x2, y1), 1))

    #     # Fake Detection and Loss
    #     pred_fake_pool = self.netD.forward(torch.cat((x1, x2, y1.detach(), y2.detach()), dim=1))
    #     loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

    #     # Real Detection and Loss        
    #     pred_real = self.netD.forward(torch.cat((x1, x2, gt1.detach(), gt2.detach()), dim=1))
    #     loss_D_real = self.criterionGAN(pred_real, True)

    #     # GAN loss (Fake Possibility Loss)
    #     pred_fake = self.netD.forward(torch.cat((x1, x2, y1, y2), dim=1))
    #     loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
    #     # GAN feature matching loss
    #     loss_G_GAN_Feat = 0
    #     if not self.opt.no_ganFeat_loss:
    #         feat_weights = 4.0 / (self.opt.n_layers_D + 1)
    #         D_weights = 1.0 / self.opt.num_D
    #         for i in range(self.opt.num_D):
    #             for j in range(len(pred_fake[i])-1):
    #                 loss_G_GAN_Feat += D_weights * feat_weights * \
    #                     self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
    #     # VGG feature matching loss
    #     loss_G_VGG = 0
    #     if not self.opt.no_vgg_loss:
    #         loss_G_VGG = (self.criterionVGG(y1, gt1) + self.criterionVGG(y2, gt2))\
    #                      * self.opt.lambda_feat

    #     # 20181012: pwc-flow matching loss
    #     loss_G_flow = 0
    #     if not self.opt.no_flow_loss:
    #         # loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow / self.nelem
    #         loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow

    #     # 20180930: Always return fake_B now, let super function decide whether to save it
    #     y1_clean = torch.squeeze(y1.detach())
    #     y2_clean = torch.squeeze(y2.detach())
    #     return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_flow, loss_D_real, loss_D_fake),
    #             torch.cat((y1_clean, y2_clean), dim=2)]

    # TODO: Implement a full training cycle in here, necessary for some carefully maneuver
    # 20180930: This is a tricky one, be careful!
    def train_one_step(self, data, save_fake=False):
        losses, generated = self.forward(Variable(data['label']), Variable(data['image']))

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(self.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) \
                 + loss_dict.get('G_VGG', 0) + loss_dict.get('G_flow', 0)

        ############### Update Now ###################
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return loss_dict, (generated if save_fake else None)

    def inference(self, label, inst=None, prev_frame=None):
        # Encode Inputs        
        input_label, _, _, _ = self.encode_input(Variable(label), infer=True)
        # prev_frame = Variable(prev_frame.data.cuda())

        # Fake Generation
        # if self.use_features:       
        #     # sample clusters from precomputed features             
        #     feat_map = self.sample_features(inst_map)
        #     input_concat = torch.cat((input_label, feat_map), dim=1)                        
        # else:
        #     input_concat = input_label

        # input_concat = torch.cat((input_concat, prev_frame), dim=1)
        # print(input_label.shape)
        input_concat = torch.cat((input_label[:,0, ...], input_label[:,1, ...], input_label[:,2, ...], input_label[:,3, ...], input_label[:,4, ...], input_label[:,5, ...], input_label[:,6, ...], input_label[:,7, ...], input_label[:,8, ...]), dim=1)
        
        # print(input_concat.shape)
        
        # TODO：Write a for loop that iteratively generates each frame
        # TODO: Also provide a option as whether to provide the first ground-truth frame
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pose2VidHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
