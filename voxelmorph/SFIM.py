import numpy as np
import torch
from scipy.special import comb
from torch import nn
import torch.nn.functional as nnf



class ProjectiveSpatialTransformer(nn.Module):
    def __init__(self,size,edge=1):
        super(ProjectiveSpatialTransformer, self).__init__()
        
        self.size=size

        self.edge=edge
        #To eliminate rounding errors at the boundaries, exclude the boundary regions from calculations. 
        #If your data contains non-background elements at the boundaries, you can set them to zero.
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
        

    def add_view(self,sam,int_locs,value): 
       
        B,C,H,W,D=int_locs.shape
        assert B==1
        # sam_flat = sam.contiguous().view(-1)
        # value_flat=value.contiguous().view(-1)
        sam_flat = sam.view(-1)
        value_flat=value.view(-1)
        #int_locs B 3 H W D
        Shape=(H,W,D)
        for i in range(len(Shape)):
            slice=int_locs[:,i,...]
            slice[int_locs[:,i,...]>=(Shape[i]-1)]=Shape[i]-1
            
        int_locs = int_locs.view(3,-1)
        int_locs_flat=int_locs[0]*W*D+int_locs[1]*D+int_locs[2]
        sam_flat.scatter_add_(0, int_locs_flat, value_flat)
        sam_flat = sam_flat.view(sam.shape)
        return sam_flat
    def forward(self, src, flow, mode='bilinear'):
        shape = flow.shape[2:]

        # To test whether three components correspond to dx, dy, and dz, 
        # flow[:,0,...]=2#(shape[0]-1)
        # flow[:,1,...]=-7#(shape[1]-1)
        # flow[:,2,...]=0#(shape[2]-1)

        if torch.cuda.is_available():
            self.grid = self.grid.cuda()
        locs=self.grid + flow
        
        locs[locs<=0]=0 
        int_locs=torch.floor(locs).type(torch.int64)
        delta_locs=locs-int_locs
        
        new_src=torch.zeros_like(src)
  
        #000
        new_src= self.add_view(new_src,int_locs, src*(1-delta_locs[:,0:1,...])*(1-delta_locs[:,1:2,...])*(1-delta_locs[:,2:3,...]))
        #100
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...]+1,int_locs[:, 1:2, ...],int_locs[:, 2:3, ...]),dim=1), 
                          src*(delta_locs[:,0:1,...])*(1-delta_locs[:,1:2,...])*(1-delta_locs[:,2:3,...]))       
        #010 
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...],int_locs[:, 1:2, ...]+1,int_locs[:, 2:3, ...]),dim=1), 
                          src*(1-delta_locs[:,0:1,...])*(delta_locs[:,1:2,...])*(1-delta_locs[:,2:3,...]))
        #001
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...],int_locs[:, 1:2, ...],int_locs[:, 2:3, ...]+1),dim=1), 
                          src*(1-delta_locs[:,0:1,...])*(1-delta_locs[:,1:2,...])*(delta_locs[:,2:3,...]))
        #110
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...]+1,int_locs[:, 1:2, ...]+1,int_locs[:, 2:3, ...]),dim=1), 
                          src*(delta_locs[:,0:1,...])*(delta_locs[:,1:2,...])*(1-delta_locs[:,2:3,...]))
        #101
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...]+1,int_locs[:, 1:2, ...],int_locs[:, 2:3, ...]+1),dim=1), 
                          src*(delta_locs[:,0:1,...])*(1-delta_locs[:,1:2,...])*(delta_locs[:,2:3,...]))
        #011
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...],int_locs[:, 1:2, ...]+1,int_locs[:, 2:3, ...]+1),dim=1), 
                          src*(1-delta_locs[:,0:1,...])*(delta_locs[:,1:2,...])*(delta_locs[:,2:3,...]))
        #111
        new_src= self.add_view(new_src,torch.cat((int_locs[:, 0:1,...]+1,int_locs[:, 1:2, ...]+1,int_locs[:, 2:3, ...]+1),dim=1), 
                          src*(delta_locs[:,0:1,...])*(delta_locs[:,1:2,...])*(delta_locs[:,2:3,...]))
        
        # The following implementation is non-differentiable, and I was troubled by this bug for a long time. 
        # If you need to modify the program for a batch size greater than 1, 
        # I hope you can avoid this implementation approach in advance.
        # new_src[:, 
        #         :,
        #         int_locs[0, 0, :, :, :], 
        #         int_locs[0, 1, :, :, :],
        #         int_locs[0, 2, :, :, :]] = src*(1-delta_locs[:,0:1,...])*(1-delta_locs[:,1:2,...])*(1-delta_locs[:,2:3,...])

        return new_src
    
    def gradient_loss(self,s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 

        if(penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0
    
    def loss_spim1(self,flow):
        ones=torch.ones(self.size).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            ones=ones.cuda()
        return torch.mean((self.forward(ones,flow)[:,:,self.edge:self.size[0]-self.edge,
                                                   self.edge:self.size[1]-self.edge,
                                                   self.edge:self.size[2]-self.edge]-
                                         ones[:,:,self.edge:self.size[0]-self.edge,
                                                   self.edge:self.size[1]-self.edge,
                                                   self.edge:self.size[2]-self.edge])**2)
    def loss_focal_spim1(self,flow,focal_weight):
        ones=torch.ones(self.size).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            ones=ones.cuda()
        
        return torch.sum(((self.forward(ones,flow)-ones)**2)*focal_weight)/torch.sum(focal_weight>0.15)
    
    def loss_spim2(self,flow):
        ones=torch.ones(self.size).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            ones=ones.cuda()
            
        return self.gradient_loss(self.forward(ones,flow)[:,:,self.edge:self.size[0]-self.edge,
                                                   self.edge:self.size[1]-self.edge,
                                                   self.edge:self.size[2]-self.edge])
    #def loss_k(self,flow):
    #def loss_k100(self,flow):
    #def loss_log(self,flow):

    def extreme_sampling_probability(self,flow,threshold_l=0.3,threshold_r=1.7):
        ones=torch.ones(self.size).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            ones=ones.cuda()
            print(ones.requires_grad,'....')
        SP=self.forward(ones,flow)[:,:,self.edge:self.size[0]-self.edge,
                                                   self.edge:self.size[1]-self.edge,
                                                   self.edge:self.size[2]-self.edge]
        return (SP<threshold_l).sum()+(SP>threshold_r).sum()


