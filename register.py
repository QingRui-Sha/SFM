import os,sys
import random
import argparse
import time
import numpy as np
import torch
from torchvision import transforms
import nibabel as nib
import torch.utils.data as Data
from scipy.ndimage import zoom
from torch.utils.tensorboard import SummaryWriter
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', default='',help='line-seperated list of training files')
parser.add_argument('--img-prefix', default='',help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='',
                    help='model and results output directory (default: models)')
parser.add_argument('--data-inshape', type=int, nargs='+',
                    help='list of unet encoder filters (default: 160 160 32)')
parser.add_argument('--test-dir', default=r'test0',help='test-results')
parser.add_argument('--load-model', default='')
parser.add_argument('--atlas-path', default='', help='get affine and header')
parser.add_argument('--test_nums', default=2)
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--device', default='cuda')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')



# loss hyperparameters
# parser.add_argument('--image-loss', default='ncc',
#                     help='image reconstruction loss - can be mse or ncc (default: mse)')
# parser.add_argument('--weight-smooth', type=float,  default=1)
# parser.add_argument('--weight-image-sim', type=float, default=1)
# parser.add_argument('--weight-sfim1', type=float, default=1)
args = parser.parse_args()
writer = SummaryWriter(log_dir=os.path.join(
        args.model_dir, 'test_tensorboard'))


# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

assert len(train_files) > 0, 'Could not find any training data.'




train_set = vxm.datasets.Dataset_demo(train_files)
train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)#,  drop_last=True,pin_memory=args.pin_memory,num_workers=args.num_works)#,,pin_memory=True)

# prefetcher = vxm.datasets.data_prefetcher(train_loader)
fetcher=iter(train_loader)
inputs=next(fetcher)
# inputs = prefetcher.next()



# extract shape from sampled input
inshape = args.data_inshape

# prepare model folder
model_dir = args.model_dir


# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense_pad.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense_pad(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.eval()

os.makedirs(os.path.join(args.model_dir,args.test_dir), exist_ok=True)


# prepare image loss
image_loss_func = vxm.losses.NCC().loss
# if args.image_loss == 'ncc':
#    
# elif args.image_loss == 'mse':
#     image_loss_func = vxm.losses.MSE().loss
# else:
#     raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# # prepare deformation loss
# smooth_loss_func=vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
PSTN = vxm.SFIM.ProjectiveSpatialTransformer(inshape)
sfim1_loss_func=PSTN.loss_spim1

# training loops

epoch_loss = []
epoch_total_loss = []
epoch_step_time = []

for step in range(args.test_nums):

    step_start_time = time.time()
    if inputs==None:
        # prefetcher = vxm.datasets.data_prefetcher(train_loader)
        # inputs = prefetcher.next()
        fetcher=iter(train_loader)
        inputs=next(fetcher)
    moving_image=inputs[0].cuda()
    fixed_image=inputs[1].cuda()
    # print(moving_image.shape,fixed_image.shape)
    # run inputs through the model to produce a warped image and flow field
    warped_image,small_v,flow = model(moving_image,fixed_image)
    # calculate total loss
    loss = 0
    loss_list = []
    
    image_sim_loss = image_loss_func(fixed_image, warped_image)# * args.weight_image_sim
    # smooth_loss = smooth_loss_func(small_v, small_v) * args.weight_smooth
    sfim1_loss=sfim1_loss_func(flow)#*args.weight_sfim1
    # loss=image_sim_loss+smooth_loss+sfim1_loss
    
    loss_list.append(image_sim_loss.item())
    # loss_list.append(smooth_loss.item())
    loss_list.append(sfim1_loss.item())
    epoch_loss.append(loss_list)#size:step per epoch *number of loss_func
    # epoch_total_loss.append(loss.item())#size:step per epoch *1

    
    # inputs = prefetcher.next()
    try:
        inputs=next(fetcher)
    except StopIteration:
        inputs = None
    # get compute time
    epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (step + 1, args.test_nums)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in epoch_loss[step]])
    loss_info = 'loss:  (%s)' % ( losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    for i, f in enumerate(epoch_loss[step]):
        writer.add_scalar('losses_info_test_' + str(i), f, step+1)
    # writer.add_scalar('total_loss_info_test_', epoch_total_loss[step],step)
    nii_data = nib.load(args.atlas_path)
    affine=nii_data.affine
    header=nii_data.header
    

    nib.save(nib.Nifti1Image(moving_image.detach().cpu().numpy().squeeze(), affine,header), os.path.join(args.model_dir,args.test_dir,str(step).zfill(5)+'moving_image.nii.gz'))
    nib.save(nib.Nifti1Image(fixed_image.detach().cpu().numpy().squeeze(), affine,header), os.path.join(args.model_dir,args.test_dir,str(step).zfill(5)+'fixed_image.nii.gz'))
    nib.save(nib.Nifti1Image(flow.permute(0,2,3,4,1).detach().cpu().numpy().squeeze(), affine,header), os.path.join(args.model_dir,args.test_dir,str(step).zfill(5)+'flow.nii.gz'))
    nib.save(nib.Nifti1Image(warped_image.detach().cpu().numpy().squeeze(), affine,header), os.path.join(args.model_dir,args.test_dir,str(step).zfill(5)+'warped_image.nii.gz'))
    ones=torch.ones(inshape).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        ones=ones.cuda()
    sfm=PSTN(ones,flow)
    nib.save(nib.Nifti1Image(sfm.permute(0,1,4,3,2).detach().cpu().numpy().squeeze(), affine,header), os.path.join(args.model_dir,args.test_dir,str(step).zfill(5)+'sfm.nii.gz'))
            
# print epoch info
losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
print(' -results- ',loss_info, flush=True)

for i, f in enumerate(np.mean(epoch_loss, axis=0)):
    writer.add_scalar('losses_avg_test_' + str(i), f, 0)