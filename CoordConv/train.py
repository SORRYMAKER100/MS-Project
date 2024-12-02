from dataset import *
import os
import torch
from utils import *
import torch.nn as nn
import skimage.io
import argparse
from torchvision import transforms
import tifffile
import pandas as pd
from pytorch_msssim import MS_SSIM
from math import log10, sqrt
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from tensorboardX import SummaryWriter

# setup parser
parser = argparse.ArgumentParser(description='Train the network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', choices=['train', 'debug'], dest='mode')
parser.add_argument('--train_continue', default='on',  dest='train_continue')
parser.add_argument('--computer', default='scc',choices=['local', 'scc'], dest='computer')
parser.add_argument("--num_gpu", type=int, default=[1], dest='num_gpu')
parser.add_argument('--num_epoch', type=int,  default=150, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
parser.add_argument('--train_ratio', type=float, default=0.9, dest='train_ratio')
parser.add_argument('--dir_chck', default='./traincc/1/checkpoints', dest='dir_chck')
parser.add_argument('--dir_save', default='./traincc/1/save', dest='dir_save')
parser.add_argument('--dir_log', default='./traincc/1/log', dest='dir_log')
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')
parser.add_argument("--early_stop", type=int, default=50, dest='early_stop', help='cancel=None')
parser.add_argument("--num_psf", type=int, default=9)
parser.add_argument("--network", default='cm2net', help='multiwiener svfourier and cm2net')
parser.add_argument("--ks", type=float, default=10.0)
parser.add_argument("--ps", type=int, default=1)

if __name__ == '__main__':
    PARSER = Parser(parser)
    args = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    torch.manual_seed(3407)
    torch.cuda.empty_cache()
    args.device = torch.device(0)

    if args.computer == 'local':
        # Change index (starts at 1)
        args.dir_data = 'T:/simulation beads/2d/debug/'
    elif args.computer == 'scc':
        args.dir_data = '/net/engnas/Research/eng_research_cisl/yqw/simulation_beads/2d/lsv_2d_beads_v17'
    # Create directories
    dir_result_val = args.dir_save + '/val/'
    dir_result_train = args.dir_save + '/train/'
    if not os.path.exists(os.path.join(dir_result_train)):
        os.makedirs(os.path.join(dir_result_train))
    if not os.path.exists(os.path.join(dir_result_val)):
        os.makedirs(os.path.join(dir_result_val))

    # Load training data
    if args.network == 'cm2net':
        # Create the complete dataset
        transform_train = transforms.Compose([ToTensorcm2()])
        whole_set = CM2Dataset(args.dir_data, transform=transform_train)
        length = len(whole_set)
        train_size, validate_size = int(args.train_ratio * length), length - int(args.train_ratio * length)
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
        train_set = Subset(train_set, isVal=False)
        validate_set = Subset(validate_set, isVal=True)
    else:
        transform_train = transforms.Compose([Noise(), Resize(), ToTensor()])
        whole_set = MyDataset(args.dir_data, transform=transform_train)
        length = len(whole_set)
        train_size, validate_size = int(args.train_ratio * length), length - int(args.train_ratio * length)
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
        print('Training images:', len(train_set),
              'Testing images:', len(validate_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(validate_set, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    num = len(args.num_gpu)
    num_batch_train = int((train_size * 81 / (args.batch_size * num)) + ((train_size * 81 % (args.batch_size * num)) != 0))
    num_batch_val = int((validate_size / args.batch_size) + ((validate_size % args.batch_size) != 0))

    # Setup the network TBD!
    if args.network == 'multiwiener':
        psfs = skimage.io.imread(args.dir_data + '/psf_v11.tif')
        psfs = np.array(psfs)
        psfs = psfs.astype('float32') / psfs.max()
        psfs = psfs[:, 57 * 2:3000, 94 * 2 + 156:4000 - 156]
        psfs = np.pad(psfs, ((0, 0), (657, 657), (350, 350)))
        Ks = args.ks * np.ones((args.num_psf, 1, 1))
        deconvolution = MultiWienerDeconvolution2D(psfs, Ks).to(args.device)
        enhancement = RCAN(args.num_psf).to(args.device)
        model = LSVEnsemble2d(deconvolution, enhancement)

    if args.network == 'svfourier':
        deconvolution = FourierDeconvolution2D_ds(args.num_psf, args.ps).to(args.device)
        enhancement = RCAN(args.num_psf).to(args.device)
        model = LSVEnsemble2d(deconvolution, enhancement)

    if args.network == 'cm2net':
        model = FPNet().to(args.device)

    # Use multiple GPUs
    model = model.to(args.device)

    # Setup loss & optimization
    ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=1)
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=1e-6)

    # Load from checkpoints
    st_epoch = 0

    # Logger
    losslogger = pd.DataFrame()
    if args.train_continue == 'on':
        checkpoint_path = os.path.join(args.dir_chck, 'best_model/model_epoch0010.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optim'])
            st_epoch = 10
            losslogger = checkpoint['losslogger']
            best_ssim = checkpoint.get('best_ssim', 0)
            print(f"Continuing training from epoch {st_epoch}")
        else:
            print(f"Checkpoint '{checkpoint_path}' not found, starting training from scratch.")

    # Save the best model
    best_ssim = 0
    trigger = 0
    best_loss = 10e7

    # Setup tensorboard
    dir_log = args.dir_log
    if not os.path.exists(os.path.join(dir_log)):
        os.makedirs(os.path.join(dir_log))
    writer = SummaryWriter(log_dir=dir_log)

    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        # Training phase
        model.train()
        loss_train = []
        ssim_train = []
        psnr_train = []
        for batch, data in enumerate(train_loader, 1):

            # Ground truth shape [Batch, H, W], Output [Batch, 1, H, W]
            if args.network == 'cm2net':
                gt = data['gt'].to(args.device)
                demix = data['meas'].unsqueeze(1).to(args.device)
                index_list = data['index'].to(args.device)
                optimizer.zero_grad()
                output = model(demix, index_list)
                loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                loss = loss_recon
            else:
                meas = data['meas'].to(args.device)
                gt = data['gt'].to(args.device)
                optimizer.zero_grad()
                output = model(meas)
                loss = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            output_n = (output - output.view(output.size(0), -1).min(1)[0].view(-1, 1, 1, 1)) / \
                       (output.view(output.size(0), -1).max(1)[0].view(-1, 1, 1, 1) - output.view(output.size(0), -1).min(1)[0].view(-1, 1, 1, 1) + 1e-8)
            gt_n = (gt.unsqueeze(1) - gt.view(gt.size(0), -1).min(1)[0].view(-1, 1, 1, 1)) / \
                   (gt.view(gt.size(0), -1).max(1)[0].view(-1, 1, 1, 1) - gt.view(gt.size(0), -1).min(1)[0].view(-1, 1, 1, 1) + 1e-8)

            ssim = ssim_loss(output_n, gt_n)
            psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))
            # Get losses
            loss_train += [loss.item()]
            ssim_train += [ssim.item()]
            psnr_train += [psnr.item()]

            if args.local_rank == 0:
                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_train), np.mean(ssim_train)))

        scheduler.step()

        if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
            gt = gt.data.cpu().numpy()
            x_recon = torch.squeeze(output, 1).data.cpu().numpy()
            for j in range(gt.shape[0]):
                im_gt = (np.clip(gt[j, ...] / np.max(gt[j, ...]), 0, 1) * 255).astype(np.uint8)
                im_recon = (np.clip(x_recon[j, ...] / np.max(x_recon[j, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_train + str(epoch) + '_recon' + '.tif'), im_recon.squeeze())
                tifffile.imwrite((dir_result_train + str(epoch) + '_gt' + '.tif'), im_gt.squeeze())

        # Validation phase
        print('Validation')
        with torch.no_grad():
            model.eval()
            loss_val = []
            ssim_val = []
            psnr_val = []

            for batch, data in enumerate(val_loader, 1):
                # Forward simulation (add noise)
                if args.network == 'cm2net':
                    gt = data['gt'].to(args.device)
                    demix = data['meas'].unsqueeze(1).to(args.device)
                    index_list = data['index'].to(args.device)
                    output = model(demix, index_list)
                    loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                    loss = loss_recon
                else:
                    meas = data['meas'].to(args.device)
                    gt = data['gt'].to(args.device)
                    output = model(meas)
                    loss = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                output_n = (output - output.view(output.size(0), -1).min(1)[0].view(-1, 1, 1, 1)) / \
                           (output.view(output.size(0), -1).max(1)[0].view(-1, 1, 1, 1) - output.view(output.size(0), -1).min(1)[0].view(-1, 1, 1, 1) + 1e-8)
                gt_n = (gt.unsqueeze(1) - gt.view(gt.size(0), -1).min(1)[0].view(-1, 1, 1, 1)) / \
                       (gt.view(gt.size(0), -1).max(1)[0].view(-1, 1, 1, 1) - gt.view(gt.size(0), -1).min(1)[0].view(-1, 1, 1, 1) + 1e-8)

                ssim = ssim_loss(output_n, gt_n)
                psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))
                # Get losses
                loss_val += [loss.item()]
                ssim_val += [ssim.item()]
                psnr_val += [psnr.item()]

                if args.local_rank == 0:
                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_val), np.mean(ssim_val)))

            if epoch == 1:
                gt = gt.data.cpu().numpy()
                im_gt = (np.clip(gt[-1, ...] / np.max(gt[-1, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_val + str(epoch) + '_gt' + '.tif'), im_gt.squeeze())

            if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
                x_recon = output.data.cpu().numpy()
                im_recon = (np.clip(x_recon[-1, ...] / np.max(x_recon[-1, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_val + str(epoch) + '_recon' + '.tif'), im_recon.squeeze())

                if args.network == 'svfourier':
                    psfs_re = model.deconvolution.psfs_re.detach().cpu().numpy()
                    psfs_im = model.deconvolution.psfs_im.detach().cpu().numpy()
                    psf_freq = psfs_re + psfs_im * 1j
                    psf = np.fft.ifftshift(np.fft.irfft2(psf_freq, axes=(-2, -1)))
                    psf_mip = np.max(psf, 0).squeeze()
                    psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
                    tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

                if args.network == 'multiwiener':
                    psf = model.deconvolution.psfs.detach().cpu().numpy()
                    psf_mip = np.max(psf, 0).squeeze()
                    psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
                    tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

        if args.local_rank == 0:
            # Log the results
            df = pd.DataFrame()
            df['loss_train'] = pd.Series(np.mean(loss_train))
            df['ssim_train'] = pd.Series(np.mean(ssim_train))
            df['psnr_train'] = pd.Series(np.mean(psnr_train))
            df['loss_val'] = pd.Series(np.mean(loss_val))
            df['ssim_val'] = pd.Series(np.mean(ssim_val))
            df['psnr_val'] = pd.Series(np.mean(psnr_val))
            losslogger = losslogger.append(df)
            writer.add_scalar('Loss/loss_train', np.mean(loss_train), epoch)
            writer.add_scalar('SSIM/ssim_train', np.mean(ssim_train), epoch)
            writer.add_scalar('PSNR/psnr_train', np.mean(psnr_train), epoch)
            writer.add_scalar('Loss/loss_val', np.mean(loss_val), epoch)
            writer.add_scalar('SSIM/ssim_val', np.mean(ssim_val), epoch)
            writer.add_scalar('PSNR/psnr_val', np.mean(psnr_val), epoch)

        trigger += 1
        if args.local_rank == 0 and (np.mean(ssim_val) > best_ssim):
            save(args.dir_chck + '/best_model/', model, optimizer, epoch, losslogger)
            best_ssim = np.mean(ssim_val)
            print("=> Saved best model")
            trigger = 0

        if not args.early_stop is not None and args.local_rank == 0:
            if trigger >= args.early_stop:
                print("=> Early stopping")
            break

        # Save checkpoint
        if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
            save(args.dir_chck, model, optimizer, epoch, losslogger)
