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
import numpy as np

parser = argparse.ArgumentParser(description='Train the network for multi-view FPNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', choices=['train', 'debug'], dest='mode')
parser.add_argument('--train_continue', default='on',  dest='train_continue')
parser.add_argument('--computer', default='scc',choices=['local', 'scc'], dest='computer')
parser.add_argument("--num_gpu", type=int, default=[1], dest='num_gpu')
parser.add_argument('--num_epoch', type=int,  default=150, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=3, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
parser.add_argument('--train_ratio', type=float, default=0.9, dest='train_ratio')
parser.add_argument('--dir_chck', default='./train00ds/1/checkpoints', dest='dir_chck')
parser.add_argument('--dir_save', default='./train00ds/1/save', dest='dir_save')
parser.add_argument('--dir_log', default='./train00ds/1/log', dest='dir_log')
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')
parser.add_argument("--early_stop", type=int, default=100, dest='early_stop', help='cancel=None')
parser.add_argument("--num_psf", type=int, default=9)
parser.add_argument("--network", default='cm2net', help='multiwiener svfourier and cm2net')
parser.add_argument("--ks", type=float, default=10.0)
parser.add_argument("--ps", type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.85, help='Weight for MSE loss', dest='alpha')
parser.add_argument('--beta', type=float, default=0.15, help='Weight for SSIM loss', dest='beta')

if __name__ == '__main__':
    PARSER = Parser(parser)
    args = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    torch.manual_seed(3407)
    torch.cuda.empty_cache()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lens_centers = [
        (664, 1192), (664, 2089), (660, 2982),
        (1564, 1200), (1557, 2094), (1548, 2988),
        (2460, 1206), (2452, 2102), (2444, 2996)
    ]

    if args.computer == 'local':
        args.dir_data = 'T:/simulation beads/2d/debug/'
    elif args.computer == 'scc':
        args.dir_data = '/net/engnas/Research/eng_research_cisl/yqw/simulation_beads/2d/lsv_2d_beads_v17'
    else:
        raise ValueError("Unsupported computer environment")
      
    dir_result_val = os.path.join(args.dir_save, 'val')
    dir_result_train = os.path.join(args.dir_save, 'train')
    os.makedirs(dir_result_train, exist_ok=True)
    os.makedirs(dir_result_val, exist_ok=True)
    os.makedirs(args.dir_chck, exist_ok=True)

    transform_train = transforms.Compose([Noisecm2(), Crop(lens_centers=lens_centers), ToTensorcm2()])
    transform_val = transforms.Compose([Crop(lens_centers=lens_centers), ToTensorcm2()])

    whole_set = CM2Dataset(args.dir_data,transform=transform_val, lens_centers=lens_centers)
    length = len(whole_set)
    train_size = int(args.train_ratio * length)
    validate_size = length - train_size
    train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
    train_set = Subset(train_set, isVal=False, patch_size=480, stride=240)
    validate_set = Subset(validate_set, isVal=True)

    train_set.dataset.transform = transform_train
    validate_set.dataset.transform = transform_val

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(validate_set, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    model = FPNet(L=4).to(args.device)

    if torch.cuda.device_count() > 1 and len(args.num_gpu) > 1:
        print(f"Use {torch.cuda.device_count()} GPU")
        model = nn.DataParallel(model)

    ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=1)
    ssim_loss9 = MS_SSIM(data_range=1, size_average=True, channel=9)
    l2_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    st_epoch = 0
    losslogger = pd.DataFrame()
    best_ssim = 0
    trigger = 0
    best_loss = 1e7
    if args.train_continue == 'on':
        checkpoint_path = os.path.join(args.dir_chck, 'best_model/model_epoch0017.pth')
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
            st_epoch = 17
            losslogger = checkpoint['losslogger']
            best_ssim = checkpoint.get('best_ssim', 0)
            print(f"Continue training from epoch {st_epoch}")
        else:
            print(f"Check point '{checkpoint_path}' not found, start from scratch")

    writer = SummaryWriter(log_dir=args.dir_log)

    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        model.train()
        loss_train = []
        ssim_train = []
        psnr_train = []
        for batch, data in enumerate(train_loader, 1):
            gt = data['gt'].to(args.device)  # [Batch, 1, H, W]
            meas = data['meas'].to(args.device)  # [Batch, 9, H, W]
            demix_gt = data['demix'].to(args.device)  # [Batch, 9, H, W]
            index_list = data['index'].to(args.device)  # [Batch, 9, H, W, 2]

            optimizer.zero_grad()

            demix_output, recon_output = model(meas, index_list)  # [Batch, 1, H, W], [Batch, 1, H, W]

            demix_ssim = 1 - ssim_loss9(demix_output,demix_gt)
            demix_mse = l2_loss(demix_output, demix_gt)
            demix_loss = args.alpha * demix_mse + args.beta * demix_ssim

            mse_loss_value = l2_loss(recon_output, gt)
            ssim_loss_value = 1 - ssim_loss(recon_output, gt)
            recon_loss = args.alpha * mse_loss_value + args.beta * ssim_loss_value

            loss = demix_loss + recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            output_n = recon_output
            gt_n = gt
            with torch.no_grad():
                ssim_index = ssim_loss(output_n, gt_n)
                mse = l2_loss(recon_output, gt).item()
                psnr = 20 * log10(1.0 / sqrt(mse)) if mse != 0 else 100

            loss_train.append(loss.item())
            ssim_train.append(ssim_index.item())
            psnr_train.append(psnr)

            if args.local_rank == 0:
                print(f'Training: Epoch {epoch}: batch {batch}/{len(train_loader)}: loss: {np.mean(loss_train):.4f} SSIM: {np.mean(ssim_train):.4f} PSNR: {np.mean(psnr_train):.2f} dB')

        scheduler.step()

        if args.local_rank == 0:
            writer.add_scalar('Loss/train', np.mean(loss_train), epoch)
            writer.add_scalar('SSIM/train', np.mean(ssim_train), epoch)
            writer.add_scalar('PSNR/train', np.mean(psnr_train), epoch)

        print('Validation')
        with torch.no_grad():
            model.eval()
            loss_val = []
            ssim_val = []
            psnr_val = []

            for batch, data in enumerate(val_loader, 1):
                gt = data['gt'].to(args.device)  # [1, 1, H, W]
                meas = data['meas'].to(args.device)  # [1, 9, H, W]
                demix_gt = data['demix'].to(args.device)  # [1, 9, H, W]
                index_list = data['index'].to(args.device)  # [1, 9, H, W, 2]

                demix_output, recon_output = model(meas, index_list)  # [1, 1, H, W], [1, 1, H, W]

                demix_ssim = 1 - ssim_loss9(demix_output,demix_gt)
                demix_mse = l2_loss(demix_output, demix_gt)
                demix_loss = args.alpha * demix_mse + args.beta * demix_ssim
                mse_loss_value = l2_loss(recon_output, gt)
                ssim_loss_value = 1 - ssim_loss(recon_output, gt)
                recon_loss = args.alpha * mse_loss_value + args.beta * ssim_loss_value

                loss = demix_loss + recon_loss

                output_n = recon_output
                gt_n = gt
                with torch.no_grad():
                    ssim_index = ssim_loss(output_n, gt_n)
                    mse = l2_loss(recon_output, gt).item()
                    psnr = 20 * log10(1.0 / sqrt(mse)) if mse != 0 else 100

                loss_val.append(loss.item())
                ssim_val.append(ssim_index.item())
                psnr_val.append(psnr)

                if args.local_rank == 0:
                    print(f'Validation: Epoch {epoch}: batch {batch}/{len(val_loader)}: loss: {np.mean(loss_val):.4f} SSIM: {np.mean(ssim_val):.4f} PSNR: {np.mean(psnr_val):.2f} dB')

            if args.local_rank == 0:
                writer.add_scalar('Loss/val', np.mean(loss_val), epoch)
                writer.add_scalar('SSIM/val', np.mean(ssim_val), epoch)
                writer.add_scalar('PSNR/val', np.mean(psnr_val), epoch)

                if epoch == 1:
                    gt_np = gt.cpu().numpy()
                    im_gt = (np.clip(gt_np[0, 0, ...], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_gt.tif'), im_gt)

                if (epoch % args.num_freq_save) == 0:
                    recon_output_np = recon_output.cpu().numpy()
                    im_recon = (np.clip(recon_output_np[0, 0, ...], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_recon.tif'), im_recon)

                    demix_output_np = demix_output.cpu().numpy()
                    im_demix = (np.clip(demix_output_np[0, 0, ...], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_demix.tif'), im_demix)

        if args.local_rank == 0:
            df = pd.DataFrame({
                'epoch': [epoch],
                'loss_train': [np.mean(loss_train)],
                'ssim_train': [np.mean(ssim_train)],
                'psnr_train': [np.mean(psnr_train)],
                'loss_val': [np.mean(loss_val)],
                'ssim_val': [np.mean(ssim_val)],
                'psnr_val': [np.mean(psnr_val)]
            })
            losslogger = pd.concat([losslogger, df], ignore_index=True)

            trigger += 1
            current_ssim = np.mean(ssim_val)
            if current_ssim > best_ssim:
                save(args.dir_chck+ '/best_model/', model, optimizer, epoch, losslogger)
                best_ssim = np.mean(ssim_val)
                print("=>saved best model")
                trigger = 0
            else:
                save(args.dir_chck+ '/best_model1/', model, optimizer, epoch, losslogger)
                best_ssim = np.mean(ssim_val)
                print("=>saved a model")
                trigger = trigger

            if args.early_stop is not None and trigger >= args.early_stop:
                print("=> Early Stopping")
                break
              
            if (epoch % args.num_freq_save) == 0:
                save(args.dir_chck, model, optimizer, epoch, losslogger)
