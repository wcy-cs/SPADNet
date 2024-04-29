import torch
import torchvision
from data.dataset import Data
from torch.utils.data import DataLoader

import os
from option import args
import util
import model
import cv2


net = model.get_model(args)

pretrained_dict = torch.load(args.load_path,
 map_location='cuda:0')
print(util.get_parameter_number(net))
net.load_state_dict(pretrained_dict)
net = util.prepare(net)

save_name = "result-test"
if "helen" in args.dir_data:
    save_name = 'result-test-helen'
elif "CelebA" in args.dir_data:
    save_name = 'result-test-CelebA'

testdata = Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
net.eval()
val_psnr = 0
val_ssim = 0
with torch.no_grad():
    os.makedirs(os.path.join('/data_c/wcy/code/spadnet/experiment/', args.writer_name, save_name), exist_ok=True)
    net.eval()

    for batch, (lr, hr,  filename) in enumerate(testset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr, heatmap = net(lr)

        psnr1, _ = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu(), crop_border=8)
        val_psnr = val_psnr + psnr1
        torchvision.utils.save_image(sr[0],
                                     os.path.join('/data_c/wcy/code/spadnet/experiment/', args.writer_name, save_name,
                                                  '{}'.format(str(filename[0])[:-4]+".png")))
        # img = cv2.cvtColor(util.tensor2uint(sr[0], data_range=1), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join('/data_c/wcy/code/spadnet/experiment/', args.writer_name, save_name,
        #                                            '{}'.format(str(filename[0])[:-4]+".png")), img)


    print("Test psnr: {:.3f}".format(val_psnr / (len(testset))))
