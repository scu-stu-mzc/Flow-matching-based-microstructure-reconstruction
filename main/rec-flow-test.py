import torch
import cv2
import os
import time

from models.unet.unet_rec_flow import UNetModelRecFlow
from torch.cuda.amp import autocast


def infer(
        model_path,
        base_channels=32,
        step=50,
        num_imgs=5,
        rec_size=64,
        save_path='./results',
        device='cuda'):

    model = UNetModelRecFlow(dim=(1, 64, 64, 64), num_channels=base_channels, num_res_blocks=1)
    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for i in range(num_imgs):
            print(f'Generating {i}th group...')
            dt = 1.0 / step
            x_t = torch.randn(1, 1, rec_size, rec_size, rec_size).to(device)
            for j in range(step):
                if j % 10 == 0:
                    print(f'Generating {i}th group, step {j}...')
                t = j * dt
                t = torch.tensor([t]).to(device)
                t = t.expand(x_t.size(0))
                with autocast():
                    v_pred = model(x=x_t, t=t, y=None)

                x_t = x_t + v_pred * dt

            x_t = (x_t + 1) * 0.5
            x_t = x_t.clamp(0, 1)

            tmp_path = save_path + '/' + str(i+1)
            os.makedirs(tmp_path, exist_ok=True)
            # 3d save
            for iter in range(x_t.size(2)):
                img = x_t[0, 0, iter, :, :].detach().cpu().numpy()
                img = img * 255
                img = img.astype('uint8')
                cv2.imwrite(os.path.join(tmp_path, f'{iter}.png'), img)


if __name__ == '__main__':
    # start time
    start_time = time.time()
    infer(model_path='./test_model/trained_model.pth',
          base_channels=32,
          step=50,
          num_imgs=1,
          rec_size=120,
          save_path='./test_model/results',
          device='cuda')

    # end time
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generation completed in {duration:.2f} seconds")
