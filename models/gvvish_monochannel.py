import sys
import numpy as np
import torch
import random
import os
import torchaudio.functional as F
from pathlib import Path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from resnet1d.resnet1d import ResNet1D

ERROR_PATH = "errors"
SAVE_PATH = "model.pt"
SEED = 1337
BATCH_SIZE = 512
NUM_EPOCHS = 200
PRETRAINED = True
NUM_CHANNELS = 4
LR = 1e-4
LIBS_DIR = Path(os.path.join(os.getcwd(),'external'))
sys.path.insert(0, str(LIBS_DIR.joinpath('resnet1d')))
N_STEPS = 8820

class GvvishMonoModel(torch.nn.Module):
    def __init__(self, audio_dir, centroid_dir, out_channels=2):
        super(GvvishMonoModel, self).__init__()
        self.resnet = ResNet1D(in_channels=1,
                                base_filters=64,
                                kernel_size=16,
                                stride=2,
                                groups=32,
                                n_block=36,
                                n_classes=out_channels,
                                downsample_gap=6,
                                increasefilter_gap=12,
                                use_bn=True,
                                use_do=True,
                                verbose=False).cuda()
        
        self.rand = random.Random(SEED)
        np.random.seed(0)
        deconv = np.load(audio_dir)
        print("Deconv Loaded")
        centroid = np.load(centroid_dir)
        print("Centroid Loaded")
        print(deconv.shape)
        print(centroid.shape)
        
        train_indices = np.load("models/indices/train_indices.npy")
        valid_indices = np.load("models/indices/valid_indices.npy")
        test_indices = np.load("models/indices/test_indices.npy")
        
        # Centroids
        train_xy = centroid[train_indices]
        valid_xy = centroid[valid_indices]
        test_xy = centroid[test_indices]
        
        self.train_mean = np.mean(train_xy, axis=0)
        self.train_std = np.std(train_xy, axis=0)
        
        print(self.train_std)
        print(np.max(train_xy, axis=0))
        print(np.min(train_xy, axis=0))
        train_xy = (train_xy - self.train_mean) / (self.train_std + 1e-8)
        valid_xy = (valid_xy - self.train_mean) / (self.train_std + 1e-8)
        test_xy =  (test_xy - self.train_mean) / (self.train_std + 1e-8)
        
        # Code insertion ends
        self.epsilon = 1e-2
        self.norm_val_min = np.min(np.concatenate((train_xy, valid_xy), axis=0))
        self.norm_val_range = np.max(np.concatenate((train_xy, valid_xy), axis=0)) - self.norm_val_min
        
        self.train_std_cuda = torch.Tensor(self.train_std).cuda()
        self.train_mean_cuda = torch.Tensor(self.train_mean).cuda()
        
        def resample(audio, ir=48000, tr=16000):
            resampled_waveform = F.resample(
            audio,
            ir,
            tr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492,
            )
            return resampled_waveform
        
        train_waves = deconv[train_indices, :]
        # 30950 seems to be the rough cutoff after which vggish treats the input as two examples.
        valid_waves = deconv[valid_indices, :]
        #Test Waves
        test_waves = deconv[test_indices, :]
        
        offset = 0
        
        precutoff = 92850
        train_waves = train_waves[..., (offset):(offset+precutoff)]
        valid_waves = valid_waves[..., (offset):(offset+precutoff)]
        test_waves = test_waves[..., (offset):(offset+precutoff)]
        
        train_waves = torch.Tensor(train_waves).cuda()
        self.train_xy = torch.Tensor(train_xy).cuda()
        
        valid_waves = torch.Tensor(valid_waves).cuda()
        self.valid_xy = torch.Tensor(valid_xy).cuda()
        
        test_waves = torch.Tensor(test_waves).cuda()
        self.test_xy = torch.Tensor(test_xy).cuda()
        
        print("Resampling")
        train_waves = resample(train_waves)
        valid_waves = resample(valid_waves)
        test_waves = resample(test_waves)
    
        vggish_cutoff = N_STEPS
        
        self.train_waves = train_waves[..., :vggish_cutoff]
        self.valid_waves = valid_waves[..., :vggish_cutoff]
        self.test_waves = test_waves[..., :vggish_cutoff]
        
        out_channels = 2
        
        print("Done Resampling")
        
        total_params = sum(p.numel() for p in self.resnet.parameters())
        trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad)
        
        print('Total parameters: %i'%total_params)
        print('Trainable parameters: %i'%trainable_params)

        self.xy_loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(params=self.resnet.parameters(), lr=LR, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999995)

        print("Number of samples per epoch:", train_waves.shape[0])

    def unnormalize(self, xy):
        return xy*(self.train_std_cuda + 1e-8) + self.train_mean_cuda

    def postprocess_net_output(self, output):
        output[:, :2] = self.norm_val_range * ((torch.tanh(output[:, :2]) * (1 + self.epsilon)) + 1) / 2 + self.norm_val_min
        return output
        
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.resnet.forward(x)
        return x

    def fit(self):
        N_iter = int(self.train_waves.shape[0] / BATCH_SIZE)
        train_losses = []
        train_xy_losses = []
        valid_losses = []
        valid_xy_losses = []
        step_count = 0
        
        for n in range(NUM_EPOCHS):
            print('Reshuffling for Epoch %i'%n, flush=True)
            rand_idx = np.random.permutation(self.train_waves.shape[0])
            self.resnet.train()
            self.optimizer.zero_grad()
            for i in range(N_iter):
                curr_idx = rand_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                self.train_waves = torch.mean(self.train_waves, dim=1, keepdim=True)
                print('HERREEEE', self.train_waves[curr_idx, :].shape)
                net_out = self.resnet(self.train_waves[curr_idx, :])
                results = self.postprocess_net_output(net_out)
                xy_loss = self.xy_loss_fn(results[:, :2], self.train_xy[curr_idx, :2])
                loss = xy_loss
                print('xy_loss: %0.3f'%loss)
                self.optimizer.zero_grad()
                loss.backward()
                train_loss = loss.item()
                train_losses.append((step_count, train_loss))
                train_xy_losses.append((step_count,xy_loss.item()))
                step_count+=1
                self.optimizer.step()
            self.scheduler.step()

        self.resnet.eval()
        valid_loss_xy_arr = np.zeros(self.valid_waves.shape[0], dtype=np.float32)
        valid_loss_arr = np.zeros(self.valid_waves.shape[0], dtype=np.float32)
        for i in range(self.valid_waves.shape[0]):
            with torch.no_grad():
                self.valid_waves = torch.mean(self.valid_waves, dim=1, keepdim=True)
                results = torch.squeeze(self.postprocess_net_output(self.resnet(torch.unsqueeze(self.valid_waves[i, :], axis=0)).view(-1, 1)))
            xy_loss = self.xy_loss_fn(results[:2], self.valid_xy[i, :2])
            valid_loss_xy_arr[i] = xy_loss.item()
            loss = xy_loss
            valid_loss_arr[i] = loss.item()
        valid_xy_loss = np.mean(valid_loss_xy_arr)
        valid_loss = np.mean(valid_loss_arr)
        print('Validation XY Loss: %0.3f'%valid_xy_loss)
        print('Validation Loss: %0.3f'%valid_loss)
        valid_losses.append((step_count, valid_loss))
        valid_xy_losses.append((step_count, valid_xy_loss))

        if not os.path.exists(ERROR_PATH):
            os.makedirs(ERROR_PATH)

        np.save(os.path.join(ERROR_PATH, 'train_losses.npy'), np.array(train_losses, dtype=np.float32))
        np.save(os.path.join(ERROR_PATH, 'valid_losses.npy'), np.array(valid_losses, dtype=np.float32))

        #Iterate through test
        test_errors = np.zeros(self.test_waves.shape[0], dtype=np.float32)

        for i in range(self.test_waves.shape[0]):
            with torch.no_grad():
                self.test_waves = torch.mean(self.test_waves, dim=1, keepdim=True)
                results = torch.squeeze(self.postprocess_net_output(self.resnet(torch.unsqueeze(self.test_waves[i, :], axis=0)).view(-1, 1)))
            
            
            test_errors[i] = torch.norm(self.unnormalize(results[:2]) - self.unnormalize(self.test_xy[i, :2])).item()

        print("TEST ERROR")
        print(test_errors)
        
        print("MEAN TEST ERROR",flush=True)
        print(np.mean(test_errors))
        print("MED TEST ERROR")
        print(np.median(test_errors))
        print("STD TEST ERROR")
        print(np.std(test_errors))

        np.save(os.path.join(ERROR_PATH, 'test_errors.npy'), np.array(test_errors, dtype=np.float32))        

        torch.save({
            'epoch': n,
            'model_state_dict': self.resnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'train_xy_losses': train_xy_losses,
            'valid_losses': valid_losses,
            'valid_xy_losses': valid_xy_losses,
            'train_mean': self.train_mean,
            'train_std': self.train_std,
            'norm_val_min':self.norm_val_min,
            'norm_val_range':self.norm_val_range,
            'lr': LR
            }, SAVE_PATH)

    def predict(self, X_test):
        with torch.no_grad():
            # Preprocess the input if needed
            X_test = torch.Tensor(X_test).cuda().unsqueeze(0)  # Add a batch dimension
            # Perform inference
            self.resnet.eval()
            print(X_test.shape)  # Print the shape to verify it's correct
            output = self.resnet(torch.mean(X_test, dim=1, keepdim=True))
            output = self.postprocess_net_output(output)
            
            # Convert output to numpy array
            np_output = output.detach().cpu().numpy()
        
        return np_output, self.unnormalize(output).detach().cpu().numpy()