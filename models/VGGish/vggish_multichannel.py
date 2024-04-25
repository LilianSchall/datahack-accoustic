import numpy as np
import torch
import os

from external.torchvggish.torchvggish.vggish import VGG, VGGish, make_layers
from external.torchvggish.torchvggish.vggish_input import waveform_to_examples
import external.torchvggish.torchvggish.mel_features as mel_features

import external.torchvggish.torchvggish.vggish_params as vggish_params

ERROR_PATH = "vggish_multi_errors"
SAVE_PATH = "vggish_muti_model.pt"

# TODO(sclarke): This was just copied from the hubconf file in the original repo, this should use an external reference.
vggish_model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}


class VGGishMulti:
    def __init__(self, saved_model=None, save_path="", error_path=""):
        in_channels = 4
        out_channels = 2
        self.model = CustomVGGish2(in_channels=in_channels, out_channels=out_channels)
        self.save_path = save_path
        self.error_path = error_path
        self.epsilon = 1e-2

        if saved_model is not None:
            saved_model = torch.load(saved_model)
            
            self.model.state_dict = saved_model['model_state_dict']
            self.train_std = saved_model['train_std']
            self.train_mean = saved_model['train_mean']
            self.norm_val_min = saved_model['norm_val_min']
            self.norm_val_range = saved_model['norm_val_range']

            # torch.save({
            #   'epoch': n,
            #   'model_state_dict': self.model.state_dict(),
            #   'optimizer_state_dict': optimizer.state_dict(),
            #   'train_losses': train_losses,
            #   'train_xy_losses': train_xy_losses,
            #   'valid_losses': valid_losses,
            #   'valid_xy_losses': valid_xy_losses,
            #   'train_mean': self.train_mean,
            #   'train_std': self.train_std,
            #   'norm_val_min': self.norm_val_min,
            #   'norm_val_range': self.norm_val_range,
            #   'lr': lr,
            # },
        else:
            # Build the model
            self.train_std = 0
            self.train_mean = 0
            self.norm_val_min = 0
            self.norm_val_range = 0
            self.epsilon = 1e-2

    def postprocess_net_output(self, output):
        output[:, :2] = self.norm_val_range * ((torch.tanh(output[:, :2]) * (1 + self.epsilon)) + 1) / 2 + self.norm_val_min
        return output
   
    def fit(self, X_train, y_train, X_valid, y_valid, num_epochs=300):

        train_xy = y_train
        valid_xy = y_valid
        
        self.train_mean = np.mean(train_xy, axis=0)
        self.train_std = np.std(train_xy, axis=0)

        train_xy = (train_xy - self.train_mean) / (self.train_std + 1e-8)
        valid_xy = (valid_xy - self.train_mean) / (self.train_std + 1e-8)

        self.norm_val_min = np.min(np.concatenate((train_xy, valid_xy), axis=0))
        self.norm_val_range = np.max(np.concatenate((train_xy, valid_xy), axis=0)) - self.norm_val_min
        
        train_waves = X_train
        valid_waves = X_valid

        precutoff = 92850
        train_waves = train_waves[..., :precutoff]
        valid_waves = valid_waves[..., :precutoff]

        train_waves = torch.Tensor(train_waves)#.cuda()
        train_xy = torch.Tensor(train_xy)#.cuda()
        valid_waves = torch.Tensor(valid_waves)#.cuda()
        valid_xy = torch.Tensor(valid_xy)#.cuda()

        vggish_cutoff = 15475
        train_waves = train_waves[..., :vggish_cutoff]
        valid_waves = valid_waves[..., :vggish_cutoff]

        lr = 1e-4

        xy_loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999995)

        batch_size = 32

        N_iter = int(train_waves.shape[0] / batch_size)
        train_losses = []
        train_xy_losses = []
        valid_losses = []
        valid_xy_losses = []
        step_count = 0
        for n in range(num_epochs):
            print(f'Reshuffling for Epoch {n}', flush=True)
            rand_idx = np.random.permutation(train_waves.shape[0])
            self.model.train()
            optimizer.zero_grad()
            for i in range(N_iter):
                curr_idx = rand_idx[i*batch_size:(i+1)*batch_size]
                net_out = self.model(train_waves[curr_idx, :])
                results = self.postprocess_net_output(net_out)
                xy_loss = xy_loss_fn(results[:, :2], train_xy[curr_idx, :2])
                loss = xy_loss
                optimizer.zero_grad()
                loss.backward()
                train_loss = loss.item()
                train_losses.append((step_count, train_loss))
                train_xy_losses.append((step_count, xy_loss.item()))
                step_count += 1
                optimizer.step()
                scheduler.step()

            self.model.eval()
            valid_loss_xy_arr = np.zeros(valid_waves.shape[0], dtype=np.float32)
            valid_loss_arr = np.zeros(valid_waves.shape[0], dtype=np.float32)
            for i in range(valid_waves.shape[0]):
                with torch.no_grad():
                    results = torch.squeeze(self.postprocess_net_output(self.model(torch.unsqueeze(valid_waves[i, :], axis=0)).view(-1, 1)))
                xy_loss = xy_loss_fn(results[:2], valid_xy[i, :2])
                valid_loss_xy_arr[i] = xy_loss.item()
                loss = xy_loss
                valid_loss_arr[i] = loss.item()
            valid_xy_loss = np.mean(valid_loss_xy_arr)
            valid_loss = np.mean(valid_loss_arr)
            print('Validation XY Loss: %0.3f'%valid_xy_loss)
            print('Validation Loss: %0.3f'%valid_loss)
            valid_losses.append((step_count, valid_loss))
            valid_xy_losses.append((step_count, valid_xy_loss))

            if not os.path.exists(self.error_path):
                os.makedirs(self.error_path)

            np.save(os.path.join(self.error_path, 'train_losses.npy'), np.array(train_losses, dtype=np.float32))
            np.save(os.path.join(self.error_path, 'valid_losses.npy'), np.array(valid_losses, dtype=np.float32))

            torch.save({
                'epoch': n,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_xy_losses': train_xy_losses,
                'valid_losses': valid_losses,
                'valid_xy_losses': valid_xy_losses,
                'train_mean': self.train_mean,
                'train_std': self.train_std,
                'norm_val_min': self.norm_val_min,
                'norm_val_range': self.norm_val_range,
                'lr': lr,
                }, self.save_path)
    
    def predict(self, X_test, y_test):

        test_xy = (y_test - self.train_mean) / (self.train_std + 1e-8)

        train_std_cuda = torch.Tensor(self.train_std)#.cuda()           
        train_mean_cuda = torch.Tensor(self.train_mean)#.cuda()

        def unnormalize(xy):
            return xy*(train_std_cuda + 1e-8) + train_mean_cuda

        test_waves = X_test

        precutoff = 92850
        test_waves = test_waves[..., :precutoff]

        test_waves = torch.Tensor(test_waves)#.cuda()
        test_xy = torch.Tensor(test_xy)#.cuda()

        vggish_cutoff = 15475
        test_waves = test_waves[..., :vggish_cutoff]

        # Iterate through test
        predictions = np.zeros((test_waves.shape[0], 2), dtype=np.float32)
        for i in range(test_waves.shape[0]):
            with torch.no_grad():
                results = torch.squeeze(self.postprocess_net_output(self.model(torch.unsqueeze(test_waves[i, :], axis=0)).view(-1, 1)))
            predictions[i] = unnormalize(results[:2])
            # test_errors[i] = torch.norm(unnormalize(results[:2]) - unnormalize(test_xy[i, :2])).item()

        return predictions



class VGGModel(torch.nn.Module):
    def __init__(self, fs=16000, pretrained=True):
        super(VGGModel, self).__init__()
        # self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish = vggish(pretrained=pretrained)
        self.vggish.eval()
        self.fs = fs
        
    def forward(self, x):
        x = (self.vggish.forward(torch.unsqueeze(x, axis=1), fs=self.fs) / 127) - 1
        # x = (self.vggish.forward(np.expand_dims(x, axis=1), fs=self.fs) / 127) - 1
        return x    
    

class CustomVGGish2(VGG):
    def __init__(self, in_channels=4, out_channels=2, device=None, p_dropout=0.0):
        super().__init__(make_layers(in_channels=in_channels))

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.to(self.device)

        self.mlp = torch.nn.Sequential(
                            torch.nn.Linear(in_features=128, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=256, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p_dropout),
                            torch.nn.Linear(in_features=256, out_features=out_channels, bias=True)
                            )#.cuda()

    def forward(self, x):
        x = self._preprocess(x)
        x = VGG.forward(self, x)
        x = self.mlp(x)
        return x

    def _preprocess(self, x):
    #The normalize flag will essentially determine if the preprocessing is like that in pretrained, or not.
        
        original_size = x.size()

        if x.dim() > 2:
            x = x.view(-1, x.size()[-1])
            reshape = True

        # Compute log mel spectrogram features.
        x = mel_features.log_mel_spectrogram(
            x,
            audio_sample_rate=vggish_params.SAMPLE_RATE,
            log_offset=vggish_params.LOG_OFFSET,
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=vggish_params.NUM_MEL_BINS,
            lower_edge_hertz=vggish_params.MEL_MIN_HZ,
            upper_edge_hertz=vggish_params.MEL_MAX_HZ)
            
        if reshape:
            x = x.view(original_size[0], original_size[1], x.size()[1], x.size()[2])

        return x
