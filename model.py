import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from mamba_ssm import Mamba


class LaplacianLayer(nn.Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight', laplacian_kernel)


    def forward(self, x):
        x = x.unsqueeze(1)
        weight = self.weight.to(x.device)
        laplacian_spec = F.conv2d(x, weight, padding=1)
        laplacian_spec = laplacian_spec.squeeze(1)
        return laplacian_spec


class CustomLoss(nn.Module):
    def __init__(self, lambda_value=0.5):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value
        self.l1_loss = nn.L1Loss()
        self.laplacian_layer = LaplacianLayer()

    def forward(self, x, y):
        l_mel = self.l1_loss(x, y)
        lap_x = self.laplacian_layer(x)
        lap_y = self.laplacian_layer(y)
        l_lap = F.mse_loss(lap_x, lap_y)
        return l_mel + self.lambda_value * l_lap




class TemporalPyramidMamba(nn.Module):
    def __init__(self, d_model, num_mamba_layers_per_block=1):
        super().__init__()
        self.d_model = d_model
        self.num_mamba_layers_per_block = num_mamba_layers_per_block


        self.downsample1 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)  # FS -> FM
        self.downsample2 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)  # FM -> FL


        self.upsample1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)


        self.mamba_S = nn.Sequential(*[MambaLayer(d_model) for _ in range(num_mamba_layers_per_block)])
        self.mamba_M = nn.Sequential(*[MambaLayer(d_model) for _ in range(num_mamba_layers_per_block)])
        self.mamba_L = nn.Sequential(*[MambaLayer(d_model) for _ in range(num_mamba_layers_per_block)])

    def forward(self, x):

        B, D, T = x.shape
        assert D == self.d_model, "unmatched"

        F_S = x
        F_M = self.downsample1(F_S)
        F_L = self.downsample2(F_M)

        Y_S = self.mamba_S(F_S.permute(2, 0, 1)).permute(1, 2, 0)  # (B, D, T)
        Y_M = self.mamba_M(F_M.permute(2, 0, 1)).permute(1, 2, 0)  # (B, D, T/2)
        Y_L = self.mamba_L(F_L.permute(2, 0, 1)).permute(1, 2, 0)  # (B, D, T/4)

        # (1) FL -> FM
        Y_L_up = self.upsample1(Y_L)
        if Y_L_up.shape[2] != Y_M.shape[2]:
            Y_L_up = F.pad(Y_L_up, (0, Y_M.shape[2] - Y_L_up.shape[2]))
        Y_M_p = Y_M + Y_L_up

        # (2) FM -> FS
        Y_M_up = self.upsample2(Y_M_p)
        if Y_M_up.shape[2] != Y_S.shape[2]:
            Y_M_up = F.pad(Y_M_up, (0, Y_S.shape[2] - Y_M_up.shape[2]))
        Y_S_p = Y_S + Y_M_up 

        return Y_S_p.permute(2, 0, 1)



class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super(MambaLayer, self).__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # Mamba layer expects input of shape (B, L, D)
        # x comes in as (L, B, D), so we need to transpose
        x = x.permute(1, 0, 2)
        x = self.mamba(x)
        # Transpose back to (L, B, D)
        x = x.permute(1, 0, 2)
        return x

class FrequencyAwareLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.filter_real = nn.Parameter(torch.randn(d_model, 1)* 0.75)  
        self.filter_imag = nn.Parameter(torch.randn(d_model, 1)* 0.75)  
        self.eps = 1e-8
    

    def forward(self, x):
        batch_size, channels, time = x.shape

        x_fft = torch.fft.fft(x, dim=2) 
        
        filter_complex = torch.complex(self.filter_real, self.filter_imag)  # [channels, 1]
        x_fft_filtered = x_fft * filter_complex 
        
        x_ifft = torch.fft.ifft(x_fft_filtered, dim=2).real
        
        x_ifft = torch.clamp(x_ifft, -1e4, 1e4)
        
        return x_ifft



class NeuroMamba(nn.Module):
    def __init__(self,
                 in_channels=396,
                 cnn_dim=260,   # note that cnn_dim here denotes output channel in the last cnn
                 rnn_dim=260,
                 KS=4,
                 num_rnn_layers=3,
                 num_transformer_layers=8,
                 num_transformer_heads=10,
                 dropout_cnn_pre=0.1,
                 dropout_rnn=0.5,
                 dropout_cnn_post=0.1,
                 bidirectional=True,
                 n_classes=80,
                 rnn_type="biLSTM",
                 relu1=True,
                 sigmoid1=True,
                 leaky_relu1=False,
                 bn1=False,
                 num_pyramid_layers=3,
                 num_mamba_layers_per_block=2,
                 ):

        super(NeuroMamba, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                               out_channels=cnn_dim*4,
                               kernel_size=4,
                               stride=4,
                               padding=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_dim*4,
                               out_channels=cnn_dim*2,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.conv3 = nn.Conv1d(in_channels=cnn_dim*2,
                               out_channels=cnn_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)        


        self.predropout = nn.Dropout(dropout_cnn_pre)
        self.bn1 = nn.BatchNorm1d(cnn_dim)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        self.leaky_relu1 = nn.LeakyReLU()
        self.is_transformer = False
        self.is_mamba = False
        self.is_pyramidmamba = False

        if rnn_type == 'biLSTM':
            self.rnn = nn.LSTM(input_size=cnn_dim,
                               hidden_size=rnn_dim,
                               num_layers=num_rnn_layers,
                               bidirectional=bidirectional,
                               dropout=dropout_rnn
                               )

        elif rnn_type == 'biGRU':
             self.rnn = nn.GRU(input_size=cnn_dim,
                          hidden_size=rnn_dim,
                          num_layers=num_rnn_layers,
                          bidirectional=bidirectional,
                          dropout=dropout_rnn
                          )

        elif rnn_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(
                                d_model=cnn_dim,
                                nhead=num_transformer_heads,
                                dim_feedforward=rnn_dim,
                                dropout=dropout_rnn
                                )
            self.rnn = nn.TransformerEncoder(
                                transformer_layer,
                                num_layers=num_transformer_layers
                                )
            self.is_transformer = True

        elif rnn_type == 'mamba':
            self.rnn = nn.Sequential(*[MambaLayer(d_model=cnn_dim) for _ in range(num_rnn_layers)])
            self.is_mamba = True
        # TP-Mamba
        elif rnn_type == 'TPmamba':
            self.rnn = TemporalPyramidMamba(
                d_model=cnn_dim,
                # num_pyramid_layers=num_pyramid_layers,
                # num_mamba_layers_per_block=num_mamba_layers_per_block
            )
            self.is_pyramidmamba = True


        if not self.is_mamba and bidirectional == True:
            mult = 2
        else:
            mult = 1

        mult = 2 if (not self.is_pyramidmamba and bidirectional) else 1 # pyramid mamba

        if self.is_mamba:
            self.fc = nn.Linear(cnn_dim * mult, n_classes)
        elif self.is_pyramidmamba:
            in_feat = cnn_dim * mult
            self.fc = nn.Linear(in_feat, n_classes)
        else:
            self.fc = nn.Linear(rnn_dim * mult, n_classes)


    def forward(self, x):
        x = x.contiguous().permute(0, 2, 1)  
        
        x = self.conv(x)
        #x = self.relu1(x)
        #x = self.conv2(x)

        x = self.conv2(x)

        x = self.conv3(x)

        #x = self.bn1(x)
        #x = self.relu1(x)

        #x = self.conv2(x)
        #x = self.frequency_aware(x)
        #x = self.bn1(x)
        #x = self.relu1(x)

        if self.bn1:
            x = self.bn1(x)        
        if self.relu1:
            x = self.relu1(x)
        if self.sigmoid1:
            x = self.sigmoid1(x)
        if self.leaky_relu1:
            x = self.leaky_relu1(x)
        x = self.predropout(x)



        if self.is_pyramidmamba:
            x = self.rnn(x)
        else:
            x = x.contiguous().permute(2, 0, 1)
            if self.is_transformer or self.is_mamba:
                x = self.rnn(x)
            else:
                x, _ = self.rnn(x)
        
        x = self.fc(x)  # (T_mel, B, n_classes)
        x = x.contiguous().permute(1, 0, 2)  # (B, T_mel, n_classes)
        
        return x


class AFM_LSTM(nn.Module):
    def __init__(self,  
                 in_channels=396,
                 cnn_dim = 260,
                 rnn_dim =260, 
                 KS = 4, 
                 num_rnn_layers =3,
                 num_transformer_layers = 8,
                 num_transformer_heads = 10,
                 dropout_cnn_pre = 0.1,
                 dropout_rnn = 0.5, 
                 dropout_cnn_post = 0.1,
                 bidirectional = True, 
                 n_classes = 80,
                 rnn_type = "biLSTM",
                 relu1 = True,
                 sigmoid1 = True,
                 leaky_relu1=False,
                    bn1 = False,
                 ):
        
        super(AFM_LSTM, self).__init__()

        self.frequency_aware1 = FrequencyAwareLayer(d_model=in_channels)
        
        self.conv = nn.Conv1d(in_channels=in_channels,
                               out_channels=cnn_dim*4,
                               kernel_size=4,
                               stride=4,
                               padding=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_dim*4,
                               out_channels=cnn_dim*2,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.conv3 = nn.Conv1d(in_channels=cnn_dim*2,
                               out_channels=cnn_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)        

        self.predropout = nn.Dropout(dropout_cnn_pre)



        if bn1:
            self.bn1 = nn.BatchNorm1d(cnn_dim)
        else:
            self.bn1 = None
        if relu1:
            self.relu1 = nn.ReLU()
        else:
            self.relu1 = None
        
        if sigmoid1:
            self.sigmoid1 = nn.Sigmoid()
        else:
            self.sigmoid1 = None
            
        if leaky_relu1:
            self.leaky_relu1 = nn.LeakyReLU()
        else:
            self.leaky_relu1 = None   
        
        self.is_transformer = False
        
        if rnn_type == 'biLSTM':
            self.rnn = nn.LSTM(input_size=cnn_dim, 
                               hidden_size=rnn_dim, 
                               num_layers=num_rnn_layers, 
                               bidirectional= bidirectional,
                               dropout= dropout_rnn
                               )
            
        elif rnn_type == 'biGRU':
             self.rnn = nn.GRU(input_size=cnn_dim, 
                          hidden_size=rnn_dim, 
                          num_layers=num_rnn_layers, 
                          bidirectional= bidirectional,
                          dropout=dropout_rnn
                          )
             
        elif rnn_type == 'transformer': # bidirectional to false
            transformer_layer = nn.TransformerEncoderLayer(
                                d_model=cnn_dim, 
                                nhead=num_transformer_heads, 
                                dim_feedforward=rnn_dim, 
                                dropout=dropout_rnn
                                )
            self.rnn = nn.TransformerEncoder(
                                transformer_layer, 
                                num_layers=num_transformer_layers
                                )
            self.is_transformer = True

        if bidirectional == True:
            mult = 2
        else:
            mult = 1


        self.fc = nn.Linear(rnn_dim*mult, n_classes)  # 260 because of bidirectionality (130*2)

    def forward(self, x):

        x = x.contiguous().permute(0, 2, 1)

        x = self.frequency_aware1(x)
        x = self.conv(x)

        x = self.conv2(x)

        x = self.conv3(x)


        
        if self.bn1:
            x = self.bn1(x)        
        if self.relu1:
            x = self.relu1(x)
        if self.sigmoid1:
            x = self.sigmoid1(x)
        if self.leaky_relu1:
            x = self.leaky_relu1(x)
        x = self.predropout(x)
        
        
        
        
        x = x.contiguous().permute(2, 0, 1)

        
        if self.is_transformer:
            x = self.rnn(x)
        else:
            x, _ = self.rnn(x)


        x = self.fc(x)

        
        x = x.contiguous().permute(1,0,2)

        
        return x    


class CNN_TPM(nn.Module):
    def __init__(self,
                 in_channels=396,
                 cnn_dim=260,   # note that cnn_dim here denotes output channel in the last cnn
                 rnn_dim=260,
                 KS=4,
                 num_rnn_layers=3,
                 num_transformer_layers=8,
                 num_transformer_heads=10,
                 dropout_cnn_pre=0.1,
                 dropout_rnn=0.5,
                 dropout_cnn_post=0.1,
                 bidirectional=True,
                 n_classes=80,
                 rnn_type="biLSTM",
                 relu1=True,
                 sigmoid1=True,
                 leaky_relu1=False,
                 bn1=False,

                 num_pyramid_layers=3,
                 num_mamba_layers_per_block=2,
                 ):

        super(CNN_TPM, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                               out_channels=cnn_dim*4,
                               kernel_size=4,
                               stride=4,
                               padding=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_dim*4,
                               out_channels=cnn_dim*2,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.conv3 = nn.Conv1d(in_channels=cnn_dim*2,
                               out_channels=cnn_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)        


        self.predropout = nn.Dropout(dropout_cnn_pre)
        self.bn1 = nn.BatchNorm1d(cnn_dim)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        self.leaky_relu1 = nn.LeakyReLU()
        self.is_transformer = False
        self.is_mamba = False
        self.is_pyramidmamba = False

        if rnn_type == 'biLSTM':
            self.rnn = nn.LSTM(input_size=cnn_dim,
                               hidden_size=rnn_dim,
                               num_layers=num_rnn_layers,
                               bidirectional=bidirectional,
                               dropout=dropout_rnn
                               )

        elif rnn_type == 'biGRU':
             self.rnn = nn.GRU(input_size=cnn_dim,
                          hidden_size=rnn_dim,
                          num_layers=num_rnn_layers,
                          bidirectional=bidirectional,
                          dropout=dropout_rnn
                          )

        elif rnn_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(
                                d_model=cnn_dim,
                                nhead=num_transformer_heads,
                                dim_feedforward=rnn_dim,
                                dropout=dropout_rnn
                                )
            self.rnn = nn.TransformerEncoder(
                                transformer_layer,
                                num_layers=num_transformer_layers
                                )
            self.is_transformer = True

        elif rnn_type == 'mamba':
            self.rnn = nn.Sequential(*[MambaLayer(d_model=cnn_dim) for _ in range(num_rnn_layers)])
            self.is_mamba = True
        # TP-Mamba
        elif rnn_type == 'TPmamba':

            self.rnn = TemporalPyramidMamba(
                d_model=cnn_dim,
                # num_pyramid_layers=num_pyramid_layers,
                # num_mamba_layers_per_block=num_mamba_layers_per_block
            )
            self.is_pyramidmamba = True


        if not self.is_mamba and bidirectional == True:
            mult = 2
        else:
            mult = 1

        mult = 2 if (not self.is_pyramidmamba and bidirectional) else 1 # pyramid mamba

        if self.is_mamba:
            self.fc = nn.Linear(cnn_dim * mult, n_classes)
        elif self.is_pyramidmamba:
            in_feat = cnn_dim * mult
            self.fc = nn.Linear(in_feat, n_classes)
        else:
            self.fc = nn.Linear(rnn_dim * mult, n_classes)


    def forward(self, x):

        x = x.contiguous().permute(0, 2, 1)  
        

        x = self.conv(x)


        x = self.conv2(x)

        x = self.conv3(x)

        #x = self.bn1(x)
        #x = self.relu1(x)

        #x = self.conv2(x)
        #x = self.frequency_aware(x)
        #x = self.bn1(x)
        #x = self.relu1(x)


        if self.bn1:
            x = self.bn1(x)        
        if self.relu1:
            x = self.relu1(x)
        if self.sigmoid1:
            x = self.sigmoid1(x)
        if self.leaky_relu1:
            x = self.leaky_relu1(x)
        x = self.predropout(x)




        if self.is_pyramidmamba:

            x = self.rnn(x)
        else:

            x = x.contiguous().permute(2, 0, 1)
            if self.is_transformer or self.is_mamba:
                x = self.rnn(x)
            else:
                x, _ = self.rnn(x)

        x = self.fc(x)  # (T_mel, B, n_classes)
        x = x.contiguous().permute(1, 0, 2)  # (B, T_mel, n_classes)
        
        return x


class CNN_LSTM(nn.Module):
    def __init__(self,  
                 in_channels=396,
                 cnn_dim = 260,
                 rnn_dim =260, 
                 KS = 4, 
                 num_rnn_layers =3,
                 num_transformer_layers = 8,
                 num_transformer_heads = 10,
                 dropout_cnn_pre = 0.1,
                 dropout_rnn = 0.5, 
                 dropout_cnn_post = 0.1,
                 bidirectional = True, 
                 n_classes = 80,
                 rnn_type = "biLSTM",
                 relu1 = True,
                 sigmoid1 = True,
                 leaky_relu1=False,
                    bn1 = False,
                 ):
        
        super(CNN_LSTM, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_channels,
                               out_channels=cnn_dim*4,
                               kernel_size=4,
                               stride=4,
                               padding=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_dim*4,
                               out_channels=cnn_dim*2,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.conv3 = nn.Conv1d(in_channels=cnn_dim*2,
                               out_channels=cnn_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)        


        self.predropout = nn.Dropout(dropout_cnn_pre)



        if bn1:
            self.bn1 = nn.BatchNorm1d(cnn_dim)
        else:
            self.bn1 = None
        if relu1:
            self.relu1 = nn.ReLU()
        else:
            self.relu1 = None
        
        if sigmoid1:
            self.sigmoid1 = nn.Sigmoid()
        else:
            self.sigmoid1 = None
            
        if leaky_relu1:
            self.leaky_relu1 = nn.LeakyReLU()
        else:
            self.leaky_relu1 = None   
        
        self.is_transformer = False
        
        if rnn_type == 'biLSTM':
            self.rnn = nn.LSTM(input_size=cnn_dim, 
                               hidden_size=rnn_dim, 
                               num_layers=num_rnn_layers, 
                               bidirectional= bidirectional,
                               dropout= dropout_rnn
                               )
            
        elif rnn_type == 'biGRU':
             self.rnn = nn.GRU(input_size=cnn_dim, 
                          hidden_size=rnn_dim, 
                          num_layers=num_rnn_layers, 
                          bidirectional= bidirectional,
                          dropout=dropout_rnn
                          )
             
        elif rnn_type == 'transformer': # bidirectional to false
            transformer_layer = nn.TransformerEncoderLayer(
                                d_model=cnn_dim, 
                                nhead=num_transformer_heads, 
                                dim_feedforward=rnn_dim, 
                                dropout=dropout_rnn
                                )
            self.rnn = nn.TransformerEncoder(
                                transformer_layer, 
                                num_layers=num_transformer_layers
                                )
            self.is_transformer = True

        if bidirectional == True:
            mult = 2
        else:
            mult = 1


        self.fc = nn.Linear(rnn_dim*mult, n_classes)  # 260 because of bidirectionality (130*2)

    def forward(self, x):
        
        # x comes in batch_size, time_step, channel
        x = x.contiguous().permute(0, 2, 1)
        # now batch_size, channel, time_step
        
        x = self.conv(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        

        
        if self.bn1:
            x = self.bn1(x)        
        if self.relu1:
            x = self.relu1(x)
        if self.sigmoid1:
            x = self.sigmoid1(x)
        if self.leaky_relu1:
            x = self.leaky_relu1(x)
        x = self.predropout(x)
        
        
        # now batch_size, channel(260), time_step(151)
        
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        
        if self.is_transformer:
            x = self.rnn(x)
        else:
            x, _ = self.rnn(x)
        # now t, bs, c*k

        x = self.fc(x)
        # now t, bs, n_classes
        
        x = x.contiguous().permute(1,0,2)
        # now bs, t(should be 151, time in mel), n_classes(should be 80, num of bins in mel)  
        
        return x

