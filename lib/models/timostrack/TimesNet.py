import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from lib.models.layers.Embed import DataEmbedding
from lib.models.layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetTracking(nn.Module):
    """
    TimesNet for Object Tracking
    Input: historical bounding box positions (x, y, w, h) [B, T, 4]
    Output: predicted bounding box positions [B, pred_len, 4]
    """

    def __init__(self, configs):
        super(TimesNetTracking, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # TimesNet layers
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # Embedding layer for 4D bbox coordinates (x, y, w, h)
        self.enc_embedding = DataEmbedding(4, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Prediction layers
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, 4, bias=True)  # Output 4 values: x, y, w, h

        # Optional: velocity prediction
        if hasattr(configs, 'predict_velocity') and configs.predict_velocity:
            self.velocity_projection = nn.Linear(configs.d_model, 4, bias=True)

        # Optional: confidence score
        if hasattr(configs, 'predict_confidence') and configs.predict_confidence:
            self.confidence_projection = nn.Linear(configs.d_model, 1, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: [B, seq_len, 4] - historical bbox positions (x, y, w, h)
            x_mark_enc: op "python.languageServer": "None",tional temporal features
            x_dec: not used in tracking
            x_mark_dec: not used in tracking
            mask: not used in tracking
        Returns:
            predicted bbox positions [B, pred_len, 4]
        """
        return self.forecast(x_enc, x_mark_enc)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension,[4,50+5,16]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))#[4,55,16]

        # project back to bbox coordinates
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))

        # Return only the prediction part
        return dec_out[:, -self.pred_len:, :],enc_out  # [B, pred_len, 4]

    def predict_with_velocity(self, x_enc, x_mark_enc=None):
        """Predict both position and velocity"""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        position_out = self.projection(enc_out)
        velocity_out = self.velocity_projection(enc_out)

        # De-normalize
        position_out = position_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        position_out = position_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))

        return position_out[:, -self.pred_len:, :], velocity_out[:, -self.pred_len:, :]

    def predict_with_confidence(self, x_enc, x_mark_enc=None):
        """Predict position with confidence score"""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        position_out = self.projection(enc_out)
        confidence_out = torch.sigmoid(self.confidence_projection(enc_out))

        # De-normalize position
        position_out = position_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        position_out = position_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))

        return position_out[:, -self.pred_len:, :], confidence_out[:, -self.pred_len:, :]


class TrackingConfig:
    """Configuration class for TimesNet Tracking"""

    def __init__(self,
                 seq_len=50,  # Length of historical sequence
                 pred_len=5,  # Number of frames to predict
                 d_model=16,  # Hidden dimension
                 d_ff=64,  # Feed-forward dimension
                 e_layers=3,  # Number of TimesBlock layers
                 top_k=3,  # Number of top frequencies
                 num_kernels=6,  # Number of convolution kernels
                 embed='timeF',  # Embedding type
                 freq='h',  # Frequency
                 dropout=0.1,  # Dropout rate
                 predict_velocity=False,  # Whether to predict velocity
                 predict_confidence=False):  # Whether to predict confidence

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.predict_velocity = predict_velocity
        self.predict_confidence = predict_confidence
        self.enc_in = 4  # 输入的维度


if __name__ == "__main__":
    # 简单的配置类
    class Configs:
        def __init__(self):
            self.task_name = 'long_term_forecast'  # 可改为 'imputation', 'anomaly_detection', 'classification'
            self.seq_len = 24
            self.label_len = 18
            self.pred_len = 1
            self.e_layers = 2
            self.d_model = 16
            self.d_ff = 32
            self.num_kernels = 6
            self.enc_in = 8
            self.c_out = 8
            self.embed = 'timeF'
            self.freq = 'h'
            self.dropout = 0.1
            self.top_k = 2
            self.num_class = 3  # 仅分类任务用

    configs = TrackingConfig()
    model = TimesNetTracking(configs)

    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)#4*30*4
    x_mark_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)#4*24*8
    x_mark_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    mask = torch.ones(batch_size, configs.seq_len, configs.enc_in)

    with torch.no_grad():
        output = model(x_enc, None)
        print("Output shape:", output.shape)
