import logging
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.layers.utterance_mvn import UtteranceMVN
import numpy as np
import torch
from typeguard import check_argument_types
from typing import Tuple

class FusedFrontends(AbsFrontend):
    def __init__(
        self, frontends=None, align_method="linear_projection", proj_dim=100, fs=16000, use_corr_loss=False, multi_layer=False
    ):
        assert check_argument_types()
        super().__init__()
        self.align_method = (
            align_method  # fusing method : linear_projection only for now
        )
        self.normalize = UtteranceMVN()
        self.proj_dim = proj_dim  # dim of the projection done on each frontend
        self.frontends = []  # list of the frontends to combine
        self.corr_mat = [] # correlation matrix of two input features
        self.use_corr_loss = use_corr_loss
        self.multi_layer = multi_layer # non-linear projection layers
        self.multilayer_cross_feature = {} # list of index for frontend that use cross feature
        if align_method == "wsum":
            num_front = len(frontends)
            if num_front > 1: # in case users use one frontend with fused
                self.weights = [1/num_front for _ in range(num_front - 1)]
                self.weights.append(1 - sum(self.weights))
            else:
                self.weights = [1]
            self.weights = torch.tensor(self.weights)
            self.weights = torch.nn.ParameterList(torch.nn.Parameter(i) for i in self.weights)

        for i, frontend in enumerate(frontends):
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = (
                    frontend.get("n_mels", 80),
                    fs,
                    frontend.get("n_fft", 512),
                    frontend.get("win_length"),
                    frontend.get("hop_length", 128),
                )
                window, center, normalized, onesided = (
                    frontend.get("window", "hann"),
                    frontend.get("center", True),
                    frontend.get("normalized", False),
                    frontend.get("onesided", True),
                )
                fmin, fmax, htk, apply_stft = (
                    frontend.get("fmin", None),
                    frontend.get("fmax", None),
                    frontend.get("htk", False),
                    frontend.get("apply_stft", True),
                )

                self.frontends.append(
                    DefaultFrontend(
                        n_mels=n_mels,
                        n_fft=n_fft,
                        fs=fs,
                        win_length=win_length,
                        hop_length=hop_length,
                        window=window,
                        center=center,
                        normalized=normalized,
                        onesided=onesided,
                        fmin=fmin,
                        fmax=fmax,
                        htk=htk,
                        apply_stft=apply_stft,
                    )
                )
            elif frontend_type == "s3prl":
                frontend_conf, download_dir, multilayer_feature, multilayer_cross_feature, layer = (
                    frontend.get("frontend_conf"),
                    frontend.get("download_dir"),
                    frontend.get("multilayer_feature", False),
                    frontend.get("multilayer_cross_feature", False),
                    frontend.get("layer"),
                )
                self.frontends.append(
                    S3prlFrontend(
                        fs=fs,
                        frontend_conf=frontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                        multilayer_cross_feature=multilayer_cross_feature,
                        layer=layer,
                    )
                )
                if multilayer_cross_feature:
                    model = self.frontends[-1]
                    self.multilayer_cross_feature[i] = [model.output_size(), model.upstream.num_layers]
            elif frontend_type == "whisper":
                from espnet2.asr.frontend.whisper import WhisperFrontend
                whisper_model, freeze_weights, download_dir = (
                    frontend.get("whisper_model"),
                    frontend.get("freeze_weights", True),
                    frontend.get("download_dir"),
                )
                self.frontends.append(
                    WhisperFrontend(
                        whisper_model=whisper_model,
                        freeze_weights=freeze_weights,
                        download_dir=download_dir,
                    )
                )
            else:
                raise NotImplementedError  # frontends are only default or s3prl

        self.frontends = torch.nn.ModuleList(self.frontends)
        if multilayer_cross_feature:
            from espnet2.tts.gst.style_encoder import MultiHeadedAttention
            dic_values = np.array(list(self.multilayer_cross_feature.values()))
            dimensions = dic_values[:,0]
            num_layers = dic_values[:,1]
            assert len(set(num_layers)) == 1 # make sure all models have the same number of layers 
            self.mha = []
            for i, _ in enumerate(self.frontends):
                for j, _ in enumerate(self.frontends[i+1:], i+1):
                    self.mha.append(
                        [MultiHeadedAttention(q_dim=dimensions[i],
                                            k_dim=dimensions[j],
                                            v_dim=dimensions[j],
                                            n_head=1,
                                            n_feat=self.proj_dim,
                                            dropout_rate=0.0,)
                        for _ in range(num_layers[i] // 2)] # using half of layers
                    )
                    self.mha.append(
                        [MultiHeadedAttention(q_dim=dimensions[j],
                                            k_dim=dimensions[i],
                                            v_dim=dimensions[i],
                                            n_head=1,
                                            n_feat=self.proj_dim,
                                            dropout_rate=0.0,)
                        for _ in range(num_layers[j] // 2)] # using half of layers
                    )
            self.mha = torch.nn.ModuleList([torch.nn.ModuleList(mha) for mha in self.mha])

        #self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        #self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]
        if self.align_method != "concat":
            if multilayer_cross_feature:
                extra_dim = self.proj_dim
            else:
                extra_dim = 0
            if self.multi_layer:
                self.projection_layers = [
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features=frontend.output_size() + extra_dim, out_features=256,),
                        torch.nn.GELU(),
                        torch.nn.Linear(in_features=256, out_features=256,),
                        torch.nn.GELU(),
                        torch.nn.Linear(in_features=256, out_features=self.proj_dim,)
                    )
                    for frontend in self.frontends
                ]
            else:
                self.projection_layers = [
                    torch.nn.Linear(
                        in_features=frontend.output_size() + extra_dim,
                        out_features=self.proj_dim,
                    )
                    for i, frontend in enumerate(self.frontends)
                ]
            self.projection_layers = torch.nn.ModuleList(self.projection_layers)
        if self.align_method == "co_attn":
            from espnet2.tts.gst.style_encoder import MultiHeadedAttention
            self.mha = []
            for i, _ in enumerate(self.frontends):
                for j, _ in enumerate(self.frontends[i+1:], i+1):
                    self.mha.append(
                        MultiHeadedAttention(q_dim=self.proj_dim,
                                            k_dim=self.proj_dim,
                                            v_dim=self.proj_dim,
                                            n_head=1,
                                            n_feat=self.proj_dim,
                                            dropout_rate=0.0,)
                    )
                    self.mha.append(
                        MultiHeadedAttention(q_dim=self.proj_dim,
                                            k_dim=self.proj_dim,
                                            v_dim=self.proj_dim,
                                            n_head=1,
                                            n_feat=self.proj_dim,
                                            dropout_rate=0.0,)
                    )
            self.mha = torch.nn.ModuleList(self.mha)

    # if no preencoder in conf, this function will return the wrong output_size. See espnet/espnet2/task/asr.py line 404.
    # TODO: return by align_method
    def output_size(self) -> int:
        return len(self.frontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # step 0 : get all frontends features
        feats = []
        for frontend in self.frontends:
            input_feats, feats_lens = frontend.forward(input, input_lengths)
            feats.append([input_feats, feats_lens])
        # processing cross-attention
        if len(self.multilayer_cross_feature) > 0:
            ts_outputs = []
            st_outputs = []
            for i, target in enumerate(self.frontends):
                for j, source in enumerate(self.frontends[i+1:], i+1):
                    #assert len(feats[i][0]) > 1 # make sure its features of all layers
                    t_input_feats, t_flens = feats[i]
                    s_input_feats, s_flens = feats[j]
                    ## make key_padding_mask for torch bulit-in MHA
                    #key_padding_mask = make_pad_mask(t_flens[0]).to(t_flens[0].device)
                    # make attn_mask
                    batch_size, flen, _ = t_input_feats[0].shape
                    attn_mask = torch.zeros(batch_size, flen, flen).bool().to(t_flens[0].device)
                    for b, length in enumerate(t_flens[0]): # iterate through every batch
                        attn_mask[b, :, length:] = False
                        attn_mask[b, length:, :] = False
                        attn_mask[b, :length, :length] = True
                    # calculating mha for every layers
                    for l in range(len(t_input_feats)):
                        q, kv = t_input_feats[l], s_input_feats[l]
                        if l in range(1,24,2):
                            # calculating mha using target -> source
                            output = self.mha[i][l//2](q, kv, kv, attn_mask)
                            ts_outputs.append(output)
                            # calculating mha using source -> target
                            output = self.mha[j][l//2](kv, q, q, attn_mask)
                            st_outputs.append(output)
                        else:
                            ts_outputs.append(q)
                            st_outputs.append(kv)
                    t_input_feats, _ = target.featurizer(t_input_feats, t_flens)
                    s_input_feats, _ = source.featurizer(s_input_feats, s_flens)
                    ts_input_feats, t_flens = target.featurizer2(ts_outputs, t_flens)
                    st_input_feats, s_flens = source.featurizer2(st_outputs, s_flens)
                    # concat input features with cross-attention features
                    t_input_feats = torch.cat((t_input_feats, ts_input_feats), dim=-1)
                    s_input_feats = torch.cat((s_input_feats, st_input_feats), dim=-1)
                    feats_lens = t_flens
                    feats[i] = [t_input_feats, t_flens]
                    feats[j] = [s_input_feats, s_flens]

        m = min([x[0].shape[1] for x in feats])
        if (
            self.align_method == "linear_projection"
        ):  # TODO(Dan): to add other align methods
            # first step : projections
            self.feats_proj = []
            #self.corr_mat = [] # for plot
            #self.correlation_matrix(feats[0][0], feats[1][0]) # for plot
            for i, _ in enumerate(self.frontends):
                input_feats = feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))
            #self.correlation_matrix(self.feats_proj[0], self.feats_proj[1]) # for plot
            
            # 2nd step : reshape
            self.feats_reshaped = []
            for i, _ in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                input_feats_proj = input_feats_proj.permute(0,2,1)
                input_feats_reshaped = torch.nn.functional.interpolate(
                    input_feats_proj, size=m
                )
                input_feats_reshaped = input_feats_reshaped.permute(0,2,1)
                self.feats_reshaped.append(input_feats_reshaped)
            if self.use_corr_loss and self.training:
                self.corr_mat = []
                self.correlation_matrix(self.feats_reshaped[0], self.feats_reshaped[1])
            # 3th step : normalize by frontend
            self.feats_normalized = []
            for i, _ in enumerate(self.frontends):
                # TODO(Sray): the feats_lens need to be divided by m first before normalize
                norm_feats, _ = self.normalize(self.feats_reshaped[i], feats_lens)
                self.feats_normalized.append(norm_feats)

            input_feats = torch.cat(
                self.feats_normalized, dim=-1
            )  # change the input size of the preencoder : proj_dim * n_frontends
        elif self.align_method == "concat":
            # 1nd step : downsample
            self.feats_reshaped = []
            for i, _ in enumerate(self.frontends):
                input_feats = feats[i][0]
                input_feats = input_feats.permute(0,2,1)
                input_feats_reshaped = torch.nn.functional.interpolate(
                    input_feats, size=m
                )
                input_feats_reshaped = input_feats_reshaped.permute(0,2,1)
                self.feats_reshaped.append(input_feats_reshaped)

            # 2nd step : normalize by frontend
            self.feats_normalized = []
            for i, _ in enumerate(self.frontends):
                norm_feats, _ = self.normalize(self.feats_reshaped[i], feats_lens)
                self.feats_normalized.append(norm_feats)

            input_feats = torch.cat(
                self.feats_normalized, dim=-1
            )
        elif self.align_method == "wsum":
            # first step : projections
            self.feats_proj = []
            #self.corr_mat = [] # for plot
            #self.correlation_matrix(feats[0][0], feats[1][0]) # for plot
            for i, _ in enumerate(self.frontends):
                input_feats = feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))
            #self.correlation_matrix(self.feats_proj[0], self.feats_proj[1]) # for plot
            # 2nd step : downsample
            self.feats_reshaped = []
            for i, _ in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                input_feats_proj = input_feats_proj.permute(0,2,1)
                input_feats_reshaped = torch.nn.functional.interpolate(
                    input_feats_proj, size=m
                )
                input_feats_reshaped = input_feats_reshaped.permute(0,2,1)
                self.feats_reshaped.append(input_feats_reshaped)
            if self.use_corr_loss and self.training:
                self.corr_mat = []
                self.correlation_matrix(self.feats_reshaped[0], self.feats_reshaped[1])
            # 3nd step : normalize by frontend
            self.feats_normalized = []
            for i, _ in enumerate(self.frontends):
                norm_feats, _ = self.normalize(self.feats_reshaped[i], feats_lens)
                self.feats_normalized.append(norm_feats)
            # 4th step : weighted sum
            input_feats = 0
            for i, _ in enumerate(self.frontends):
                input_feats += self.feats_normalized[i] * (self.weights[i] / sum(self.weights))
        elif self.align_method == "co_attn": # This is the co-attention method
            # first step : projections
            self.feats_proj = []
            for i, frontend in enumerate(self.frontends):
                input_feats = feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))
            # 2nd step : downsample
            self.feats_reshaped = []
            for i, frontend in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                input_feats_proj = input_feats_proj.permute(0,2,1)
                input_feats_reshaped = torch.nn.functional.interpolate(
                    input_feats_proj, size=m
                )
                input_feats_reshaped = input_feats_reshaped.permute(0,2,1)
                self.feats_reshaped.append(input_feats_reshaped)
            # 3rd step : co_attn
            # make attn_mask
            batch_size, flen, _ = self.feats_reshaped[0].shape
            attn_mask = torch.zeros(batch_size, flen, flen).bool().to(feats_lens.device)
            for b, length in enumerate(feats_lens): # iterate through every batch
                attn_mask[b, :, length:] = False
                attn_mask[b, length:, :] = False
                attn_mask[b, :length, :length] = True
            q, kv = self.feats_reshaped
            self.feats_final = []
            self.feats_final.append(self.mha[0](q, kv, kv, attn_mask) + q)
            self.feats_final.append(self.mha[1](kv, q, q, attn_mask) + kv)
            # 4th step: concat
            input_feats = torch.cat(
                self.feats_final, dim=-1
            )
        else:
            raise NotImplementedError
        if self.use_corr_loss:
            return input_feats, feats_lens, self.corr_mat
        return input_feats, feats_lens
    def correlation_matrix(self, f1: torch.Tensor, f2: torch.Tensor):
        # normalize through time and batch. [B,T,D]
        f1_normT = (f1 - f1.mean(1, keepdim=True)) / f1.std(1, keepdim=True)
        f2_normT = (f2 - f2.mean(1, keepdim=True)) / f2.std(1, keepdim=True)
        # We keep the code for normalize through BT or B only here:
        #f1_normBT = (f1_normT - f1_normT.mean(0, keepdim=True)) / f1_normT.std(0, keepdim=True)
        #f2_normBT = (f2_normT - f2_normT.mean(0, keepdim=True)) / f2_normT.std(0, keepdim=True)
        #f1_normB = ((f1 - f1.mean(0, keepdim=True)) / f1.std(0, keepdim=True)).transpose(0,1) # [T, B, D]
        #f2_normB = ((f2 - f2.mean(0, keepdim=True)) / f2.std(0, keepdim=True)).transpose(0,1)
        
        #self.corr_mat.append(torch.bmm(f1_normBT.transpose(-2,-1), f2_normBT) / (f2_normBT.shape[1] - 1)) # bmm(D*T, T*D) / (T - 1) / . Divide by T-1 because default unbiased option in std() is True.
        self.corr_mat.append(torch.bmm(f1_normT.transpose(-2,-1), f2_normT) / (f2_normT.shape[1] - 1))
        assert(self.corr_mat[-1].max() < 1 and self.corr_mat[-1].min() > -1), f"max: {self.corr_mat[-1].max()}, min: {self.corr_mat[-1].min()}"


