import torch
import torch.nn as nn
from src.base.model import BaseModel

from .bfs_block import get_link_list, get_tree_emb_link_list
from .mlp import GraphMLP, FusionMLP, MultiLayerPerceptron
from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

device = 'cuda'

class DADiffNet(BaseModel):

    def __init__(self, supports, model_args, **args):
        super(DADiffNet, self).__init__(**args)
        # attributes
        self.num_nodes = args["node_num"]
        self.input_len = model_args["seq_len"]
        self.output_len = model_args["horizon"]

        self.his_len = model_args["his_len"]
        self.if_enhance = model_args["if_enhance"]
        self.enhance_dim = model_args["enhance_dim"]
        self.if_en = model_args["if_en"]
        self.if_de = model_args["if_de"]

        self.fusion_num_step = model_args["fusion_num_step"]
        self.fusion_num_layer = model_args["fusion_num_layer"]
        self.fusion_dim = model_args["fusion_dim"]
        self.fusion_out_dim = model_args["fusion_out_dim"]
        self.fusion_dropout = model_args["fusion_dropout"]

        self.if_forward = model_args["if_forward"]
        self.if_backward = model_args["if_backward"]
        self.if_use_augment = model_args["if_augment"]
        self.if_use_diff = model_args["if_diff"]
        self.adj_mx = supports
        self.node_dim = model_args["node_dim"]
        self.nhead = model_args["nhead"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.tree_emb_len_list = model_args["tree_emb_len_list"]
        self.long_diff_len = model_args["long_diff_len"]
        self.diff_len = model_args["diff_len"]

        self.augment_adj_len = []
        self.augment_adj_mx = []

        if self.if_use_diff:
            self.input_len = self.input_len - 1
            self.his_len = self.his_len - 1
        if self.if_use_augment:
            adj_link_list = get_link_list(self.adj_mx[0], self.num_nodes)
            for l in self.tree_emb_len_list:
                emb_len, temp_emb = get_tree_emb_link_list(adj_link_list, l, self.num_nodes, "max", device)
                self.augment_adj_len.append(emb_len)
                self.augment_adj_mx.append(temp_emb)

        self.graph_num = 1 * int(self.if_forward) + 1 * int(self.if_backward) + 1 * int(self.if_use_augment)

        self.st_dim = (self.graph_num > 0) * self.node_dim + \
                      self.if_time_in_day * self.temp_dim_tid + \
                      self.if_day_in_week * self.temp_dim_diw

        self.output_dim = self.fusion_num_step * self.fusion_out_dim
        if self.if_use_augment:
            if self.if_forward:
                self.adj_mx_forward_encoder = nn.Sequential(
                    GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
                )
            self.adj_mx_augment_encoder = nn.Sequential(
                GraphMLP(input_dim=self.augment_adj_len[0], hidden_dim=self.node_dim)
            )
        else:
            if self.if_forward:
                self.adj_mx_forward_encoder = nn.Sequential(
                    GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
                )
            if self.if_backward:
                self.adj_mx_backward_encoder = nn.Sequential(
                    GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
                )

        self.fusion_layers = nn.ModuleList([
            FusionMLP(
                input_dimz=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                hidden_dim=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                out_dim=self.fusion_out_dim,
                graph_num=self.graph_num,
                first=True, **model_args)
        ])
        for _ in range(self.fusion_num_step - 1):
            self.fusion_layers.append(
                FusionMLP(input_dimz=self.st_dim + self.fusion_out_dim,
                          hidden_dim=self.st_dim + self.fusion_out_dim,
                          out_dim=self.fusion_out_dim,
                          graph_num=self.graph_num,
                          first=False, **model_args)
            )
        if self.fusion_num_step > 1:
            self.regression_layer = nn.Sequential(
                *[MultiLayerPerceptron(input_dimz=self.output_dim,
                                       hidden_dim=self.output_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.output_dim, out_features=self.output_len, bias=True),
            )

        if self.if_enhance:
            self.long_linear = nn.Sequential(
                nn.Linear(in_features=self.his_len, out_features=self.enhance_dim, bias=True),
            )

        if self.if_en:
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=self.input_len, nhead=1, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)
        if self.if_de:
            self.decoder = TransformerDecoder(
                TransformerDecoderLayer(d_model=self.input_len, nhead=1, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)

    def forward(self, history_data, label, his):  # (b, t, n, f)

        long_input_data_emb = []
        if self.if_enhance:
            long_input_data = his[..., 0].transpose(1, 2)
            if self.if_use_diff:
                long_input_data = long_input_data[..., 1:] - long_input_data[..., :-1]
            long_input_data = self.long_linear(long_input_data)
            long_input_data_emb.append(long_input_data)

        input_data = history_data[..., 0].transpose(1, 2)
        batch_size, num_nodes, _ = input_data.shape
        if self.if_use_diff:
            input_data = input_data[..., 1:] - input_data[..., :-1]
        input_data_en = []
        input_data_de = []
        if self.if_en:
            input_data_en.append(self.encoder(input_data))
        else:
            input_data_en.append(input_data)
        if self.if_de:
            input_data_de.append(self.decoder(input_data, input_data_en[0]))

        time_series_emb = [torch.cat(long_input_data_emb + input_data_en + input_data_de, dim=2)]

        node_forward_emb = []
        node_augment_emb = []
        if self.if_forward:
            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(batch_size, -1, -1)
            node_forward_emb.append(node_forward)

        if self.if_use_augment or self.if_backward:
            if self.if_use_augment:
                node_augment = self.augment_adj_mx[0].to(device)
                node_augment = self.adj_mx_augment_encoder(node_augment.unsqueeze(0)).expand(batch_size, -1, -1)
                node_augment_emb.append(node_augment)
            else:
                node_backward = self.adj_mx[1].to(device)
                node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(batch_size, -1, -1)
                node_augment_emb.append(node_backward)

        predicts = []
        predict_emb = []
        hidden_forward_emb = []
        hidden_augment_emb = []
        for index, layer in enumerate(self.fusion_layers):
            predict, hidden_forward, hidden_augment, \
            node_forward_emb_out, node_augment_emb_out = layer(history_data, time_series_emb, predict_emb,
                                                                node_forward_emb, node_augment_emb,
                                                                hidden_forward_emb, hidden_augment_emb)
            predicts.append(predict)
            predict_emb = [predict]
            time_series_emb = []
            hidden_forward_emb = hidden_forward
            hidden_augment_emb = hidden_augment

            node_forward_emb = node_forward_emb_out
            node_augment_emb = node_augment_emb_out

        predicts = torch.cat(predicts, dim=2)
        if self.fusion_num_step > 1:
            predicts = self.regression_layer(predicts)
            if self.if_use_diff:
                trend = history_data[:, -1:,:, 0].transpose(1, 2).expand(-1, -1, self.output_len)
                predicts = predicts + trend
        return predicts.transpose(1, 2).unsqueeze(-1)
