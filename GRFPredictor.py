import os
from collections import deque
import copy
import random
import numpy as np
import pickle
from .const import IMU_LIST, IMU_FIELDS, ACC_ALL, GYR_ALL, MAX_BUFFER_LEN, GRAVITY, WEIGHT_LOC, HEIGHT_LOC
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch


lstm_unit, fcnn_unit = 100, 200


class InertialNet(nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)       # !!!
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class OutNet(nn.Module):
    def __init__(self, input_dim, device, output_dim=6, high_level_locs=[2, 3, 4]):  # Changed output_dim to 6
        super(OutNet, self).__init__()
        self.high_level_locs = high_level_locs
        self.linear_1 = nn.Linear(input_dim + len(high_level_locs), globals()['fcnn_unit'], bias=True).to(device)
        self.linear_2 = nn.Linear(globals()['fcnn_unit'], output_dim, bias=True).to(device)
        self.relu = nn.ReLU().to(device)
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        if len(self.high_level_locs) > 0:
            sequence = torch.cat((sequence, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        height = others[:, 0, HEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence

import torch.nn.functional as F

class LmfImuOnlyNet(nn.Module):
    def __init__(self, acc_dim, gyr_dim):
        super(LmfImuOnlyNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.rank = 10
        self.fused_dim = 40
        self.device = torch.device('cpu')
        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim)).to(self.device)
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim)).to(self.device)
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank)).to(self.device)
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim)).to(self.device)

        # Added layers
        self.fc1 = nn.Linear(self.fused_dim, self.fused_dim // 2).to(self.device)
        self.dropout1 = nn.Dropout(0.3).to(self.device)
        self.fc2 = nn.Linear(self.fused_dim // 2, self.fused_dim // 4).to(self.device)
        self.dropout2 = nn.Dropout(0.3).to(self.device)

        self.out_net = OutNet(self.fused_dim // 4, self.device, output_dim=6)  # Set the output_dim value to 6
        self.fc_out = nn.Linear(self.fused_dim // 4, 6).to(self.device)  # Change the output dimension to 6
        # init factors
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def __str__(self):
        return 'LMF IMU only net'

    def set_scalars(self, scalars):
        self.scalars = scalars

    def set_fields(self, x_fields):
        self.acc_fields = x_fields['input_acc']
        self.gyr_fields = x_fields['input_gyr']

    def forward(self, acc_x, gyr_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).to(self.device).type(data_type), requires_grad=False), acc_h), dim=2)

        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).to(self.device).type(data_type), requires_grad=False), gyr_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_zy = fusion_acc * fusion_gyr
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias

        # Added layers
        sequence = F.relu(self.fc1(sequence))
        sequence = self.dropout1(sequence)
        sequence = F.relu(self.fc2(sequence))
        sequence = self.dropout2(sequence)

        sequence = self.out_net(sequence, others)
        return sequence

class GRFPredictor:
    def __init__(self, weight, height):
        self.data_buffer = deque(maxlen=MAX_BUFFER_LEN)
        self.data_margin_before_step = 20
        self.data_margin_after_step = 20
        self.data_array_fields = [axis + '_' + sensor for sensor in IMU_LIST for axis in IMU_FIELDS]
        base_path = os.path.abspath(os.path.dirname(__file__))
        model_state_path = base_path + '/models/7IMU_FUSION40_LSTM20.pth'
        device = torch.device('cpu')  # specify CPU device
        self.model = LmfImuOnlyNet(21, 21)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_state_path, map_location=device))  # specify map_location
        self.model.set_fields({'input_acc': ACC_ALL, 'input_gyr': GYR_ALL})
        scalar_path = base_path + '/models/scalars.pkl'
        self.model.set_scalars(pickle.load(open(scalar_path, 'rb')))
        self.model.acc_col_loc = [self.data_array_fields.index(field) for field in self.model.acc_fields]
        self.model.gyr_col_loc = [self.data_array_fields.index(field) for field in self.model.gyr_fields]

        self.weight = weight
        self.height = height

        anthro_data = np.zeros([1, 152, 5], dtype=np.float32)  # Change 2 to 5
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        anthro_data[:, :, HEIGHT_LOC] = self.height
        self.model_inputs = {'others': torch.from_numpy(anthro_data),
                             'input_acc': None, 'input_gyr': None}

    def update_stream(self, data):
        self.data_buffer.append([data, 0., 0., 0., 0., 0., 0.])
        print("Buffer length:", len(self.data_buffer))

        if len(self.data_buffer) >= self.data_margin_before_step + self.data_margin_after_step:
            inputs = self.transform_input(self.data_buffer, self.model_inputs)

            # Get the lens tensor
            lens = self.get_single_step_len(inputs['input_acc'], inputs['input_gyr'])
            # Pass the lens as an argument in the model's forward call
            pred = self.model(inputs['input_acc'], inputs['input_gyr'], inputs['others'], lens)
            pred = pred.detach().numpy().astype(np.float)[0]

            for i_sample in range(self.data_margin_before_step, len(self.data_buffer) - self.data_margin_after_step):
                self.data_buffer[i_sample][1:3] = [pred[i_sample - self.data_margin_before_step, 0], pred[i_sample - self.data_margin_before_step, 1]]

        if len(self.data_buffer) == MAX_BUFFER_LEN:
            result_data = self.data_buffer.popleft()
            results = [(result_data, grf) for grf in pred]
            return results

        return []



    def transform_input(self, data_buffer, model_inputs):
        raw_data = []
        for sample_data in list(data_buffer):
            raw_data_one_row = []
            for i_sensor in range(len(IMU_LIST)):
                raw_data_one_row.extend([sample_data[0][i_sensor][field] for field in IMU_FIELDS])
            raw_data.append(raw_data_one_row)
        data = np.array(raw_data, dtype=np.float32)
        data[:, self.model.acc_col_loc] = self.normalize_array_separately(
            data[:, self.model.acc_col_loc], self.model.scalars['input_acc'], 'transform')
        model_inputs['input_acc'] = torch.from_numpy(np.expand_dims(data[:, self.model.acc_col_loc], axis=0))
        data[:, self.model.gyr_col_loc] = self.normalize_array_separately(
            data[:, self.model.gyr_col_loc], self.model.scalars['input_gyr'], 'transform')
        model_inputs['input_gyr'] = torch.from_numpy(np.expand_dims(data[:, self.model.gyr_col_loc], axis=0))

        return model_inputs


    def get_single_step_len(self, input_acc, input_gyr, feature_col_num=0):
        data_acc_feature = input_acc[:, :, feature_col_num]
        data_gyr_feature = input_gyr[:, :, feature_col_num]
        zero_loc_acc = data_acc_feature == 0.
        zero_loc_gyr = data_gyr_feature == 0.
        data_len_acc = torch.sum(~zero_loc_acc, axis=1)
        data_len_gyr = torch.sum(~zero_loc_gyr, axis=1)

        # Take the maximum length between accelerometer and gyroscope data
        data_len = torch.max(data_len_acc, data_len_gyr)

        return data_len


    @staticmethod
    def normalize_array_separately(data, scalar, method, scalar_mode='by_each_column'):
        input_data = data.copy()
        original_shape = input_data.shape
        target_shape = [-1, input_data.shape[1]] if scalar_mode == 'by_each_column' else [-1, 1]
        input_data = input_data.reshape(target_shape)
        scaled_data = getattr(scalar, method)(input_data)
       
        scaled_data = scaled_data.reshape(original_shape)
        return scaled_data