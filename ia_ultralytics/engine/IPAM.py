import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import math

#-----Filter相关的基础函数------
def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]
def tanh01(x):
  # return tf.tanh(x) * 0.5 + 0.5
  return torch.tanh(x) * 0.5 + 0.5
def tanh_range(l, r, initial=None):
  def get_activation(left, right, initial):
    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left
    return activation
  return get_activation(l, r, initial)
def lerp(a, b, l):
  return (1 - l) * a + l * b
    
#-----Filter的相关实现------
class Filter(nn.Module):

  def __init__(self, net, cfg):
    super(Filter, self).__init__()

    self.cfg = cfg
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features):
    return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
           features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param, defog, IcA):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

  # Apply the whole filter with masking
  def apply(self,
            img,
            img_features=None,
            defog_A=None,
            IcA=None,
            specified_parameter=None,
            high_res=None):
    assert (img_features is None) ^ (specified_parameter is None)
    if img_features is not None:
      filter_features, mask_parameters = self.extract_parameters(img_features)
      filter_parameters = self.filter_param_regressor(filter_features)
    else:
      assert not self.use_masking()
      filter_parameters = specified_parameter

    if high_res is not None:
      # working on high res...
      pass
    debug_info = {}
    # We only debug the first image of this batch
    if self.debug_info_batched():
      debug_info['filter_parameters'] = filter_parameters
    else:
      debug_info['filter_parameters'] = filter_parameters[0]
    # self.mask_parameters = mask_parameters
    # self.mask = self.get_mask(img, mask_parameters)
    # debug_info['mask'] = self.mask[0]
    #low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
    low_res_output = self.process(img, filter_parameters, defog_A, IcA)

    if high_res is not None:
      if self.no_high_res():
        high_res_output = high_res
      else:
        self.high_res_mask = self.get_mask(high_res, mask_parameters)
        # high_res_output = lerp(high_res,
        #                        self.process(high_res, filter_parameters, defog, IcA),
        #                        self.high_res_mask)
    else:
      high_res_output = None
    #return low_res_output, high_res_output, debug_info
    return low_res_output, filter_parameters

  def use_masking(self):
    return self.cfg.masking

  def get_num_mask_parameters(self):
    return 6

  # Input: no need for tanh or sigmoid
  # Closer to 1 values are applied by filter more strongly
  # no additional TF variables inside
  def get_mask(self, img, mask_parameters):
    if not self.use_masking():
      print('* Masking Disabled')
      return torch.ones(shape=(1, 1, 1, 1), dtype=torch.float32)
    else:
      print('* Masking Enabled')
    with tf.name_scope(name='mask'):
      # Six parameters for one filter
      filter_input_range = 5
      assert mask_parameters.shape[1] == self.get_num_mask_parameters()
      mask_parameters = tanh_range(
          l=-filter_input_range, r=filter_input_range,
          initial=0)(mask_parameters)
      size = list(map(int, img.shape[1:3]))
      grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

      shorter_edge = min(size[0], size[1])
      for i in range(size[0]):
        for j in range(size[1]):
          grid[0, i, j,
               0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
          grid[0, i, j,
               1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
      grid = tf.constant(grid)
      # Ax + By + C * L + D
      inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
            grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
            mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
            mask_parameters[:, None, None, 3, None] * 2
      # Sharpness and inversion
      inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
                                                          None] / filter_input_range
      mask = tf.sigmoid(inp)
      # Strength
      mask = mask * (
          mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
          0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
      print('mask', mask.shape)
    return mask

  # def visualize_filter(self, debug_info, canvas):
  #   # Visualize only the filter information
  #   assert False

  def visualize_mask(self, debug_info, res):
    return cv2.resize(
        debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
        dsize=res,
        interpolation=cv2.cv2.INTER_NEAREST)

  def draw_high_res_text(self, text, canvas):
    cv2.putText(
        canvas,
        text, (30, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0),
        thickness=5)
    return canvas


class ExposureFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'E'
    self.begin_filter_parameter = cfg.exposure_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):#param is in (-self.cfg.exposure_range, self.cfg.exposure_range)
    return tanh_range(
        -self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

  def process(self, img, param, defog, IcA):
    return img * torch.exp(param * np.log(2))


class UsmFilter(Filter):#Usm_param is in [Defog_range]

  def __init__(self, net, cfg):

    Filter.__init__(self, net, cfg)
    self.short_name = 'UF'
    self.begin_filter_parameter = cfg.usm_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.usm_range)(features)

  def process(self, img, param, defog_A, IcA):


    self.channels = 3
    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = np.repeat(kernel, self.channels, axis=0)
    if torch.float16==img.dtype:
        kernel=kernel.half()

    # print('      param:', param)

    kernel = kernel.to(img.device)
    # self.weight = nn.Parameter(data=kernel, requires_grad=False)
    # self.weight.to(device)

    output = F.conv2d(img, kernel, padding=2, groups=self.channels)


    img_out = (img - output) * param + img
    # img_out = (img - output) * torch.tensor(0.043).cuda() + img

    return img_out

class ContrastFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'Ct'
    self.begin_filter_parameter = cfg.contrast_begin_param

    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.cont_range)(features)

  def process(self, img, param, defog, IcA):
    # print('      param.shape:', param.shape)

    # luminance = torch.minimum(torch.maximum(rgb2lum(img), 0.0), 1.0)
    luminance = rgb2lum(img)
    zero = torch.zeros_like(luminance)
    one = torch.ones_like(luminance)

    luminance = torch.where(luminance < 0, zero, luminance)
    luminance = torch.where(luminance > 1, one, luminance)

    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param)
    # return lerp(img, contrast_image, torch.tensor(0.015).cuda())


class ToneFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.short_name = 'T'
    self.begin_filter_parameter = cfg.tone_begin_param

    self.num_filter_parameters = cfg.curve_steps

  def filter_param_regressor(self, features):
    tone_curve = tanh_range(*self.cfg.tone_curve_range)(features)
    return tone_curve

  def process(self, img, param, defog, IcA):
    param = torch.unsqueeze(param, 3)
    # print('      param.shape:', param.shape)

    tone_curve = param
    tone_curve_sum = torch.sum(tone_curve, axis=1) + 1e-30
    # print('      tone_curve_sum.shape:', tone_curve_sum.shape)

    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                     * param[:, i, :, :]
    total_image *= self.cfg.curve_steps / tone_curve_sum
    img = total_image
    return img

class GammaFilter(Filter):  #gamma_param is in [1/gamma_range, gamma_range]

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'G'
    self.begin_filter_parameter = cfg.gamma_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg.gamma_range)
    # return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))
    return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param, defog_A, IcA):
    # print('      param:', param)

    # param_1 = param.repeat(1, 3)
    zero = torch.zeros_like(img) + 0.00001
    img = torch.where(img <= 0, zero, img)
    # print("GAMMMA", param)
    return torch.pow(img, param)
      
#----------Filter模块的参数------------
from easydict import EasyDict as edict
cfg=edict()
cfg.num_filter_parameters = 4
#这里的配置均被用于DIF模块的滤波操作
cfg.exposure_begin_param = 0
cfg.gamma_begin_param = 1
cfg.contrast_begin_param = 2
cfg.usm_begin_param = 3
# Gamma = 1/x ~ x
cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)
cfg.cont_range = (0.0, 1.0)

#----------DIF模块------------
class DIF(nn.Module):
    def __init__(self, Filters=[ExposureFilter, GammaFilter, ContrastFilter, UsmFilter]):
        super(DIF, self).__init__()
        self.Filters=Filters
    def forward(self, img_input,Pr):
        self.filtered_image_batch = img_input
        filters = [x(img_input, cfg) for x in self.Filters]
        self.filter_parameters = []
        self.filtered_images = []
        for j, filter in enumerate(filters):
            self.filtered_image_batch, filter_parameter = filter.apply(
                self.filtered_image_batch, Pr)
            self.filter_parameters.append(filter_parameter)
            self.filtered_images.append(self.filtered_image_batch)
        return self.filtered_image_batch, self.filtered_images, Pr, self.filter_parameters    
    def __deepcopy__(self,xxx):
        newm=DIF()
        return newm
#----------IPAM模块------------
def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers
class IPAM(nn.Module):
    def __init__(self):
        super(IPAM, self).__init__()
        
        self.CNN_PP = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True),
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, cfg.num_filter_parameters, 8, padding=0),
        )
        self.dif=DIF()

    def forward(self, img_input):
        self.Pr = self.CNN_PP(img_input)
        out = self.dif(img_input,self.Pr)
        self.filtered_image_batch, self.filtered_images, self.Pr, self.filter_parameters =out
        return self.filtered_image_batch
    
    def __deepcopy__(self,xxx):
        newm=IPAM()
        newm.CNN_PP=copy.deepcopy(self.CNN_PP) 
        newm.dif=DIF()
        return newm
import os
import copy
class IA_Model(nn.Module):
    def __init__(self,oldconv):
        super(IA_Model, self).__init__()
        self.oldconv=copy.deepcopy(oldconv) 
        self.f=oldconv.f
        self.i=oldconv.i
        self.ipam=IPAM()
    def forward(self,x):
        x=self.ipam(x)
        x=self.oldconv(x)
        return x