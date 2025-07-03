import torch
import numpy as np
import itertools 
import torch.nn as nn
import torch
from torchvision import models
import torch.nn as nn
from collections import namedtuple
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.lpips_arch import NetLinLayer, default_model_urls, upsample, spatial_average, normalize_tensor, ScalingLayer


device = ("cuda" if torch.cuda.is_available() else "cpu")


class Config:
	def __init__(self):
		self.metric = 'vgg_ensemble'
		
		self.enable_dropout = True
		self.dropout_keep_prob = 0.99
		
		self.enable_offset = True
		self.offset_max = 7
		
		self.enable_flip = True
		self.enable_swap = True
		self.enable_color_permutation = True
		
		self.enable_color_multiplication = True
		self.color_multiplication_mode = 'color' # 'brightness'
		
		self.enable_scale = True
		self.set_scale_levels(8)
		
		# Enables cropping instead of padding. Faster but may randomly skip edges of the input.
		self.fast_and_approximate = False
		
		self.batch_size = 1 
		self.average_over = 1  # How many runs to average over.
	
		self.dtype = torch.float32
		
	def set_scale_levels(self, num_scales):
		# Crop_size / num_scales should be at least 64.
		self.num_scales = num_scales
		self.scale_probabilities = [1.0 / float(i)**2 for i in range(1, self.num_scales + 1)]
		
	def set_scale_levels_by_image_size(self, image_h, image_w):
		'''Sets the number of scale levels based on the image size.'''
		image_size = min(image_h, image_w)
		self.set_scale_levels(max(1, image_size // 64))
		
	def validate(self):
		assert self.metric in ('vgg_ensemble', 'vgg', 'squeeze', 'squeeze_ensemble_maxpool')
		assert self.color_multiplication_mode in ('color', 'brightness')
		assert self.num_scales == len(self.scale_probabilities)

def switch_case_cond(cases, default_case):
	if cases:
		condition, effect = cases[0]
		return effect() if condition else switch_case_cond(cases[1:], default_case)
	return default_case()

def switch_case_where(cases, default_case):
	if cases:
		condition, effect = cases[0]
		if condition:
			return effect
		return switch_case_where(cases[1:], default_case)
		#return np.where(condition, effect, switch_case_where(cases[1:], default_case))
	return default_case

def as_tuple(x):
	'''Formats x as a tuple. If x is already a tuple returns it as is, otherwise returns a 1-tuple containing x.'''
	if isinstance(x, tuple):
		return x
	else:
		return (x,)
		
def sample_ensemble(config):
	'''Samples a random transformation according to the config.
	   Uses Latin Hypercube Sampling when batch size is greater than 1.'''
	
	N = config.batch_size

	# Offset randomization.
	offset_xy = np.random.uniform(low=0, high=config.offset_max + 1, size=[N, 2]).astype(int)
			
	# Sample scale level.
	cumulative_sum = np.cumsum(config.scale_probabilities)
	u = cumulative_sum[-1] * np.random.uniform()
	
	scale_level = switch_case_cond(
		[(u < x, (lambda j=i: j+1)) for i, x in enumerate(cumulative_sum[:-1])],
		lambda: len(cumulative_sum)
	)
	scale_level = np.clip(scale_level, 1, config.num_scales)	
	
	
	# Scale randomization.
	scale_offset_xy = np.random.uniform(low=0, high=scale_level, size=[2]).astype(int)
	
	# Sample flips.
	flips = np.arange((N + 3)//4*4, dtype=np.int32)
	flips = flips % 4
	np.random.shuffle(flips)
	flips = flips[:N]
		
	# Sample transposing.
	swap_xy = np.random.binomial(1, 0.5, 1)

	# Color multiplication.
	def sample_colors():
		color = np.random.uniform(low=0.0, high=1.0, size=N)
		color += np.arange(N).astype(np.float32)
		color /= N
		np.random.shuffle(color)
		return color
	colors_r = np.reshape(sample_colors(), [-1, 1, 1, 1])
	colors_g = np.reshape(sample_colors(), [-1, 1, 1, 1])
	colors_b = np.reshape(sample_colors(), [-1, 1, 1, 1])
	
	if config.color_multiplication_mode == 'color':
		color_factors = np.concatenate([colors_r, colors_g, colors_b], axis=3)
	elif config.color_multiplication_mode == 'brightness':
		color_factors = np.concatenate([colors_r, colors_r, colors_r], axis=3)
	else:
		raise Exception('Unknown color multiplication mode.')
	
	color_factors = 0.2 + 0.8 * color_factors
	
	# Sample permutations.
	permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int32)
	repeat_count = (N + len(permutations) - 1) // len(permutations)
	permutations = np.tile(permutations, [repeat_count, 1])
	np.random.shuffle(permutations)
	permutations = permutations[:N, :].ravel()
			
	base_indices = 3 * np.reshape(np.tile(np.reshape(np.arange(N), [-1, 1]), [1, 3]), [-1]) # [0, 0, 0, 3, 3, 3, 6, 6, 6, ...]
	permutations += base_indices
						
	return (torch.Tensor(offset_xy), torch.Tensor(flips), torch.Tensor(swap_xy), torch.Tensor(
		color_factors), torch.Tensor(permutations), torch.Tensor(scale_offset_xy), torch.Tensor(
			[scale_level]))

def apply_ensemble(config, sampled_ensemble_params, X):
	'''Applies the sampled random transformation to image X.'''
	offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level = sampled_ensemble_params
	
	scale_offset_xy = scale_offset_xy.int()
	shape = X.shape
	N, C, H, W = shape[0], shape[1], shape[2], shape[3]

	# Resize image.
	if config.enable_scale:		
		def downscale_nx_impl(image, scale):
			shape = image.shape
			N, C, H, W = shape[0], shape[1], shape[2], shape[3]
			image = torch.reshape(image, [N, C, H//int(scale), int(scale), W//int(scale), int(scale)])
			image = torch.mean(image, axis=[3, 5])
			return image
			
		def downscale_1x():
			return X
		
		def downscale_nx():
			nonlocal X

			if config.fast_and_approximate:
				# Crop to a multiple of scale_level.
				crop_left = scale_offset_xy[1]
				full_width = (W - scale_level + 1) // scale_level * scale_level
				crop_right = crop_left + full_width
			
				crop_bottom = scale_offset_xy[0]
				full_height = (H - scale_level + 1) // scale_level * scale_level
				crop_top = crop_bottom + full_height
				
				X = X[..., crop_bottom:crop_top, crop_left:crop_right]
			else:
				# Pad to a multiple of scale_level.
				pad_left = scale_offset_xy[1]
				full_width = (scale_level - 1 + W + scale_level - 1) // scale_level * scale_level
				pad_right = full_width - W - pad_left
			
				pad_bottom = scale_offset_xy[0]
				full_height = (scale_level - 1 + H + scale_level - 1) // scale_level * scale_level
				pad_top = full_height - H - pad_bottom
				#X = torch.nn.ReflectionPad2d((int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)))(X.float())
				X = torch.nn.functional.pad(X, (int(pad_left), int(pad_right), int(pad_bottom), int(pad_top)), 'reflect')
			return downscale_nx_impl(X, scale_level)
		
		X = downscale_1x() if scale_level.item() == 1 else downscale_nx()
	# Pad image.
	if config.enable_offset:
		L = []

		shape = X.shape
		N, C, H, W = shape[0], shape[1], shape[2], shape[3]

		for i in range(config.batch_size):
			if config.fast_and_approximate:
				# Crop.
				crop_bottom = offset_xy[i, 0]
				crop_left = offset_xy[i, 1]
				crop_top = H - config.offset_max + crop_bottom
				crop_right = W - config.offset_max + crop_left
			
				L.append(X[i, :, crop_bottom:crop_top, crop_left:crop_right])
			else:
				# Pad.
				pad_bottom = config.offset_max - offset_xy[i, 0]
				pad_left = config.offset_max - offset_xy[i, 1]
				pad_top = offset_xy[i, 0]
				pad_right = offset_xy[i, 1]
				
				L.append(torch.nn.functional.pad(X[i,:,:,:][None, :], (int(pad_left), int(pad_right), int(pad_bottom), int(pad_top)), 'reflect')[0])
		X = torch.stack(L, axis=0)
	# Apply flips.	
	if config.enable_flip:
		def flipX(X):
			return torch.flip(X, (-1, ))
		def flipY(X):
			return torch.flip(X, (-2, ))
		def flipXandY(X):
			return torch.flip(X, (-2, -1))
		X = switch_case_where(
			[((flips.item() == 0), flipX(X)),
			((flips.item() == 1), flipY(X)),
			((flips.item() == 2), flipXandY(X))],
			X
		)
	# Apply transpose.
	if config.enable_swap:
		def swapXY(X):
			return torch.permute(X, (0, 1, 3, 2))
		X = swapXY(X) if swap_xy == 1 else X
				
	# Apply color permutations.
	if config.enable_color_permutation:
		def permuteColor(X, perms):
			shape = X.shape
			N, C, H, W = shape[0], shape[1], shape[2], shape[3]
			#X = np.transpose(X, [0, 3, 1, 2]) # NHWC -> NCHW
			X = X.reshape([N * C, H, W])  # (NC)HW
			perms = torch.tile(perms.view(-1, 1, 1), (1, X.shape[1], X.shape[2]))
			X = torch.gather(X, 0, perms.long().to(device))           # Permute rows (colors)
			X = X.reshape([N, C, H, W])   # NCHW
			#X = np.transpose(X, [0, 2, 3, 1]) # NCHW -> NHWC
			return X

		X = permuteColor(X, permutations)
	
	if config.enable_color_multiplication:
		X = X * color_factors.reshape([config.batch_size, 3, 1, 1]).to(device)

	return X

def elpips_vgg(batch_size=1, n=1, dtype=np.float32):
	'''E-LPIPS-VGG configuration with all input transformations and dropout with p=0.99. Returns the average result over n samples.
	Warning: Some versions of TensorFlow might have bugs that make n > 1 problematic due to the tf.while_loop used internally when n > 1.'''
	config = Config()
	config.metric = 'vgg_ensemble'
	config.batch_size = batch_size
	config.average_over = n
	config.dtype = dtype
			
	return config

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        mask = (torch.rand(input.shape[1:]) < self.p).to(input.device) / self.p
        return input*mask
    
class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        p = 0.99
        i = 1
        for x in range(4):
            if type(vgg_pretrained_features[x]) == nn.Conv2d:
                self.slice1.add_module(f'dropout{i}', Dropout(p=p))
                i += 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            if type(vgg_pretrained_features[x]) == nn.Conv2d:
                self.slice2.add_module(f'dropout{i}', Dropout(p=p))
                i += 1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            if type(vgg_pretrained_features[x]) == nn.Conv2d:
                self.slice3.add_module(f'dropout{i}', Dropout(p=p))
                i += 1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            if type(vgg_pretrained_features[x]) == nn.Conv2d:
                self.slice4.add_module(f'dropout{i}', Dropout(p=p))
                i += 1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            if type(vgg_pretrained_features[x]) == nn.Conv2d:
                self.slice5.add_module(f'dropout{i}', Dropout(p=p))
                i += 1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.slice2[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.slice3[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.slice4[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.slice5[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_slice1(self, input):
        o1 = input
        o2 = self.slice1[:3](o1)
        o3 = self.slice1[3:](o2)
        return o1, o2, o3
    
    def get_slice2(self, input):
        o1 = self.slice2[:4](input)
        o2 = self.slice2[4:](o1)
        return o1, o2
    
    def get_slice3(self, input):
        o1 = self.slice3[:4](input)
        o2 = self.slice3[4:7](o1)
        o3 = self.slice3[7:](o2)
        return o1, o2, o3
    
    def get_slice4(self, input):
        o1 = self.slice4[:4](input)
        o2 = self.slice4[4:7](o1)
        o3 = self.slice4[7:](o2)
        return o1, o2, o3
    
    def get_slice5(self, input):
        o1 = self.slice5[:4](input)
        o2 = self.slice5[4:7](o1)
        o3 = self.slice5[7:](o2)
        return o1, o2, o3

    def forward(self, X):
        o11, o12, o13 = self.get_slice1(X)
        o21, o22 = self.get_slice2(o13)
        o31, o32, o33 = self.get_slice3(o22)
        o41, o42, o43 = self.get_slice4(o33)
        o51, o52, o53 = self.get_slice5(o43)
        
        vgg_outputs = namedtuple("VggOutputs", ['o11', 'o12', 'o13', 'o21', 'o22', 'o31', 'o32', 'o33', 'o41', 'o42', 'o43', 'o51', 'o52', 'o53'])
        out = vgg_outputs(o11, o12, o13, o21, o22, o31, o32, o33, o41, o42, o43, o51, o52, o53)

        return out

class LPIPS(nn.Module):
    def __init__(self,
                 pretrained=True,
                 net='vgg',
                 version='0.1',
                 lpips=True,
                 spatial=False,
                 pnet_rand=False,
                 pnet_tune=False,
                 use_dropout=True,
                 pretrained_model_path='elpips.pth',
                 eval_mode=True,
                 **kwargs):

        super(LPIPS, self).__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.chns = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.L = len(self.chns)

        self.net = vgg16(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if (lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lin7 = NetLinLayer(self.chns[7], use_dropout=use_dropout)
            self.lin8 = NetLinLayer(self.chns[8], use_dropout=use_dropout)
            self.lin9 = NetLinLayer(self.chns[9], use_dropout=use_dropout)
            self.lin10 = NetLinLayer(self.chns[10], use_dropout=use_dropout)
            self.lin11 = NetLinLayer(self.chns[11], use_dropout=use_dropout)
            self.lin12 = NetLinLayer(self.chns[12], use_dropout=use_dropout)
            self.lin13 = NetLinLayer(self.chns[13], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4, 
                         self.lin5, self.lin6, self.lin7, self.lin8, self.lin9,
                         self.lin10, self.lin11, self.lin12, self.lin13]
            self.lins = nn.ModuleList(self.lins)

            if pretrained_model_path is not None:
                load_pretrained_network(self, pretrained_model_path, False)
            elif pretrained:
                load_pretrained_network(self, default_model_urls[f'{version}_{net}'], False)

        if (eval_mode):
            self.eval()

    def forward(self, in1, in0, retPerLayer=False, normalize=True):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        
        out = self.net.forward(torch.concat([in1_input, in0_input]))
        
        res = []
        val = 0
        for kk in range(self.L):
            diffs = (normalize_tensor(out[kk][0][None, :]) - normalize_tensor(out[kk][1][None, :]))**2
            val += spatial_average(self.lins[kk](diffs), keepdim=True)

        if (retPerLayer):
            return (val, res)
        else:
            return val.squeeze()

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path='src/elpips'):
        super().__init__()
        self.model = LPIPS(pretrained_model_path=model_path).to(device)
        self.device = device
        self.lower_better = True
        self.full_reference = True
        self.model.eval()
        #self.n = 200
        self.n = 3
        self.BATCH_SIZE = 10
        self.max_relative_error = 0.025
        self.max_absolute_error = 0.01

    def forward(self, dist, ref):
        scores = torch.empty(0).to(device)
        dist = dist.to(device)
        ref = ref.to(device)
        config = elpips_vgg()
        for i in range(1, self.n + 1):
            sampled_ensemble_params = sample_ensemble(config)
            dist_ans = apply_ensemble(config, sampled_ensemble_params, dist)
            ref_ans = apply_ensemble(config, sampled_ensemble_params, ref)
            score = self.model(dist_ans, ref_ans)
            scores = torch.concat([scores, score.ravel()])
            """
            if i % 10 == 0:
                mean = torch.mean(scores)
                stddev_of_mean = torch.std(scores) / np.sqrt(len(scores))
                #relative_bound = 1.96 * stddev_of_mean / (1e-12 + mean)
                #absolute_bound = 1.96 * stddev_of_mean
                relative_error_satistfied = 1.96 * stddev_of_mean < self.max_relative_error * mean
                absolute_error_satistfied = 1.96 * stddev_of_mean < self.max_absolute_error
                if relative_error_satistfied and absolute_error_satistfied:
                    break
                #print("   [Processed samples: {}.  Current estimate: {} +- {} ({:.4f}%)]".format(i, mean, absolute_bound, 100.0 * relative_bound))
			"""
        return torch.mean(scores)