import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from PIL import Image
import h5py
import openslide

from normalizers.base import BaseNormalizer
#from normalizers.reinhard import ReinhardNormalizer
#from normalizers.staingan import ResnetGenerator, norm, un_norm

#from tiatoolbox.tools.stainnorm import MacenkoNormalizer
#from torchvision.transforms.functional import pil_to_tensor, to_pil_image


class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		normalization_technique,
		normalizer_target,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.wsi = wsi
		self.roi_transforms = img_transforms
		self.normalization_technique = normalization_technique
		self.normalizer_target = normalizer_target
		self.file_path = file_path
		self.normalizer = BaseNormalizer.from_string(normalization_technique, normalizer_target)

		#self.model = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).cuda()
		#self.model.load_state_dict(torch.load("/shared/cgorner/SN_IDC_Grading/staingan_checkpoint.pth"))

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
		
        # Fit the normalizer if one was created
		if self.normalizer is not None and normalizer_target is not None:
			print("Fitting normalizer...")
			target_img = openslide.open_slide(self.normalizer_target)
			target_patch = target_img.read_region((int(98054/2), int(79408/4)), 0, (1024, 1024)).convert('RGB')
			self.normalizer.fit(target_patch)
			print("Normalizer fitted.")

		"""
		# macenko
		target_img = openslide.open_slide(self.normalizer_target)
		target_patch = target_img.read_region((int(98054/2),int(79408/4)),0, (1024,1024)).convert('RGB')
		target_patch = pil_to_tensor(target_patch).to(self.device)
		print("fitting normalizer")
		self.normalizer.fit(target_patch)
		"""

		"""
		# reinhard
		target_img = openslide.open_slide(self.normalizer_target)
		target_patch = np.array(target_img.read_region((int(98054/2),int(79408/4)),0, (1024,1024)).convert('RGB'))
		print("fitting normalizer")
		self.normalizer.fit(target_patch)
		"""

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		
		"""
		#Staingan
		img = Image.fromarray(un_norm(self.model(norm(img).cuda())))
		"""

		"""
		# Macenko
		img = pil_to_tensor(img).to(self.device)
		try:
			img, _, _ = self.normalizer.normalize(img, stains=False)
		except Exception as e:
			print(f"Exception occurred while normalizing patch: {str(e)}")
			img = img.permute(*torch.arange(img.ndim - 1, -1, -1))

		#print(f"img shape: {img.shape}")
		#print("#5")
		#img = to_pil_image(img.T.to(torch.uint8))
		img = to_pil_image(img.permute(*torch.arange(img.ndim - 1, -1, -1)).to(torch.uint8))
		"""

		"""
		# Reinhard
		img = np.array(img)

		if self.normalizer:
			img = self.normalizer.transform(img)
		
		img = Image.fromarray(np.uint8(img))
		"""

        # Apply normalization if a normalizer is available
		if self.normalizer is not None:
			try:
				img = self.normalizer.transform(img)
			except Exception as e:
				print(f"Exception occurred while normalizing patch: {str(e)}")

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




