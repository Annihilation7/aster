import os
import random
import numpy as np
import torch
from torchvision import transforms

from torch.utils import data
import cv2

class MyDataset(data.Dataset):
	def __init__(
			self, root_dirs, charset_path, is_replace_ch, is_train,
			max_len, transform, is_gray
	):
		self.max_len = max_len
		self.transform = transform
		self.is_gray = is_gray
		self.data_list = []
		self.root_dirs = root_dirs.split(',')

		with open(charset_path, 'r', encoding='utf-8') as fp:
			self.charset = list(fp.read())
		self.EOS = 'EOS'
		self.PADDING = 'PADDING'
		self.UNKNOWN = 'UNKNOWN'
		self.charset.extend([self.EOS, self.PADDING, self.UNKNOWN])
		self.char2id = dict(zip(self.charset, range(len(self.charset))))
		self.id2char = dict(zip(range(len(self.charset)), self.charset))
		self.rec_num_classes = len(self.charset)

		self.is_replace_ch = is_replace_ch
		self.is_train = is_train

		self.re_table = {ord(f): ord(t) for f, t in zip(
			'，！：（）；—“”‘’～',
			',!:();-""\'\'~'
		)}

		for root_dir in self.root_dirs:
			if not root_dir:
				continue
			label_txt = os.path.join(root_dir, 'label.txt')
			img_path_list, label_list = self._load_file_data(label_txt, root_dir)
			for i, img_path in enumerate(img_path_list):
				self.data_list.append((img_path, label_list[i]))
		if self.is_train:
			random.shuffle(self.data_list)

	def _load_file_data(self, file_path, root_dr):
		img_path_list = []
		label_list = []

		with open(file_path, 'r', encoding='utf-8') as fp:
			for line in fp:
				line = line.strip('\n')
				p = line.find(' ')
				img_name = line[:p]
				img_path = os.path.join(root_dr, img_name)
				label_text = line[p + 1:]
				if self.is_replace_ch:
					label_text = label_text.translate(self.re_table)
				img_path_list.append(img_path)
				label_list.append(label_text)

		return img_path_list, label_list

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, index):
		img_path, text = self.data_list[index]
		img = cv2.imread(img_path)  # bgr
		if img is None:
			return self[index + 1]
		label = np.full((self.max_len), self.char2id[self.PADDING], dtype=np.int32)
		label_list = []
		for char in text:
			if char in self.char2id:
				label_list.append(self.char2id[char])
			else:
				label_list.append(self.char2id[self.UNKNOWN])
		label_list.append(self.char2id[self.EOS])
		assert len(label_list) <= self.max_len
		label[:len(label_list)] = np.array(label_list)
		label_len = len(label_list)
		if self.transform is not None:
			img = self.transform(img)
		if self.is_gray:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, label, label_len


class AlignCollate:
	def __init__(self, imgH=48, imgW=800, is_gray=False):
		self.imgH = imgH
		self.imgW = imgW
		self.is_gray = is_gray
		if not self.is_gray:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.406, 0.456, 0.485], std=[0.224, 0.224, 0.229]
				)
			])
		else:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5], std=[0.5])
			])

	def __call__(self, batch):
		images, labels, lengths = zip(*batch)
		b_labels = torch.IntTensor(labels)
		b_lengths = torch.IntTensor(lengths)

		max_w = -1
		images_re = []

		for image in images:
			h, w = image.shape[:2]
			ratio = self.imgH / h
			if int(w * ratio) > self.imgW:
				ratio = self.imgW / w
			image_re = cv2.resize(
				image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
			)
			max_w = max(max_w, image_re.shape[1])

		for image_re in images_re:
			h_re, w_re = image_re.shape[:2]
			if not self.is_gray:
				canvas = np.zeros(shape=[self.imgH, max_w, 3], dtype=np.uint8)
			else:
				canvas = np.zeros(shape=[self.imgH, max_w], dtype=np.uint8)
			canvas[: h_re, : w_re] = image_re





if __name__ == '__main__':
	test = MyDataset(
		'/data/mazhenyu/Code_2rd/crnn/crnn/testdata/test1,/data/mazhenyu/Code_2rd/crnn/crnn/testdata/test2',
		'/data/mazhenyu/Code_2rd/aster/lib/datasets/doc_charset.txt',
		True, True, 100, None
	)
	print(test.data_list)
	print(test[0])
	print(test[1])

