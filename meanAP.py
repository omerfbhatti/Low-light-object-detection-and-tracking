import numpy as np
import sys 
from collections import defaultdict

sys.path.append('/usr/local/opencv-4.5.2/lib/python3.8/dist-packages/')
import cv2
import os 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision
from sklearn.metrics.pairwise import cosine_similarity

from model import Net
from triplet_anchor_pair_batch_dataset import TripletPairBatchDataset

class meanap:
	def __init__(self, net, dataset, device):
		self.dataset = dataset
		self.net = net
		self.device = device
		self.tracks = {}
		self.anchors = {}
		self.results = defaultdict(list)
		self.distMatrix = None
		#self.results['anchor'] = []
		#self.results['track'] = []
		#self.results['distance'] = []
		self.load_embeddings()
		self.gc_ov = self.calc_average_camera_feature('ov')
		self.gc_hd = self.calc_average_camera_feature('hd')
		self.alpha = 1
		self.beta = 0.18
		
	def load_embeddings(self):
		with torch.no_grad():
			for tracklet_index in range(len(self.dataset.tracklets)):
				
				index = self.dataset.tracklets[tracklet_index]['id']
				anchors, positives, _ = self.dataset.load_tracklet(tracklet_index)
				
				anchor_outputs = self.batched_propagate(self.net, self.device, anchors)
				self.anchors[tracklet_index] = anchor_outputs
				
				positive_outputs = self.batched_propagate(self.net, self.device, positives)
				self.tracks[tracklet_index] = positive_outputs

				
	def calc_average_camera_feature(self, camera):
		average_tensors = torch.zeros((len(self.dataset.tracklets),128,1,1), device='cuda')
		for track_id in range(len(self.dataset.tracklets)):
			if camera == 'hd':
				average_tensors[track_id,:,:,:] = self.weighted_embedding( self.tracks[track_id] )
			elif camera == 'ov':
				average_tensors[track_id,:,:,:] = self.weighted_embedding( self.anchors[track_id] )
		gc = self.weighted_embedding( average_tensors )
		return gc
			
	def get_image_feature(self, image_embedding, camera):#, w_embedding):
		if camera=='ov':
			image_embedding = image_embedding - self.alpha*self.gc_ov
		elif camera=='hd':
			image_embedding = image_embedding - self.alpha*self.gc_hd
		#image_embedding = self.beta*image_embedding + (1-self.beta)*w_embedding
		return image_embedding

	def eval_track(self, tracklet_index):
		distances = {}
		n = self.tracks[tracklet_index].shape[0]
		prime_anchor = np.random.choice(n)
		prime_embedding = self.anchors[tracklet_index][prime_anchor]
		prime_embedding = self.get_image_feature(prime_embedding, 'ov')
		#w_embedding = self.tracks[tracklet_index]
		
		#cos = torch.nn.CosineSimilarity(dim=2)
		print("Prime embedding selected from track ", tracklet_index)
		#print("prime embedding: ", prime_embedding.shape)
		#print("w_embedding: ", w_embedding.shape)
		#print("track", tracklet_index, "distance: ", distances[tracklet_index])
		for track_id in range(len(self.dataset.tracklets)):
			#if track_id != tracklet_index:
			w_embedding = self.weighted_embedding( self.tracks[track_id] - self.gc_hd )
			distances[track_id] = torch.norm( torch.cdist(prime_embedding, w_embedding) ).item()
			#distances[track_id] = cos(prime_embedding, w_embedding).item()
			#print("track", track_id, "distance: ", distances[track_id])
				
		self.saveplot(tracklet_index, distances)
		del distances
		
	def get_distMatrix(self):
		for track_id in range(len(self.dataset.tracklets)):
			self.eval_track(track_id)
		self.cmatrix(save = False)
	
	def saveplot(self, tracklet_index, distances):
				
		dist = {}
		#print(distances.keys())
		#print(distances.values())
		#myKeys = list(distances.keys())
		#myKeys.sort()
		#distances = {i: distances[i] for i in myKeys}
		
		dist['distance'] = distances.values()
		dist['track'] = distances.keys()
		
		#self.results['anchor'] = list( np.ones(31)*tracklet_index )
		#self.results['track'] = distances.keys()
		#self.results['distance'] = distances.values()
		
		self.results['anchor'].extend( list( np.ones(31, dtype=np.int16)*tracklet_index ))
		self.results['track'].extend( distances.keys() )
		self.results['distance'].extend( distances.values() )
		
		df = pd.DataFrame.from_dict(dist)
		#print(df)
		#plt.figure()
		#plots = sns.barplot(data=df,  x="track", y="distance")
		#plots.text(0,1.5, f"anchor image chosen from track {tracklet_index}", fontsize=12, color="red")
		#plt.show()
		#plots = sns.barplot(  x=distances.keys(), y=distances.values() )
		#plots.figure.savefig(f"./results/track {tracklet_index}.png")
		#plt.close()
		#del plots
		del dist
	
	def cmatrix(self, save = False):
				
		#print(self.results)
		df = pd.DataFrame.from_dict(self.results)
		results = pd.pivot_table(data=df,index='anchor',columns='track',values='distance')
		print(results)
		print(results['0'])
		self.distMatrix = np.array(results)
		#print(self.distMatrix)
		
		plt.figure()
		fig, ax = plt.subplots(figsize=(20, 20))
		matrix = sns.heatmap(results, cmap='coolwarm', square=True, annot=True)
		plt.show()
		if save:
			matrix.figure.savefig(f"./results/matrix.png")
		plt.close()
		
	def rerank(self):
		pass
		
	
	def weighted_embedding(self, w_embedding):
		return torch.mean(w_embedding, 0)
		
				
	def batched_propagate(self, net, device, inputs):
		batch_size = 64 #256
		num_batches = inputs.shape[0] // batch_size
		if inputs.shape[0] % batch_size != 0:
			num_batches += 1
		outputs = torch.tensor([]).to(device)
		for batch_i in range(num_batches):
			batch = inputs[(batch_i * batch_size):((batch_i+1) * batch_size)]
			batch = batch.to(device)
			batch_outputs = net(batch)
			del batch
			outputs = torch.cat((outputs, batch_outputs), 0)
		assert outputs.shape[0] == inputs.shape[0]
		assert outputs.shape[1] == 128 #256
		return outputs

