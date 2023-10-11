import glob
import numpy as np
import torch
import os
import tqdm
import re
import pickle
import random
import cv2


class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, type_="train", EMBEDDING_SOURCE=None, RESULTS_PATH=None):
        self.type_ = type_
        self.EMBEDDING_SOURCE = EMBEDDING_SOURCE
        self.RESULTS_PATH = RESULTS_PATH
        
        all_files = list(set([re.findall(r"(.+)_(\d+)_(.+)_(\d+)_(\d+)_.+", filename)[0] for filename in os.listdir(f"{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training")]))
        if self.type_ == "train":
            all_files = all_files[:int((2/3) * len(all_files))]
        else:
            all_files = all_files[int((2/3) * len(all_files)):]
        
        self.final_database = []
        for person1_name, triple_to_reconstruct_index, person2_name, triple_to_change_input_index, triple_to_change_output_index in all_files:
            triple_to_reconstruct_index = int(triple_to_reconstruct_index)
            triple_to_change_input_index = int(triple_to_change_input_index)
            triple_to_change_output_index = int(triple_to_change_output_index)
            self.final_database.append((person1_name, triple_to_reconstruct_index, person2_name, triple_to_change_input_index, triple_to_change_output_index))
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_database_training.pickle', 'rb') as handle:
            self.database = pickle.load(handle)
        
        ids = list(self.database.keys())
        self.ids_dict = {id:idx for idx, id in enumerate(list(self.database.keys()))}
        self.emotions_dict = {emo:idx for idx, emo in enumerate(np.unique([triplet[0] for id in ids for triplet in self.database[id]]).tolist())}
        self.poses_dict = {pose:idx for idx, pose in enumerate(np.unique([triplet[1] for id in ids for triplet in self.database[id]]).tolist())}

    def __len__(self):
        return len(self.final_database)
        
    def __getitem__(self, idx):
        tup = self.final_database[idx]
        
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_pose_img_to_reconstruct.npy', 'rb') as f:
            pose_img_to_reconstruct = np.load(f)
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_input.npy', 'rb') as f:
            embedding_input = np.load(f)
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_output.npy', 'rb') as f:
            embedding_output = np.load(f)
            
        expected_pose_label = self.database[tup[2]][tup[4]][1]
        input_id = tup[2]
        input_emo = self.database[tup[2]][tup[3]][0]
        
        # negative id
        available_negative_choices = list(filter(lambda item: item[2] != input_id, self.final_database))
        neg_tup = random.choice(available_negative_choices)
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{neg_tup[0]}_{neg_tup[1]}_{neg_tup[2]}_{neg_tup[3]}_{neg_tup[4]}_embedding_input.npy', 'rb') as f:
            negative_id_embedding_input = np.load(f)
            
        # negative pose
        available_negative_choices = list(filter(lambda item: item[2] == input_id and self.database[item[2]][item[4]][1] != expected_pose_label, self.final_database))    
        neg_tup = random.choice(available_negative_choices)
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{neg_tup[0]}_{neg_tup[1]}_{neg_tup[2]}_{neg_tup[3]}_{neg_tup[4]}_embedding_input.npy', 'rb') as f:
            negative_pose_embedding_input = np.load(f)
        
        # negative emotion
        available_negative_choices = list(filter(lambda item: item[2] == input_id and self.database[item[2]][item[4]][0] != input_emo, self.final_database))
        neg_tup = random.choice(available_negative_choices)
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/training/{neg_tup[0]}_{neg_tup[1]}_{neg_tup[2]}_{neg_tup[3]}_{neg_tup[4]}_embedding_input.npy', 'rb') as f:
            negative_emo_embedding_input = np.load(f)

        return pose_img_to_reconstruct, embedding_input, embedding_output, negative_id_embedding_input, negative_pose_embedding_input, negative_emo_embedding_input


class CustomDataSetEvaluation(torch.utils.data.Dataset):

    def __init__(self, EMBEDDING_SOURCE=None, RESULTS_PATH=None): 
        self.EMBEDDING_SOURCE = EMBEDDING_SOURCE
        self.RESULTS_PATH = RESULTS_PATH
        
        all_files = list(set([re.findall(r"(.+)_(\d+)_(.+)_(\d+)_(\d+)_.+", filename)[0] for filename in os.listdir(f"{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/evaluation")]))
        
        self.final_database = []
        for person1_name, triple_to_reconstruct_index, person2_name, triple_to_change_input_index, triple_to_change_output_index in all_files:
            triple_to_reconstruct_index = int(triple_to_reconstruct_index)
            triple_to_change_input_index = int(triple_to_change_input_index)
            triple_to_change_output_index = int(triple_to_change_output_index)
            self.final_database.append((person1_name, triple_to_reconstruct_index, person2_name, triple_to_change_input_index, triple_to_change_output_index))
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_database_evaluation.pickle', 'rb') as handle:
            self.database = pickle.load(handle)
        
        ids = list(self.database.keys())
        self.ids_dict = {id:idx for idx, id in enumerate(list(self.database.keys()))}
        self.emotions_dict = {emo:idx for idx, emo in enumerate(np.unique([triplet[0] for id in ids for triplet in self.database[id]]).tolist())}
        self.poses_dict = {pose:idx for idx, pose in enumerate(np.unique([triplet[1] for id in ids for triplet in self.database[id]]).tolist())}

    def __len__(self):
        return len(self.final_database)
        
    def __getitem__(self, idx):        
        tup = self.final_database[idx]
        
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/evaluation/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_pose_img_to_reconstruct.npy', 'rb') as f:
            pose_img_to_reconstruct = np.load(f)
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/evaluation/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_input.npy', 'rb') as f:
            embedding_input = np.load(f)
            
        with open(f'{self.RESULTS_PATH}/{self.EMBEDDING_SOURCE}_matrices/evaluation/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_output.npy', 'rb') as f:
            embedding_output = np.load(f)
            
        expected_pose_label = self.database[tup[2]][tup[4]][1]
        input_pose_label = self.database[tup[2]][tup[3]][1]
        input_id = tup[2]
        input_emo = self.database[tup[2]][tup[3]][0]
        
        expected_pose_label = self.poses_dict[expected_pose_label]
        input_pose_label = self.poses_dict[input_pose_label]
        input_id = self.ids_dict[input_id]
        input_emo = self.emotions_dict[input_emo]

        return pose_img_to_reconstruct, embedding_input, embedding_output, expected_pose_label, input_pose_label, input_id, input_emo
