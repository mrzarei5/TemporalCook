import cv2
import os
import numpy as np

import argparse
import csv

from utils import read_json, write_json, load_image


from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image


from useGPT import GPT


from useWhisper import TranscriptExtractor


def parse():
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--youcook_root', type=str, default='../datasets/youcook2', 
                        help='root path to the dataset')
    parser.add_argument('--temporalcook_root', type=str, default='./dataset', 
                        help='root path to the dataset')
    args = parser.parse_args()
    return args

def read_video_collections_data(video_annotation_file):

    data_dic = read_json(video_annotation_file)['database']

    return data_dic



def collect_transcripts(questions_dic, video_collection_dic, args):
    
    recipe_type_list = []
    video_ids_used = []
    for q_id, q_info in questions_dic.items():
        recipe_type = str(q_info['recipe_type'])
        video_id = q_info['video_id']

        if recipe_type not in recipe_type_list:
            recipe_type_list.append(recipe_type)
        if video_id not in video_ids_used:
            video_ids_used.append(video_id)
    
    
    video_transcript_path = os.path.join(args.temporalcook_root, 'recipe_transcript_dic.json')

    if os.path.isfile(video_transcript_path):
        recipe_transcript_dic = read_json(video_transcript_path)
        return recipe_transcript_dic
    else:
        recipe_transcript_dic = {}

    recipe_transcript_dic = {}

    for recipe_type in recipe_type_list:
        recipe_transcript_dic[recipe_type] = {}

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"]

    transcriber = TranscriptExtractor('turbo')


    for video_id, video_info in video_collection_dic.items():
        recipe_type = video_info['recipe_type']
   
        if recipe_type in recipe_type_list:


            recipe_videos = recipe_transcript_dic[recipe_type]

            if video_id not in video_ids_used and video_id not in recipe_videos.keys():
                
                video_dir = os.path.join(args.youcook_root, "raw_videos", video_info["subset"], str(video_info["recipe_type"]))
                for extension in video_extensions:
                    video_path = os.path.join(video_dir, video_id + extension)
                    if os.path.exists(video_path):
                        break

                trasncript = transcriber.video_to_transcript(video_path)

                recipe_transcript_dic[recipe_type][video_id] = trasncript
        
                write_json(video_transcript_path, recipe_transcript_dic)
    
    return recipe_transcript_dic

if __name__ == "__main__":
    args = parse()

    questions_file = os.path.join(args.temporalcook_root, 'TemporalCook.json')

    questions_dic = read_json(questions_file)
    
    video_annotation_file = os.path.join(args.youcook_root,'youcookii_annotations_trainval.json')
    
    video_collection_dic = read_video_collections_data(video_annotation_file)
    
    
    recipe_transcripts_dic = collect_transcripts(questions_dic, video_collection_dic, args)
    print(f"Collected transcripts for {len(recipe_transcripts_dic)} recipe types.")