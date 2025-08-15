import cv2
import os
import numpy as np
import json
from tqdm import tqdm

from utils import read_json, write_json, load_image, write_output

import yaml

import argparse
from models.llava import LLAVA
from models.gpt4o import GPT4O
from models.llama4scout17b import Llama4Scout17B
from models.llama32b90 import Llama3_2B90
from models.llama4maverick17b import Llama4Maverick17B
from models.gemini25pro import GeminiPro25
from models.pixtral import PixtralLarge
from models.gpt41 import GPT41
from models.qwenvl25 import QwenVL_25
from models.gemma3_27b_it import Gemma3_27b_it
from models.gpt41_mini import GPT41Mini


from useGPT import GPT

def parse():
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--data_root', type=str, default='./dataset', 
                        help='root path to the dataset')
    parser.add_argument('--save_root', type=str, default='./exp_result/', 
                        help='root path for saving results')
    parser.add_argument('--vqa_model', type=str, default='gpt41', choices=['llava','gpt4o','llama4scout','llama3290b','llama4maverick', 'geminipro25', 'pixtral_large','gpt41','qwen25','gemma3_27b_it','gpt41_mini'],
                        help='model as Answerer.')
    parser.add_argument('--llm_model', type=str, default='gpt41_mini', choices=['gpt4o_mini','gpt41_mini'])
    parser.add_argument('--device_id', type=int, default=0, 
                        help='Which GPU to use.')
    parser.add_argument('--exp_type', type=str,  default='vqa', choices=['vqa','qa','vqa_rag'],
                        help='Experiment type, vqa: Standard VQA, qa: VQA without image,  vqa_rag: RAG baseline')
    parser.add_argument('--n_answers', type=int, default=1,
                        help='Number of answers to provide in vqa. Default is 1.')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    args = parser.parse_args()
    return args

def read_video_collections_data(video_annotation_file):

    data_dic = read_json(video_annotation_file)['database']

    return data_dic

def visual_question_answering(questions_dic,model,data_root,save_path, n_answers = 1, text_only = False):
    results_path = os.path.join(save_path, 'results.json')
    if os.path.exists(results_path):
        export_dic = read_json(results_path)
    else:
        export_dic = {}


    for q_id in tqdm(questions_dic.keys()):
        q_info = questions_dic[q_id]

        if q_id in export_dic.keys():
            continue

        question = q_info['question']
        answer = q_info['answer']

    

        instruction = '''
        Provide {N} different possible answers in a JSON array of strings.
        Each answer should be one complete sentence.
        For example: ["answer1", "answer2", ..., "answerN"].
        If {N} is 1, your output should be a JSON array with one answer, like ["answer1"].
        Make sure all answers are distinct.
        '''
        if text_only:
            predict_answer = model.ask_text(question, ins_prompt = 'If no image is provided, answer the question based on the text only.' + instruction.format(N=n_answers))
        else:
            image = load_image(os.path.join(data_root, q_info['image_path']))
            predict_answer = model.ask(image, question, ins_prompt = instruction.format(N=n_answers))
        
        
        data_to_write ={'question':question,
                        'answer_label':answer,
                        'predict_answer':predict_answer,
                        'img':q_info['image_path']}

        write_output(q_id,data_to_write,save_path)
 
    return

def visual_question_answering_rag(questions_dic,model,llm_model,data_root,save_path, n_answers = 1): #refinement of llm_vlm4, better performance
    gpt = GPT(llm_model)

    video_transcript_path = os.path.join(data_root, 'recipe_transcript_dic.json')

    recipe_transcript_dic = read_json(video_transcript_path)
    
    
    results_path = os.path.join(save_path, 'results.json')
    if os.path.exists(results_path):
        export_dic = read_json(results_path)
    else:
        export_dic = {}

    if os.path.exists(os.path.join(save_path,'dataset.json')):
        questions_dic = read_json(os.path.join(save_path,'dataset.json'))

    
    caption_transcripts_dic_path = os.path.join(save_path,'caption_transcripts_dic.json')
    
    caption_transcripts_dic = read_json(caption_transcripts_dic_path) if os.path.exists(caption_transcripts_dic_path) else {}

    for q_id in tqdm(questions_dic.keys()):
        q_info = questions_dic[q_id]
        if q_id in export_dic.keys():
            continue

        recipe_type = q_info['recipe_type']

        relevant_videos = recipe_transcript_dic[recipe_type]


        image_path = q_info['image_path']
        image = load_image(os.path.join(data_root, image_path))

        question = q_info['question']


        if q_id in caption_transcripts_dic.keys():
            caption = caption_transcripts_dic[q_id]['caption']
            relevant_texts = caption_transcripts_dic[q_id]['transcripts']
            IDs = caption_transcripts_dic[q_id]['IDs']
        else:
            caption = model.ask(image,'Give a clear and concise summary of the image below in one paragraph.')
            response = gpt.rank_transcripts(question, caption, relevant_videos)
            IDs = response['IDs']
            relevant_texts = response['relevant_texts'] 

        questions_dic[q_id]['caption'] = caption

        transcripts_parts = "\n".join(relevant_texts)
        
        vlm_input = f'''
        You are given an image from a stage of making {q_info['category']}.

        {question}

        Below are suggested next-step answers extracted from similar recipe videos, selected using [a caption that may itself be generic]:
        {transcripts_parts}

        If the image is not informative, try to use the support text *only if it is a clear, stepwise match* for what should logically come next. 
        If the support text is also generic, ambiguous, or doesn’t clearly fit, rely on your own recipe knowledge to infer the most likely next step.
        If both the image and the suggested texts are non-informative, answer according to the typical process for this recipe.
        '''

        answer = q_info['answer']

        instruction = '''
        Provide {N} different possible answers in a JSON array of strings.
        Each answer should be one complete sentence.
        For example: ["answer1", "answer2", ..., "answerN"].
        If {N} is 1, your output should be a JSON array with one answer, like ["answer1"].
        Make sure all answers are distinct.
        '''
        
        predict_answer = model.ask(image, vlm_input, ins_prompt = instruction.format(N=n_answers))

              
        data_to_write ={'vlm_input':vlm_input,
                        'question':question,
                        'answer_label':answer,
                        'predict_answer':predict_answer,
                        'img':image_path,
                        'ranked_videos_relevant_texts':relevant_texts,
                        'IDs':IDs}
                        
        write_output(q_id,data_to_write,save_path)

        write_json(os.path.join(save_path,'dataset.json'), questions_dic)

        caption_transcripts_dic[q_id] = {'caption': caption, 'transcripts': relevant_texts, 'IDs': IDs}

        write_json(os.path.join(save_path,'caption_transcripts_dic.json'), caption_transcripts_dic)
    return




if __name__ == '__main__':
    args = parse()
    dataset_file = os.path.join(args.data_root, 'TemporalCook.json')

    questions_dic = read_json(dataset_file)

    if args.vqa_model == 'llava':
        vqa_model = LLAVA()
    elif args.vqa_model == 'gpt4o':
        vqa_model = GPT4O()
    elif args.vqa_model == 'llama4scout':
        vqa_model = Llama4Scout17B()
    elif args.vqa_model == 'llama3290b':
        vqa_model = Llama3_2B90()
    elif args.vqa_model == 'llama4maverick':
        vqa_model = Llama4Maverick17B()
    elif args.vqa_model == 'geminipro25':
        vqa_model = GeminiPro25()
    elif args.vqa_model == 'pixtral_large':
        vqa_model = PixtralLarge()
    elif args.vqa_model == 'gpt41':
        vqa_model = GPT41()
    elif args.vqa_model == 'qwen25':
        vqa_model = QwenVL_25()
    elif args.vqa_model == 'gemma3_27b_it':
        vqa_model = Gemma3_27b_it()
    elif args.vqa_model == 'gpt41_mini':
        vqa_model = GPT41Mini()

    if args.exp_type == 'vqa' or args.exp_type == 'qa':
        args.llm_model = 'no_llm'
    if args.llm_model == 'gpt4o_mini':
        llm_model = 'gpt-4o-mini'
    elif args.llm_model == 'gpt41_mini':
        llm_model = 'gpt-4.1-mini'

    save_path = os.path.join(args.save_root, f'{args.vqa_model}_{args.llm_model}_{args.exp_type}', f'n_answers_{args.n_answers}')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    exp_type = args.exp_type
    if exp_type == 'vqa':
         visual_question_answering(questions_dic, vqa_model, args.data_root, save_path, n_answers = args.n_answers)
    elif exp_type == 'qa':
        visual_question_answering(questions_dic, vqa_model, args.data_root, save_path, n_answers = args.n_answers, text_only = True)
    elif exp_type == 'vqa_rag':
        visual_question_answering_rag(questions_dic, vqa_model, llm_model, args.data_root, save_path, n_answers = args.n_answers)
    else:
        raise ValueError('Invalid experiment type. Choose from vqa, qa, vqa_rag')
    

   
    
    

    