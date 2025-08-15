from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from dotenv import dotenv_values
import json
import os
import argparse
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class AnswerMatches(BaseModel):
    answer_justification: str = Field(...,description="state the verifying process in one or two sentences")
    answer_matches: Literal["yes","no"] = Field(...,description="state whether any of the predicted answers matches with the ground truth answer or not")
    

def parse():
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--save_root', type=str, default='./exp_result/', help='root path to the saving directory')
    parser.add_argument('--exp_type', type=str,  default='vqa', choices=['vqa','qa','vqa_rag'],
                        help='Experiment type, vqa: Standard VQA, qa: VQA without image,  vqa_rag: RAG baseline')
    parser.add_argument('--n_answers', type=int, default=1,
                        help='Number of answers to provide in vqa. Default is 1.')
    parser.add_argument('--vqa_model', type=str, default='gpt41', choices=['llava','gpt4o','llama4scout','llama3290b','llama4maverick', 'geminipro25', 'pixtral_large','gpt41','qwen25','gemma3_27b_it','gpt41_mini'],
                        help='model as Answerer.')
    parser.add_argument('--llm_model', type=str, default='gpt41_mini', choices=['gpt4o_mini','gpt41_mini'])
    parser.add_argument('--api_type_key', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--evaluator', type=str, default='gpt41_mini', choices=['gpt41_mini', 'gpt41'], help='evaluator model')
    return parser.parse_args()


def get_user_base_prompt():
    user_base_prompt = '''
        You are a meticulous evaluator for Visual Question Answering (VQA) models. Your task is to determine whether any of the predicted answers matches the ground truth answer semantically (even if phrased differently).

        Rules:
        Criteria for Acceptance:

        At least one predicted answer must explicitly contain or logically imply the ground truth answer.
        Synonyms, rephrasings, or expanded details (e.g., “azure” instead of “blue”) are acceptable only if they do not alter the core meaning.
        Extra correct information in a predicted answer is allowed if it does not contradict the ground truth.
        
        Criteria for Rejection:

        Answers that are vague, tangential, or overly broad (e.g., “It is a color” when the ground truth is “red”).
        Answers that assume unstated context (e.g., “The vehicle” when the ground truth is “ambulance”).
        Instructions:
        Given the question, ground truth answer, and a list of predicted answers, you must:

        Compare the ground truth answer to each item in the predicted answer list using the above rules.
        Demonstrate your verification process in one or two sentences (state which, if any, predicted answer matches and why).
        Clearly state whether any predicted answer matches the ground truth answer (“Yes” or “No”).
        Your reply format:
        Verification: [A brief description of your checking process and conclusion (in one or two sentences).]
        Match: Yes/No

        You have to respond with a structured solution including: 
        1. demonstrate your verifying process in one or two sentences. 
        2. whether any of the predicted answers matches with the ground truth answer or not'''
    return user_base_prompt


def prepare_user_prompt(question, predicted_answer, ground_truth):
    user_prompt = f'''
    Question: {question} 
    Ground truth answer:{ground_truth}
    Predicted answers: {predicted_answer} 
    '''
    return user_prompt


def append_history(history, new_message, role="user"):
    """
    Appends a new message to the conversation history.
    
    Args:
        history (list): A list of dictionaries representing the conversation history.
        new_message (str): The new message to append.
        role (str): The role of the message, either 'user' or 'assistant'.
        
    Returns:
        list: Updated conversation history.
    """
    history.append({"role": role, "content": new_message})
    return history


def get_sys_prompt():
    sysContent = '''
        You are an expert evaluator for VQA models. You evaluate model answers to visual questions based on correctness.'''
    return sysContent



def score_answer_ground_truth(client, model, question, predicted_answer, ground_truth, history=[]):
    if not history:
        history = append_history(history, get_sys_prompt(), role="system")
        history = append_history(history, get_user_base_prompt(), role="user")        
    history = append_history(history, prepare_user_prompt(question,predicted_answer,ground_truth), role="user")

    completion = client.beta.chat.completions.parse(
        model= model,
        messages = history,
        temperature=0,
        top_p=1,
        response_format= AnswerMatches,
    
    )
    response = completion.choices[0].message.parsed
    response = response.model_dump_json()
    response = json.loads(response)

    try:
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        tokens = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
    except KeyError:
        print("Token usage information not available in the response.")
    return response, tokens


def format_response(response):
    formatted_text = ""
    for key, value in response.items():
        formatted_text += f"{key}: {value}\n"
    return formatted_text.strip()

def prepare_evaluator(evaluator):
    if evaluator == 'gpt41_mini':
        return "gpt-4.1-mini-2025-04-14"
    elif evaluator == 'gpt41':
        return "gpt-4.1-2025-04-14"
    else:
        raise ValueError("Invalid evaluator name")

def prepare_client(api_key):
    config = dotenv_values(".env")  
    if "openai" in api_key.lower():
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
    elif "hf" in api_key.lower() or "huggingface" in api_key.lower():
        client = OpenAI(api_key=config["HF_API_KEY"])
    else:
        raise ValueError("Invalid API key")
    return client


if __name__ == '__main__':
    args = parse()

    save_root = args.save_root
    vqa_model = args.vqa_model
    llm_model = args.llm_model
    exp_type = args.exp_type
    if exp_type == 'vqa' or exp_type == 'qa':
        llm_model = 'no_llm'
    
    evaluator = prepare_evaluator(args.evaluator)

    client = prepare_client(args.api_type_key)


    save_path = os.path.join(save_root, f'{vqa_model}_{llm_model}_{exp_type}')
    save_path = os.path.join(save_path, f'n_answers_{args.n_answers}')

    path_s = os.path.join(save_path, 'results.json')
    path_a = os.path.join(save_path, 'results_a_'+evaluator+'.json')
    path_metrics = os.path.join(save_path, 'metrics_'+evaluator+'.json')
    path_tokens = os.path.join(save_path, 'tokens_'+evaluator+'.json')
    
    if os.path.exists(path_a):
        export_dic = read_json(path_a)
        metrics = read_json(path_metrics)
        tokens = read_json(path_tokens)
    else:
        export_dic = {}    
        metrics = {'accuracy': 0}
        tokens = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    model_answers_dic = read_json(path_s)
    
    count = 0

    for id in tqdm(model_answers_dic.keys()):
        data_dic = model_answers_dic[id]
      
        if export_dic.get(id, None):
            count += 1
            continue        
        
    
        judge_answer, request_tokens = score_answer_ground_truth(client, evaluator, data_dic['question'], data_dic['predict_answer'], data_dic['answer_label'],[])
        answer_justification = judge_answer['answer_justification']
        answer_matches = judge_answer['answer_matches']
    
        if 'yes' in answer_matches.lower():
            metrics['accuracy'] += 1


        export_dic[id] = {}
        export_dic[id]['question'] = data_dic['question']
        export_dic[id]['predict_answer'] = data_dic['predict_answer']
        export_dic[id]['answer_label'] = data_dic['answer_label']
        
        export_dic[id]['img'] = data_dic['img']

        if llm_model != 'no_llm':
            export_dic[id]['vlm_input'] = data_dic['vlm_input']
        
        export_dic[id]['answer_matches'] = judge_answer['answer_matches']
        export_dic[id]['answer_justification'] = judge_answer['answer_justification']
     
        tokens['input_tokens'] += request_tokens['input_tokens']
        tokens['output_tokens'] += request_tokens['output_tokens']
        tokens['total_tokens'] += request_tokens['total_tokens']

        with open(path_a, "w") as outfile:
            json.dump(export_dic, outfile, indent = 2) 
    
        
        with open(path_metrics, "w") as outfile:
            json.dump(metrics, outfile, indent = 2)  

        with open(path_tokens, "w") as outfile:
            json.dump(tokens, outfile, indent = 2)

        count += 1
    
    metrics['accuracy'] = metrics['accuracy'] / count
    

    with open(path_metrics, "w") as outfile:
        json.dump(metrics, outfile, indent = 2)  

    print('vqa_model: ', vqa_model, 'mode: ', exp_type, 'n_answers: ' , args.n_answers,'accuracy: ', metrics['accuracy']) 
  
    print('vqa_model: ', vqa_model, 'mode: ', exp_type, 'n_answers: ' , args.n_answers, ' done')
    