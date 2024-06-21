import os
from openai import OpenAI

model_name = 'gpt-3.5-turbo'
client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-hDvrIHlZWITXyXLTF7uLFh467dugT35W27U3LRfjaqMHihPc",
    # api_key=os.environ.get("OPENAI_API_KEY"),
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )
# print(chat_completion)

# output = list() 
# for k in chat_completion.choices: 
#     output.append(k['text'].strip()) 
# print(output)


# import openai
# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")

# # function that takes in string argument as parameter 
# def comp(PROMPT, MaxToken=50, outputs=3): 
# 	# using OpenAI's Completion module that helps execute 
# 	# any tasks involving text 
# 	response = openai.Completion.create( 
# 		# model name used here is text-davinci-003 
# 		# there are many other models available under the 
# 		# umbrella of GPT-3 
# 		model="text-davinci-003", 
# 		# passing the user input 
# 		prompt=PROMPT, 
# 		# generated output can have "max_tokens" number of tokens 
# 		max_tokens=MaxToken, 
# 		# number of outputs generated in one call 
# 		n=outputs 
# 	) 
# 	# creating a list to store all the outputs 
# 	output = list() 
# 	for k in response['choices']: 
# 		output.append(k['text'].strip()) 
# 	return output


# PROMPT = """Write a story to inspire greatness, take the antagonist as a Rabbit and protagnist as turtle.  
# Let antagonist and protagnist compete against each other for a common goal.  
# Story should atmost have 3 lines."""
# ret = comp(PROMPT, MaxToken=3000, outputs=3)
# print(ret)



def gpt_sum(val1: int, val2: int):
    # 加载提示词文件并获取提示词
    with open('./sum.prompt', 'r', encoding='utf-8') as f:
        prompt = f.read()
        
    # 首先给出gpt任务提示词
    messages = [{'role': 'system', 'content': prompt}]
    # 模拟gpt的确认响应，后续可以直接以user角色给出gpt问题
    messages.append({'role': 'assistant', "content": 'yes'})
    # 以user角色给出gpt问题
    user_input = f"\input: {val1}, {val2}"
    messages.append({'role': 'user', 'content': user_input})
    gpt_resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=30
    )
    if gpt_resp.choices and gpt_resp.choices[0]:
        resp_str: str = gpt_resp.choices[0].message.content
        if resp_str and resp_str.startswith('\output: '):
            return int(resp_str[len('\output: '):].strip())
    raise Exception(
        f'Failed to get available response from gpt, resp_str={resp_str}')

def act_gpt(pursuers_positions, evader_pos):
    with open('llm/pursuit_reward.prompt', 'r', encoding='utf-8') as f:
        prompt = f.read()
    messages = [{'role': 'system', 'content': prompt}]
    messages.append({'role': 'assistant', "content": 'yes'})
    
    user_input = "\input: "
    user_input += "["
    for i,pursuer_pos in enumerate(pursuers_positions):
        user_input += f"[{pursuer_pos[0]}, {pursuer_pos[1]}]"
        if i < len(pursuers_positions)-1:
            user_input += ", "
    user_input += "]"
    user_input += f", [{evader_pos[0]}, {evader_pos[1]}]"
    messages.append({'role': 'user', 'content': user_input})
    gpt_resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=30
    )
    if gpt_resp.choices and gpt_resp.choices[0]:
        resp_str: str = gpt_resp.choices[0].message.content
        if resp_str and resp_str.startswith('\output: '):
            print(resp_str)
            return float(resp_str[len('\output: '):].strip())
    raise Exception(
        f'Failed to get available response from gpt, resp_str={resp_str} <- {user_input}')

if __name__ == '__main__':
    p_poses = [[-0.6,-1.5],[-0.37,-0.9],[0.05,-0.12]]
    e_pos = [0.5,-0.32]
    ret = act_gpt(p_poses, e_pos)
    print(ret)

