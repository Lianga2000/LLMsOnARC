import timeit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM


class ArcathonPipeline(torch.nn.Module):
    def __init__(self, prompt_convertor, model_id, max_new_tokens, min_new_tokens, top_p, temperature, device,
                 tries_amount, repetition_penalty, load_in_8bit, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_convertor = prompt_convertor
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self.init_model(device, max_new_tokens, min_new_tokens, model_id, temperature, top_p, repetition_penalty)
        self.tries_amount = tries_amount
        self.model.eval()

    def init_model(self, device, max_new_tokens, min_new_tokens, model_id, temperature, top_p, repetition_penalty,
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=self.load_in_8bit, device_map=device,
                                                          trust_remote_code=True)
        self.generation_config = GenerationConfig(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                  top_p=top_p, temperature=temperature, do_sample=True)

    def forward(self, train, test):
        success = False
        saved_outputs = []
        for i in range(self.tries_amount):
            start = timeit.default_timer()
            prompt = self.prompt_convertor.convert_task_to_prompt(train, test)
            text_output = self.forward_prompt(prompt)
            saved_outputs.append(text_output)
            # print(f'Inserted prompt {prompt}')
            # print(f'Length of output {len(text_output)}')
            # print('Output:')
            # print(text_output)
            real_out_mat_as_txt = self.prompt_convertor.convert_mat_to_text(test[0]['output'])
            if real_out_mat_as_txt in text_output:
                success = True
            end = timeit.default_timer()
            if success:
                # print(f'Success! Generation took {end - start} seconds')
                break
            # else:
            #     if i + 1 == self.tries_amount:
            #         print(f'Failed to generate correct output completly.')
            #     else:
            #         print(f'Failed to generate correct output, try : {i + 1} / {self.tries_amount}. trying again')
            #     print(f'Generation took {end - start}s')
        return success, saved_outputs

    def forward_prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        to_find = '\n assistant\n'
        location = text_output.find(to_find)
        out = text_output[location + len(to_find):]
        return out, text_output

    def __str__(self):
        return f'{type(self)}_use8bit={self.load_in_8bit}'

    def __repr__(self):
        return self.__str__()

class Mixtral87B(ArcathonPipeline):
    def init_model(self, device, max_new_tokens, min_new_tokens, model_id, temperature, top_p, repetition_penalty,
                   **kwargs):
        super().init_model(device, max_new_tokens, min_new_tokens, model_id, temperature, top_p, repetition_penalty,
                   **kwargs)
        self.bos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id

    def convert_messages_to_prompt(self, messages):
        final_str = ''

        for msg in messages:
            role = msg['role']
            if role == 'system' or role == 'System':
                role = 'System'
            elif role == 'user' or role == 'User':
                role = 'User'
            elif role == 'assistant' or role == 'Assistant':
                role = 'Assistant'
            else:
                raise Exception('Unknown role')
            content = msg['content']
            if role == 'User':
                final_str+=f'[INST] {content} [/INST] '
            elif role == 'Assistant':
                final_str+=content
        return final_str

    def forward_prompt(self, prompt_as_msgs):
        prompt = self.convert_messages_to_prompt(prompt_as_msgs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')  # self.device)
        input_token_amount = inputs['input_ids'][0].shape[0]
        # print(f'Input token amount for generation : {input_token_amount}')
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.pad_token_id)
            outputs = outputs[0][input_token_amount:]
        text_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return text_output

    # def forward_prompt(self, prompt_as_msgs):
    #     input_ids = []
    #     for msg in prompt_as_msgs:
    #         if msg['role'] in ['user','User']:
    #             input_ids.append(self.bos_token)
    #             input_ids+=self.tokenizer.encode('[INST]',add_special_tokens=False)
    #             input_ids+=self.tokenizer.encode(msg['content'],add_special_tokens=False)
    #             input_ids+=self.tokenizer.encode('[/INST]',add_special_tokens=False)
    #         elif msg['role'] in ['Assistant','assistant']:
    #             input_ids+=self.tokenizer.encode(msg['content'],add_special_tokens=False)
    #             input_ids.append(self.eos_token)
    #     input_token_amount = len(input_ids)
    #     inputs = self.tokenizer.prepare_for_model(input_ids, return_tensors='pt').to('cuda')
    #
    #     print(f'Input token amount for generation : {input_token_amount}')
    #     with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    #         outputs = self.model.generate(**inputs, generation_config=self.generation_config)
    #         outputs = outputs[0][input_token_amount:]
    #     text_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
    #     return text_output


class LLama2Code(Mixtral87B):
    "Same exact code, class exists just for the name change."

    def __str__(self):
        return self.model_id
class PhindArcathonPipeline(ArcathonPipeline):
    def init_model(self, device, max_new_tokens, min_new_tokens, model_id, temperature, top_p, repetition_penalty,
                   **kwargs):
        tokenizer_name = 'Phind/Phind-CodeLlama-34B-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(model_id, device_map=device,
                                                      load_in_8bit=self.load_in_8bit,
                                                      do_sample=True,
                                                      trust_remote_code=True)
        self.generation_config = GenerationConfig(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                  top_p=top_p, temperature=temperature, do_sample=True,
                                                  repetition_penalty=repetition_penalty)

    def convert_messages_to_prompt(self, messages):
        final_str = ''

        for msg in messages:
            role = msg['role']

            if role == 'system' or role == 'System':
                role = 'System'
            elif role == 'user' or role == 'User':
                role = 'User'
            elif role == 'assistant' or role == 'Assistant':
                role = 'Assistant'
            else:
                raise Exception('Unknown role')
            content = msg['content']
            final_str += f"### {role} Prompt\n"
            final_str += f'{content}\n'
        final_str += f"### Assistant Prompt\n"
        return final_str

    def forward_prompt(self, prompt_as_msgs):
        prompt = self.convert_messages_to_prompt(prompt_as_msgs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')  # self.device)
        input_token_amount = inputs['input_ids'][0].shape[0]
        # print(f'Input token amount for generation : {input_token_amount}')
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.pad_token_id)
            outputs = outputs[0][input_token_amount:]
        text_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return text_output
