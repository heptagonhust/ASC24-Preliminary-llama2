from vllm import LLM, SamplingParams
import json

llm = LLM(model = "/data/70B-chat-hf", tensor_parallel_size=2, pipeline_parallel_size=3)
with open("./scrambled_sampled_dataset.json", "r") as f:
    data = json.load(f)

prompt_list = []
for src in data:
    prompt_list.append(src[0])

sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
outputs = llm.generate(prompt_list, sampling_params)

for output in outputs:
    prompt = output.prompt # 获取原始的输入提示
    generated_text = output.outputs[0].text # 从输出对象中获取生成的文本
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
