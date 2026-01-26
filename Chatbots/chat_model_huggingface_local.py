from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# It will download model in D drive.
# import os
# os.environ["HF_HOME"] = "D:/huggingface_cache"

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

llm = HuggingFacePipeline.from_model_id(model_id= model_id,
                                        task = "text-generation", pipeline_kwargs= dict
                                        (temperature = 0.5,
                                        max_new_tokens = 100)
                            )
model_use = ChatHuggingFace(llm =llm)
results = model_use.invoke("What are transformers in AI?")

print(results)