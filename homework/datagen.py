from homework.base_llm import BaseLLM

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    
    # load the llm model (going to use 1.7B model)
    llm_17 = BaseLLM(checkpoint=)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
