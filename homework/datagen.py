from homework.data import Dataset, is_answer_valid
from homework.cot import CoTModel
import numpy as np
import json

def get_data_batches(data: Dataset, batch_size=1) -> list[str]:

    question_batch = list()
    answer_batch = list()

    for question, answer in data:

        if len(question_batch) == batch_size:
            yield question_batch, answer_batch
            question_batch = list()
            answer_batch = list()

        
        question_batch.append(question)
        answer_batch.append(answer)
    
    if question_batch:
        yield  question_batch, answer_batch

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6, batch_size=1):
    
    # load the llm model (going to use 1.7B model)
    llm_17 = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    training_data = get_data_batches(Dataset("train"), batch_size=batch_size)
    cot_training_data = list()

    processed_questions = 0
    for question_batch, answer_batch in training_data:

        # format the questions
        formatter = np.vectorize(lambda x: llm_17.format_prompt(x))
        formatted_questions = formatter(question_batch)
        
        responses = llm_17.batched_generate(
                                list(formatted_questions),
                                num_return_sequences=oversample,
                                temperature=temperature
                            )        

        # pull out valid data
        for question, response, answer in zip(
                                                question_batch,
                                                responses,
                                                answer_batch
                                            ):
            

            # get rid of im_end tags
            replacer = np.vectorize(lambda x: x.replace("<|im_end|>", ""))
            replaced_response = replacer(response)

            # parse the answers out
            parser = np.vectorize(lambda x: llm_17.parse_answer(x))
            parsed_responses = parser(replaced_response)

            # validate the answers
            validator = np.vectorize(lambda x: is_answer_valid(x, answer))            
            valid_responses = replaced_response[validator(parsed_responses)]

            # choose one valid response if available
            if valid_responses.size > 0:
                valid_response = valid_responses[0]

                cot_training_data.append(
                    [
                        question,
                        answer,
                        valid_response.item()
                    ]
                )   
        
        processed_questions += batch_size
        print(f"processed {processed_questions} questions.")
    
    print(f"generated {len(cot_training_data)} examples.")
    with open(output_json, mode="w") as f:
        json.dump(cot_training_data, f, indent=4)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
