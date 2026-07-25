from .base_llm import BaseLLM


SYSTEM_PROMPT = (
    "Task: Perform the conversions the user asks for. Provide the answer only in format <answer>FINAL_NUMERIC_ANSWER</answer>\n"
    "Be concise"
)


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role":  "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: please convert 6 days into weeks. provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 day = 0.14285714285 weeks. 6 * 0.14285714285 = <answer>0.8571428571428571</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: What is the equivalent of 7 KB in bit? provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 kilobyte = 1000 bytes. 1 Byte = 8 bits. 7000 * 8 = <answer>54000</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: How do we express 6 pound in terms of gram? provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 pound = 453.592 grams. 453.592 * 6 = <answer>2721.55422</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: Express 10 m/s as a quantity of mi/h. provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 meter = 0.000621371 miles. 1 second = 0.00027777777 hours. 10 * 0.000621371 / 0.00027777777 = <answer>22.369362920544024</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: What is the conversion from ton to pound for 8 units? provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 metric ton = 2,204.62 pounds. 8 * 2,204.62 = <answer>17636.980974790207</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: How much is 6 km/h when converted to in/s? provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 kilometer = 1000 meters. 1 meter = 39.3701 inches. 1 hour = 3600 seconds 8. \n"
                    "6000 * 39.3701 / 3600 = <answer>65.61679790026247</answer>\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "User Prompt: How many MB make up 1 G? provide answer in <answer></answer>. Do not include units in answer tag\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Answer: 1 gigabyte = 1000MB. <answer>1000</answer>"
                )
            },
            {
                "role": "user",
                "content": f"User Prompt: {question} provide answer in <answer></answer>. Do not include units in answer tag\n"
            }
        ]

       

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return formatted_prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
