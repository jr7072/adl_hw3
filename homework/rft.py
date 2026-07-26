from .base_llm import BaseLLM
from .sft import test_model, tokenize
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from .data import Dataset


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class TokenizedDatasetRFT:
    def __init__(self, tokenizer, data: Dataset):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """

        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, _, response = self.data[idx]
        return tokenize(self.tokenizer, question=question, answer=response)


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    # grab the pretrained model
    llm = BaseLLM()

    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=64
    )

    lora_model = get_peft_model(llm.model, lora_config) # ty: ignore
    lora_model.enable_input_require_grads()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        gradient_checkpointing=True,
        logging_dir=output_dir,
        report_to="tensorboard",
        eval_strategy="epoch",        # Evaluate at the end of each epoch
        save_strategy="no",        # Save checkpoint at the end of each epoch
        learning_rate=2e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        lr_scheduler_type="reduce_lr_on_plateau"
    )

    trainer = Trainer(
        lora_model,
        args=training_args,
        train_dataset=TokenizedDatasetRFT(llm.tokenizer, data=Dataset("rft")),
        eval_dataset=None
    )
    
    trainer.train() # ty: ignore
    trainer.save_model(output_dir) # ty: ignore
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
