from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


emotions = load_dataset("json", data_files={"train": "data/emotions_train.json"})["train"]
rituals = load_dataset("json", data_files={"train": "data/rituals_train.json"})["train"]
smalltalk = load_dataset("json", data_files={"train": "data/smalltalk_train.json"})["train"]
identity = load_dataset("json", data_files={"train": "data/train.json"})["train"]


dataset = concatenate_datasets([emotions, rituals, smalltalk, identity])

def preprocess(example):
    
    model_inputs = tokenizer(
        example["input"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        text_target=example["output"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=25,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
trainer.save_model("./my-sacred-model")
tokenizer.save_pretrained("./my-sacred-model")
