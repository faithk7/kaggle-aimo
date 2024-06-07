from transformers import Trainer, TrainingArguments

from config import config
from dataset import load_data
from model import apply_lora, initialize_model


def finetune(model, dataset_dict, output_dir: str):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
    )

    trainer.train()


def main():
    tokenizer, model = initialize_model(config.MODEL, cache_dir=config.CACHE_DIR)
    model = apply_lora(model)

    dataset_dict = load_data(config.TRAIN_PATH, tokenizer)

    finetune(model, dataset_dict, output_dir=config.MODEL_ROOT)


if __name__ == "__main__":
    main()
