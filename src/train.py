import logging
import os
import sys

from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.arguments import ModelArguments, DataArguments, TrainingArguments, IndexArguments
from src.factory.factory import create_train_components

logger = logging.getLogger(__name__)

def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, IndexArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, index_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        model_args, data_args, training_args, index_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, index_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, index_args

def check_output_dir(output_dir: str, training_args: TrainingArguments):
    if os.path.exists(output_dir) and os.listdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    os.makedirs(output_dir, exist_ok=True)

def main():
    model_args, data_args, training_args, index_args = parse_args()
    check_output_dir(training_args.output_dir, training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16 or training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    
    set_seed(training_args.seed)

    (
        _model,
        processor,
        _collator,
        _train_dataset,
        trainer,
        _torch_dtype,
    ) = create_train_components(model_args, data_args, training_args, index_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info(f"Resuming from last checkpoint: {training_args.output_dir}")

    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()