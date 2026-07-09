from transformers import AutoTokenizer, AutoModelForNextSentencePrediction

MODEL_ID = "desilo-ai/bert-base-uncased-finetuned-mrpc"
# desilo model does not have a tokenizer, so we use the base model's tokenizer instead.
TOKENIZER_ID = "google-bert/bert-base-uncased"


def get_model(device="cpu"):
    model = AutoModelForNextSentencePrediction.from_pretrained(MODEL_ID).to(device)
    return model


def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)
