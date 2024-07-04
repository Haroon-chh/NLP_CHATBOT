from transformers import MarianMTModel, MarianTokenizer

# Initialize translation model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-ur'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(text):
    translated = model.generate(**tokenizer.prepare_seq2seq_batch([text]))
    return tokenizer.decode(translated[0], skip_special_tokens=True)
