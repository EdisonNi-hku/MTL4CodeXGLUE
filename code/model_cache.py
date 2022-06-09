from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

T5Config.from_pretrained('t5-base', cache_dir='cache')
T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir='cache')
T5Tokenizer.from_pretrained('t5-base', cache_dir='cache')

T5Config.from_pretrained('Salesforce/codet5-base', cache_dir='cache')
T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base', cache_dir='cache')
T5Tokenizer.from_pretrained('Salesforce/codet5-base', cache_dir='cache')

T5Config.from_pretrained('razent/cotext-2-cc', cache_dir='cache')
T5ForConditionalGeneration.from_pretrained('razent/cotext-2-cc', cache_dir='cache')
T5Tokenizer.from_pretrained('razent/cotext-2-cc', cache_dir='cache')
