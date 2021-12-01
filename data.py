from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class wikisql(Dataset):
    def __init__(self,type_path: str, 
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M"), 
                       sql2txt: bool = True) -> None:      

        # self.dataset =  load_dataset('wikisql', 'all', data_dir='data/', split=type_path)
        self.dataset =  load_dataset('wikisql', split=type_path)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.sql2txt = sql2txt
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
  
    def __len__(self) -> int:
        return self.dataset.shape[0]
        # return len(self.dataset["train"])
    
    def clean_text(self, text: str) -> str:
        return text.replace('\n','').replace('``', '').replace('"', '')

    
    def convert_to_features(self, example_batch):                
        if self.sql2txt:
            # sql to text
            input_ = "translate SQL to English: " + self.clean_text(example_batch['sql']['human_readable'])
            target_ = self.clean_text(example_batch['question'])
        else: 
            # text to sql
            input_ = "translate English to SQL: " + self.clean_text(example_batch['question'])
            target_ = self.clean_text(example_batch['sql']['human_readable'])
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        # return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}


def get_dataset(type_path: str, num_samples: int, max_input_length, max_output_length, sql2txt) -> wikisql:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", max_length=max_output_length)
    return wikisql(type_path=type_path,
                num_samples=num_samples,  
                input_length=max_input_length, 
                output_length=max_output_length,
                tokenizer=tokenizer,
                sql2txt=sql2txt)