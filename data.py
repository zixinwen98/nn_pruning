from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

class wikisql(Dataset):
    def __init__(self, dataset_path: str,
                       type_path: str, 
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       sql2txt: bool = True) -> None:      

        # if dataset_path is None:
        self.dataset =  load_dataset('wikisql', split=type_path)
        # else:
        #     self.dataset =  load_from_disk(dataset_path)[type_path]
        
        # self.dataset =  load_dataset('wikisql', 'all', data_dir='data/', split=type_path)
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
            0/0
            input_ = "translate SQL to English: " + self.clean_text(example_batch['sql']['human_readable'])
            target_ = self.clean_text(example_batch['question'])
        else: 
            # text to sql
            context = ["Question"] \
                        + [example_batch['question']] \
                        + ["Context"] \
                        + ["header"] + example_batch['table']['header'] \
                        + ["page_title", example_batch['table']['page_title']] \
                        + ["types"] + example_batch['table']['types'] \
                        + ["id", example_batch['table']['id']] \
                        + ["section_title", example_batch['table']['section_title']] \
                        + ["caption", example_batch['table']['caption']] \
                        + ["name", example_batch['table']['name']] \
                        + ["Question"] \
                        + [example_batch['question']] \
                        + ["SQL "] \
            
            context = " ".join(context)
            # input_ = "translate English to SQL: " + self.clean_text(example_batch['question'])
            # target_ = self.clean_text(example_batch['sql']['human_readable'])

            input_ = self.clean_text(context)
            target_ = self.clean_text(example_batch['sql']['human_readable'])
        
        # print(input_)
        # 0/0
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([input_ + target_], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus(["S " + target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")

        return source, targets
        
        # encoded_input = self.tokenizer(input_, target_, max_length=self.output_length, padding="max_length", truncation=True, return_tensors="pt")
       
        # return encoded_input
  
    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        # return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        # return {"input_ids": source_ids, "attention_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}

        # encoded_input =  self.convert_to_features(self.dataset[index])
        # output = {}
        # output["input_ids"] = encoded_input["input_ids"]
        # output["labels"] = encoded_input["input_ids"]
        # output["attention_mask"] = encoded_input["attention_mask"]
        # return output


def get_dataset(tokenizer_path: str, dataset_path: str, type_path: str, num_samples: int, max_input_length, max_output_length, sql2txt) -> wikisql:
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", max_length=max_output_length, add_special_tokens=True)
    # tokenizer.save_pretrained("./tokenizer/")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)

    return wikisql( dataset_path=dataset_path,
                    type_path=type_path,
                    num_samples=num_samples,  
                    input_length=max_input_length, 
                    output_length=max_output_length,
                    tokenizer=tokenizer,
                    sql2txt=sql2txt)