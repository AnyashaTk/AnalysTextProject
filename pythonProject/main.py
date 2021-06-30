import telebot
import config
import traceback
import torch
import tqdm as tqdm
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import RobertaModel, RobertaTokenizer



class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()

model = torch.load('ROBERTA.pt', map_location = torch.device('cpu'))

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True, truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

bot = telebot.TeleBot(config.TOKEN)

MAX_LEN = 256
def tokenized_and_modelwork(test_data, model):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

    testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

    test_params = {'batch_size': 1,
                   'shuffle': True,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)
    model.eval()

    for data in testing_loader:
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        outputs = model(ids, mask, token_type_ids).squeeze()
        print(len(outputs.data), outputs.data)
        if len(outputs.data)==1:
            outputs.data =[outputs.data, outputs.data]
        '''
        tens = torch.tensor([[-2.9300, 2.7853],
         [1.1235, -1.0567],
         [-0.4472, 0.3536],
         [1.8517, -1.8065],
         [-1.1720, 1.0690],
         [-1.9747, 1.8661],
         [0.1995, -0.2787],
         [-2.7848, 2.7081],
         [2.8482, -2.6744],
         [-0.1953, 0.1545],
         [-0.5027, 0.4557],
         [-2.3924, 2.2536],
         [-2.7311, 2.5852],
         [-1.8916, 1.8069],
         [-3.1411, 2.9779],
         [-1.8415, 1.8894], list([outputs.data[0].item(), outputs.data[1].item()])])'''

        tens = torch.tensor([list([outputs.data[0].item(), outputs.data[1].item()])])

        big_val, big_idx = torch.max(tens, dim=1)

        return big_idx[0].item() #outputs.data[0].item()

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,
                     'Привет! Пришли небольшой текст сюда, а нейронная сеть определит его эмоциональную окраску.')


@bot.message_handler(content_types=['text'])
def repeat_all_messages(message):
    try:
        bot.send_message(message.chat.id, text='Увидел, думаем...')
        model.eval()
        outputs = tokenized_and_modelwork(message, model)

        if (outputs < 1):
            bot.send_message(message.chat.id, text="Это позитивный текст!")
        else:
            bot.send_message(message.chat.id, text="Это негативный текст!")

    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так :(')

bot.polling(none_stop=True)

if __name__ == '__main__':
    import time

    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')