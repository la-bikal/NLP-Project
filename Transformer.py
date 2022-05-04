from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
import torch


class CustomBERTModel(torch.nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
          ### New layers:
          self.linear1 = nn.Linear(32, 4)

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

          return linear1_output

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
model = CustomBERTModel() # You can pass the parameters if required to have more flexible model
model.to(torch.device("cuda:0")) ## can be gpu

criterion = nn.CrossEntropyLoss() ## If required define your own criterion
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# for epoch in epochs:
#     for batch in data_loader: ## If you have a DataLoader()  object to get the data.

#         data = batch[0]
#         targets = batch[1] ## assuming that data loader returns a tuple of data and its targets
        
#         optimizer.zero_grad()   
#         encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,max_length=50, add_special_tokens = True)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         outputs = F.log_softmax(outputs, dim=1)
#         input_ids = encoding['input_ids']
#         attention_mask = encoding['attention_mask']
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
        
