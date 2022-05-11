from dataclasses import replace
from json import load
import torch, re, argparse, pickle, random

class Embeddings(torch.nn.Module):
    def __init__(self, word_dict:dict, word_set_size:int, embedding_size):
        super(Embeddings, self).__init__()

        self.embeddings = torch.nn.Parameter(torch.rand(word_set_size, embedding_size, dtype=torch.float)*2.0 -1.0 )

    def forward(self, x):
        """Expects x to be a tensor of indices referring to the words."""
        #print(x)

        return self.embeddings[x, :] 

    def clip_embeddings(self):
        """Clips all embeddings to be between a value of -1 and 1"""

        with torch.no_grad():
            torch.clamp(self.embeddings, 0, 1, out=self.embeddings)

class WordPredictor(torch.nn.Module):
    def __init__(self, embedding_size: int, sequence_length: int, hidden_layers: int, hidden_layer_size:int):
        """A model that maps from word N embeddings to a predicted word embedding.
        Predicts some resulting word from some context of words.
        The size of each embedding vector, the length of the sequence and the number/size of all the hidden layers are parameters."""
        
        super(WordPredictor, self).__init__()

        self.moduleList = torch.nn.ModuleList()
        
        hidden_layer_sizes: list[int] = [hidden_layer_size]*hidden_layers
        
        for i, (c, d) in enumerate(zip([embedding_size*sequence_length]+hidden_layer_sizes, hidden_layer_sizes+[embedding_size])) :
            if i!=0: self.moduleList.append(torch.nn.LeakyReLU())
            self.moduleList.append(torch.nn.Linear(c, d))
            
        self.moduleList.append(torch.nn.Sigmoid())

    def forward(self, x):
        for module in self.moduleList:
            x = module(x)

        return x

class WordIdentifier(torch.nn.Module):
    def __init__(self, embedding_size: int, sequence_length: int, hidden_layers: int, hidden_layer_size:int):
        """A model that, given N word embeddings, will create a probability distribution that each one of those words does not belong in the set.
        """
        
        super(WordIdentifier, self).__init__()

        self.moduleList = torch.nn.ModuleList()
        
        hidden_layer_sizes: list[int] = [hidden_layer_size]*hidden_layers
        
        for i, (c, d) in enumerate(zip([embedding_size*sequence_length]+hidden_layer_sizes, hidden_layer_sizes+[embedding_size])) :
            if i!=0: self.moduleList.append(torch.nn.LeakyReLU())
            self.moduleList.append(torch.nn.Linear(c, d))

    def forward(self, x):
        for module in self.moduleList:
            x = module(x)

        return x

class Model(torch.nn.Module):
    def __init__(self, word_dict:dict, word_set_size:int, args):
        super(Model, self).__init__()
        self.sequence_length : int = args.sl
        self.embedding_size : int = args.s

        self.embeddings = Embeddings(word_dict, word_set_size, self.embedding_size)
        self.word_predictor = WordPredictor(self.embedding_size, self.sequence_length, args.nh, args.sh)

    def forward(self, x):

        x = self.embeddings(x)
        x = x.reshape(-1, self.embedding_size*self.sequence_length)
        x = self.word_predictor(x)

        return x


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, text_path: str, sequence_length: int, word_dict: dict, device:str):
        self.word_dict = word_dict
        self.device = device

        string: str = None
        with open(text_path) as f:
            string = f.read()
            string = string.lower()

        all_words: list[str] = re.findall("[a-z]+", string)
        self.data: list[int] = list(map(lambda s: word_dict[s], all_words))

        self.sequence_length:int = sequence_length

    def __getitem__(self, index):
        sequence: torch.tensor = torch.tensor(self.data[index : index + self.sequence_length//2] + 
                self.data[index + self.sequence_length//2 +1 : index+self.sequence_length],
                dtype=torch.long, device=self.device )

        label:torch.tensor = torch.tensor(self.data[index + self.sequence_length//2], dtype=torch.long, device=self.device)
        
        return sequence, label

    def __len__(self):
        return len(self.data)-self.sequence_length-1

class IdentificationDataset(torch.utils.data.Dataset):
    def __init__(self, text_path: str, sequence_length: int, word_dict: dict, word_set_size:int, device:str):
        self.word_dict = word_dict
        self.word_set_size = word_set_size
        self.device = device

        string: str = None
        with open(text_path) as f:
            string = f.read()
            string = string.lower()

        all_words: list[str] = re.findall("[a-z]+", string)
        self.data: list[int] = list(map(lambda s: word_dict[s], all_words))

        self.sequence_length:int = sequence_length

    def __getitem__(self, index):
        random.seed(index)
        word_id: int = random.randint(0, self.word_set_size-1)
        replace_indx: int = random.randint(0, self.sequence_length-1)

        data_tensor:torch.tensor = torch.tensor(self.data[index : index + self.sequence_length], dtype=torch.long, device=self.device)
        data_tensor[replace_indx] = word_id

        return data_tensor, torch.tensor(replace_indx, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.data)-self.sequence_length-1

def get_word_dict(word_set: set):
    """Given a set containing all the words in the dataset,
    get a correspondence between words and indices and vice versa"""

    #Word dict can be used to get a word from an index and vice versa
    word_dict : dict = {i:word for i, word in enumerate(word_set)} | {word:i for i, word in enumerate(word_set)}

    return word_dict

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", type=int, help="The size of the embedding vectors.", default=100)
    parser.add_argument("-sl", type=int, help="The length of the input sequence.", default=5)
    parser.add_argument("-nh", type=int, help="The number of hidden layers in the feed forward", default=2)
    parser.add_argument("-sh", type=int, help="The size of hidden the layers in the feed forward", default=100)
    parser.add_argument("-mb", type=int, help="The size of the minibatches.", default=32)
    parser.add_argument("-epochs", type=int, help="The number of training epochs.", default=10)
    parser.add_argument("-lr", type=float, help="The learning rate of the moddel.", default=0.01)
    parser.add_argument("-repfreq", type=int, help="How often the progess is deisplayed", default=1000)

    return parser.parse_args()
    
def calculate_accuracy(prediction:torch.tensor, labels:int):
    return sum(prediction.argmax(dim=1) == labels) / labels.numel()

def report(epoch, total_loss, total_acc, n, dev_loader, criterion, model):
    #dev_acc:float = 0
    dev_loss:float = 0
    dev_acc = 0
    dev_n:int = 0
        
    print("Evaluating Dev...")
    for i, (x, y) in enumerate(dev_loader):
        y_pred = model(x)

        with torch.no_grad():
            y_embed = model.embeddings(y)

        dev_loss += criterion(y_pred, y_embed)
        #dev_acc += calculate_accuracy(y_pred, y)
        dev_n    += 1


    print("Epoch: %d, Train Loss %0.4f, Dev Loss %0.4f" % (epoch, total_loss/n, dev_loss/dev_n))


def train(model:Model , train_loader, dev_loader, args, device):
    # define our loss function

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_loss = 0
    total_acc = 0
    n = 0
    for epoch in range(args.epochs):
        for update, (x, y) in enumerate(train_loader):
            # make prediction

            #Finds the embedding of the label without trying to compute gradients for it
            with torch.no_grad():
                y_embed = model.embeddings(y)

            y_pred = model(x)
            loss = criterion(y_pred, y_embed)
            
            total_loss+=float(loss)
            #total_acc += calculate_accuracy(y_pred, y)
            n+=1

            # take gradient step
            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            model.embeddings.clip_embeddings()

            if update%args.repfreq==0:
                report(epoch, total_loss, total_acc, n, dev_loader, criterion, model)
                total_loss = 0
                total_acc = 0
                n = 0

def word_set_from_path(path):
    word_set = None
    with open(path) as f:
        word_set = { *(re.findall("[a-z]+", f.read().lower())) }

    return word_set

def main():
    args = parse_all_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device="cpu"

    print("Getting word set from dataset...")
    word_set  : set  = word_set_from_path("all_data.txt")
    word_dict : dict = get_word_dict(word_set)


    print("\nInitializing test and dev sets...")
    train_set: IdentificationDataset = IdentificationDataset("train_data.txt", args.sl, word_dict, len(word_set), device)
    dev_set: IdentificationDataset = IdentificationDataset("dev_data.txt", args.sl, word_dict, len(word_set), device)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.mb, shuffle=True, num_workers=6)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.mb, shuffle=True)


    print("\nInitializing the model (on device %s)..." % device)
    model = Model(word_dict, len(word_set), args)
    model.to(device)

    print("\nTraining Model...")
    train(model, train_loader, dev_loader, args, device)

    print("\nSaving Model")
    torch.save(model.embeddings, "models/embeddings.pyt")
    torch.save(model.word_predictor, "models/word_predictors.pyt")

    with open('models/word_dict.pickle', 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

##
## Interactive Shell Functions
##

def load_embeddings():
    return torch.load("models/embeddings.pyt")

def load_word_dict():
    word_dict = None
    with open('models/word_dict.pickle', 'rb') as handle:
        word_dict = pickle.load(handle)

    return word_dict

def load():
    global e
    global d
    
    e, d = load_embeddings(), load_word_dict()

def embed(word, embeddings:Embeddings=None, word_dict:dict=None):
    if embeddings is None or word_dict is None:
        embeddings=e; word_dict=d

    return embeddings(word_dict[word])

def nearest_indx(vector:torch.tensor, embeddings:Embeddings=None, word_dict:dict=None):
    if embeddings is None or word_dict is None:
        embeddings=e; word_dict=d
    
    nearest_index: int = 0
    nearest_dist:float = float('inf')
    
    for i, embedding in enumerate(embeddings.embeddings):
        #Compare the current vector to all predictions
        dist = float(torch.linalg.vector_norm(vector-embedding))

        #For all predictions where vec is closer, update the distance and the new index
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_index = i
    
    return nearest_index, nearest_dist

def nearest(vector:torch.tensor, embeddings: Embeddings=None, word_dict:dict=None):
    if embeddings is None or word_dict is None:
        embeddings=e; word_dict=d

    tuples = ( (k, float(torch.sum((embeddings(v)-vector)**2)**0.5)  ) for k, v in word_dict.items() if type(k)==str)
    sorted_embed = min(tuples, key = lambda x: x[1] )

    return sorted_embed

def nearests(vector:torch.tensor, embeddings: Embeddings=None, word_dict:dict=None):
    if embeddings is None or word_dict is None:
        embeddings=e; word_dict=d

    tuples = ( (k, float(torch.sum((embeddings(v)-vector)**2)**0.5 )) for k, v in word_dict.items() if type(k)==str)
    sorted_embed = sorted(tuples, key = lambda x: x[1] )

    return sorted_embed

def show(iter):
    for i in iter:
        print(i)
        input()

e = None
d = None

if __name__ == "__main__":
    main()
