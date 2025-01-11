from torch.utils.data import TensorDataset,DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.components.model import LSTMModel
import sqlite3
import pickle


def main():
    parser = argparse.ArgumentParser(description="Program to train the model")
    
    # Add required arguments
    parser.add_argument("--db",type=str,required=False,default='Data/hawkeye.db',help="database path")
    parser.add_argument("--table",type=str,required=False,default='data',help="table name")
    parser.add_argument("--num_epochs",type=int,required=False,default=50,help="number of epochs")
    parser.add_argument("--lr",type=float,required=False,default=0.001,help="learning rate")
    parser.add_argument("--limit",type=int,required=False,default=1000,help="limit of data to load")
    parser.add_argument("--batch_size",type=int,required=False,default=100,help="batch size")
    parser.add_argument("--save",type=str,required=False,default='models/model2.pth',help="model save path")
    parser.add_argument("--num_classes",type=int,required=False,default=2,help="number of classes")
    parser.add_argument("--input_size",type=int,required=False,default=51,help="input size")
    parser.add_argument("--offset",type=int,required=False,default=0,help="offset")
    parser.add_argument("--modelpath",type=str,required=False,default=None,help="model path")
    # Parse the arguments
    args = parser.parse_args()

    #connecting to Database
    try:
        connection=sqlite3.connect(args.db)
        cursor=connection.cursor()
    except:
        print("Database not found")
        exit(0)

    limit=args.limit
    offset=args.offset
    currlen=limit

    #check if table exists
    try:
        cursor.execute("SELECT COUNT(*) FROM {}".format(args.table))
        row_count = cursor.fetchone()[0]
    except:
        print("Table not found")
        exit(0)


    #define model
    trainer=Trainer(args.modelpath,args.input_size,args.num_classes,args.lr,args.num_epochs)

    while currlen == limit:
        #loading data
        query = "SELECT x,y FROM {} ORDER BY id ASC LIMIT ? OFFSET ?".format(args.table)
        cursor.execute(query, (limit,offset))
        xy = cursor.fetchall()
        currlen=len(xy)
        offset+=limit

        #process Data
        x= torch.stack([pickle.loads(item[0]) for item in xy])
        x= x.reshape(x.shape[0],20,-1)
        y= torch.tensor([item[1] for item in xy])

        #batching data        
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        #model Traning
        trainer.train(dataloader,offset,row_count)



        #saving model
        print("--------{} COMPLETED----------".format(offset))
        print("--------SAVING MODEL------------")
        try:
            torch.save(trainer.model,args.save)
            print("--------MODEL SAVED-------------")
        except:
            print("-----FAILED TO SAVE MODEL-----")

    connection.close()
    

class Trainer():
    def __init__(self,modelpath,input_size,num_classes,lr,num_epochs):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modelpath is not None:
            self.model = torch.load(modelpath).to(self.device)
        else:
            self.model = LSTMModel(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs


    def train(self,dataloader,offset,row_count):
        for epoch in range(self.num_epochs):
            self.model.train()  
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)    
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)               
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Offset: {offset}/{row_count}")
            

if __name__ == "__main__":
    main()