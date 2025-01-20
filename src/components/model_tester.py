from torch.utils.data import TensorDataset,DataLoader
import argparse
import torch
from src.components.model import LSTMModel
import sqlite3
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle


def main():
    
    parser = argparse.ArgumentParser(description="Program To Test the model performance")
    
    # Add required arguments
    parser.add_argument("--db",type=str,required=False,default='Data/hawkeye.db',help="database path")
    parser.add_argument("--table",type=str,required=False,default='data',help="table name")
    parser.add_argument("--limit",type=int,required=False,default=1000,help="limit of data to load")
    parser.add_argument("--offset",type=int,required=False,default=0,help="offset")
    parser.add_argument("--modelpath",type=str,required=True,help="model path")
    parser.add_argument("--batch_size",type=int,required=False,default=100,help="batch size")
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
    tester=Tester(args.modelpath)

    while currlen == limit:
        #loading data
        query = "SELECT x,y FROM {} ORDER BY id ASC LIMIT ? OFFSET ?".format(args.table)
        cursor.execute(query, (limit,offset))
        xy = cursor.fetchall()
        currlen=len(xy)
        offset+=currlen

        #process Data
        x= torch.stack([pickle.loads(item[0]) for item in xy])
        x= x.reshape(x.shape[0],20,-1)
        y= torch.tensor([item[1] for item in xy])

        #batching data        
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        #model Testing
        tester.test(dataloader,offset,row_count)


    connection.close()
    

class Tester():
    def __init__(self,modelpath):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = torch.load(modelpath).to(self.device)
      
      
    def test(self,dataloader,offset,row_count):
        
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            all_predictions = []
            all_labels = []

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                label = batch_y

                total_correct += (predictions == label).sum().item()
                total_samples += label.size(0)

                # Collect all predictions and labels for metric calculations
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            # Calculate accuracy
            accuracy = total_correct / total_samples

            # Calculate other metrics
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            # Print metrics
            print("---------------------Model Performance---------------------")
            print("FOR UPTO: {}/{}".format(offset,row_count))
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print("----------------------------------------------------------")

   
                

if __name__ == "__main__":
    main()