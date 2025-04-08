from body_combined import train_combined
from body_combined import predict_combined
from text_encoder import run_llm
from gnn_encoder import run_supervised
import torch   

if __name__ == "__main__":
    import os
    os.chdir(".....") # Change to the directory of the project
    #Converting cif to txt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    print("Training started")
    # run_llm(device)
    # run_supervised()
    # train_combined()
    print("Training completed successfully")
    predict_combined()
