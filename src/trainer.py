#Consists of training loop and user menu for selecting training mode and dataset
from datasets_path import *
from dataloader import *

print("HEllO")
df = input("PLEASE ENTER THE DATASET YOU WANT TO WORK WITH:")

if df == "dsc":
    setpath = DSC_setpath()
    path = setpath.set_path()
    print("CURRENT PATH", path)