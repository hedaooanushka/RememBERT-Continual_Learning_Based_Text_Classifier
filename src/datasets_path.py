import tkinter
from tkinter.filedialog import askdirectory



class ASC_setpath:
    def __init__(self) -> None:
        self.ascpath = None
        self.root = tkinter.Tk()

    def set_path(self):
        self.ascpath = askdirectory(parent=self.root)
        
    def get_path(self):
        return self.ascpath

class DSC_setpath:
    def __init__(self) -> None:
        self.dscpath = None
        self.root = tkinter.Tk()
    
    def set_path(self):
        self.dscpath = askdirectory(parent=self.root)
        return self.dscpath
    
    def get_path(self):
        return self.dscpath