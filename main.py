
from utils import *


class MainApp(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
#        self.statusbar = Statusbar(self, ...)
        self.imagedisp = ImageDisp(self)
        self.imagedisp.pack()
        self.pack(fill='both', expand=True)


if __name__ == '__main__':
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

