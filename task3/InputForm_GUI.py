from tkinter import *
import preprocess as pre

myWindow  = Tk()
myWindow.title("INPUT DATA OF MODEL")
myWindow.geometry("700x600")
myWindow.colormapwindows



Label(text="Hidden Layers ?",font=23).place(x=99,y=99)
hiddenLayers = IntVar()
headenLayerEntry  = Entry(myWindow,textvariable=hiddenLayers,width=9,bd=3,font=21)
headenLayerEntry.place(x=299,y=99)

Label(text="Number of Neurons ?",font=22).place(x=99,y=149)
neurons = StringVar()
NeuronsEntry  = Entry(myWindow,textvariable=neurons,width=9,bd=3,font=21)
NeuronsEntry.place(x=299,y=149)
Label(text="Learning rate",font=22).place(x=99,y=199)
Lr = DoubleVar()
LrEntry  = Entry(myWindow,textvariable=Lr,width=10,bd=2,font=20)
LrEntry.place(x=299,y=199)
Label(text="accuracy ", font=23).place(x=249, y=414)
accuracy = StringVar()
Accuracy  = Entry(myWindow,textvariable=accuracy,width=10,bd=2,font=20)
Accuracy.place(x=350,y=415)
Label(text="Number Of Itration",font=22).place(x=99,y=249)
epochs = IntVar()
EpochsEntry  = Entry(myWindow,textvariable=epochs,width=10,bd=2,font=20)
EpochsEntry.place(x=300,y=250)
Label(myWindow,text="Please Fill The Data Below",font="arial 17").pack(pady=50)
Label(text="Activation Function",font=23).place(x=99,y=349)
activation = IntVar()
sigmoid_choose = Radiobutton(myWindow,text="Sigmoid",variable=activation,value=1)
sigmoid_choose.place(x=319,y=349)
tanh_choose = Radiobutton(myWindow,text="Tanh",variable=activation,value=2)
tanh_choose.place(x=399,y=349)
Biased = BooleanVar()
BiasBox  = Checkbutton(text="Bias ?",variable=Biased,font=19)
BiasBox.place(x=99,y=299)

Bounes = BooleanVar()
Bounesbox  = Checkbutton(text="Bounes ?",variable=Bounes,font=19)
Bounesbox.place(x=200,y=299)

def converTotString(string):
    listt = list(string.split(","))
    listt = [int(x) for x in listt]
    return listt


def getInputs():


    neuronsList = converTotString(neurons.get())
    data = pre.fire(learning_rate=Lr.get(),hide_num=hiddenLayers.get(),nurans=neuronsList,epochs=epochs.get(),bais=Biased.get(),activation=activation.get(),Bounes=Bounes.get())
    acc = data[0]
    accuracy.set((str(round(acc)))+'%')
    print(f'Accuracy Test: {acc}%')
    print('=============================================================')



Button(text="Run",font=19,width=11,height=2,command=getInputs).place(x=99,y=399)




myWindow.mainloop()
