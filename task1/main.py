from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import preprocess

def create_label(window, txt, font, fontSize, row, column, padx=0, pady=0):
    ttk.Label(window, text=txt, font=(font, fontSize)).grid(row=row, column=column, padx=padx, pady=pady)


def create_combox(window, values, width, row, column, padx, pady):
    feature = ttk.Combobox(window, width=width)
    feature['values'] = values
    feature.grid(row=row, column=column, padx=padx, pady=pady)
    feature.current()
    return feature


def check_features(self, event=None):
    if (feature1.get() == feature2.get()):
        messagebox.showinfo("Error!!", "can't choose same features")
        feature1.current(0)
        feature2.current(1)


def check_classes(self, event=None):
    if (class1.get() == class2.get()):
        messagebox.showinfo("Error!!", "can't choose same classes")
        class1.current(0)
        class2.current(1)



def bias_btn_checked():
    biasbtn = bias_btn.get()
    return biasbtn


def make_Classification():
    if learning_Rate.get() and feature1.get() and feature2.get() and class1.get() and class2.get() and epochs.get():
        accuracy = preprocess.fire(float(learning_Rate.get()), feature1.get(), feature2.get(), class1.get(), class2.get(),
                              int(epochs.get()),
                              bias_btn.get())
        accuracy_value.set(accuracy)
    else:
        messagebox.showinfo('Error !!!', 'missed data founded!!!')
    return "accuracy"


Window = Tk()
Window.title("Task 1 NN")
Window.geometry("830x700")

bg=PhotoImage(file="background.png")
bg_label=Label(Window,image=bg)
bg_label.place(x=0,y=0,relheight=1,relwidth=1)

# selecting the two features



ttk.Label(Window, text="Select Two Features", font=('Helvatical bold', 20)).grid(row=0, column=1,pady=20)


features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
# feature 1
create_label(Window, "feature 1:", 'Helvatical', 15, 1, 0, padx=20, pady=0)
feature1 = create_combox(Window, features, 27, row=1, column=1, padx=0,pady= 0)
feature1.bind("<<ComboboxSelected>>", check_features)

# feature 2
create_label(Window, "feature 2:", 'Helvatical', 15, row=1,column= 2, padx=20, pady=0)
feature2 = create_combox(Window, features, 27, 1, 3, 0, 0)
feature2.bind("<<ComboboxSelected>>", check_features)

# selecting the two classes
create_label(Window, "Select Two Classes :", 'Helvatical bold', 20, 2, 1, padx=0, pady=20)


classes = ['Adelie', 'Gentoo', 'Chinstrap']

# class 1
create_label(Window, "class 1:", 'Helvatical', 15, 3, 0, padx=0, pady=0)
class1 = create_combox(Window, classes, 27,3, 1, 0, 5)
class1.bind("<<ComboboxSelected>>", check_classes)

# class 2
create_label(Window, "class 2:", 'Helvatical', 15, 3, 2, padx=0, pady=0)
class2 = create_combox(Window, classes, 27, 3, 3, 0, 5)
class2.bind("<<ComboboxSelected>>", check_classes)

# getting the learning rate
create_label(Window, "Enter learning rate:", 'Helvatical bold', 20, 4, 1, padx=20, pady=20)
learning_Rate = ttk.Entry(Window, width=25)
learning_Rate.grid(row=4, column=2, padx=0, pady=20)

# getting the epochs
create_label(Window, "Enter number of epochs:", 'Helvatical bold', 20, 5, 1, padx=20, pady=20)
epochs = ttk.Entry(Window, width=25)
epochs.grid(row=5, column=2, padx=0, pady=0)

# check the Bias
create_label(Window, "Bias:", 'Helvatical bold', 20, 6, 1, padx=20, pady=20)
bias_btn = IntVar()
ttk.Checkbutton(Window, text="Bias", variable=bias_btn, command=bias_btn_checked, onvalue=1,
                offvalue=0).grid(row=6, column=2)

# run classifier btn
ttk.Button(Window, text="Run Classifier", width=30, command=make_Classification) \
    .grid(row=7, column=1, padx=0, pady=20)


# accuracy value txt box
create_label(Window, "Accuracy :", 'Helvatical bold', 20, 8, 1, padx=20, pady=20)
accuracy_value = StringVar()
accuracyEntry = ttk.Entry(Window, width=25, textvariable=accuracy_value)
accuracyEntry.config(state='disabled')
accuracyEntry.grid(row=8, column=2, padx=0, pady=0)


Window.mainloop()
