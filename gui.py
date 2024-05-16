from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import re
import pickle
import numpy as np
import joblib
import pandas as pd


_scaler=joblib.load(open('Model/scaler.pkl','rb'))
# Load the encoder
nb_model = joblib.load('Model/naive_bayes_model.pkl')

#window creation
a = Tk()
a.title("PCOS Detection System")
a.geometry("1000x500")
a.minsize(1000,500)
a.maxsize(1000,500)
a.iconbitmap('Icons/my_icon.ico')

def prediction():
    
    input_values=text1.get("1.0",'end')
    print("input_values : ",input_values)
    if input_values=='' or input_values=='\n':
        message.set("fill the empty field!!!")
    else:
        message.set("")
        list1=input_values.split(",")
        print(list1)
        values=[float(x)for x in list1]
        print(values)

        # Define the input values as a list
        list_box.insert("end", "Load Values\n")
        list_box.insert("end", values)
        list_box.insert("end", "\n")

        # Define the column names as a list
        column_names = ['Vit D3 (ng/mL)', 'I beta-HCG(mIU/mL)', 'LH(mIU/mL)', 'FSH(mIU/mL)', 'Follicle No. (R)', 'Follicle No. (L)', 
                        'AMH(ng/mL)', 'FSH/LH', 'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Weight (Kg)', 
                        'Fast food (Y/N)', 'Cycle(R/I)', 'PRG(ng/mL)', 'Pimples(Y/N)', 'Marraige Status (Yrs)', 'Age (yrs)', 
                        'II beta-HCG(mIU/mL)', 'BMI']


        input_data = pd.DataFrame([values], columns=column_names)
        list_box.insert("end", "Generate Dataframe\n")
        list_box.insert("end", input_data)
        list_box.insert("end", "\n")
        scaled_input = _scaler.transform(input_data)
        list_box.insert("end", "Loading Standardscaler\n")
        list_box.insert("end", scaled_input)
        list_box.insert("end", "\n")
        # Perform prediction
        prediction = nb_model.predict(scaled_input)
        list_box.insert("end", "Loading Naive Bayes Model\n")
        list_box.insert("end", prediction)
        list_box.insert("end", "\n")
        # Print the predicted value
        print("Predicted label:", prediction[0])
        pred=prediction[0]
        print(pred)
        list_box.insert("end", "Predicted label")
        list_box.insert("end", pred)
        list_box.insert("end", "\n")

        if pred == 1:
            print("PCOS Detected.")
            output="PCOS Detected"
            list_box.insert("end", "Result")
            list_box.insert("end", output)
            list_box.insert("end", "\n")
        else:
            print("No PCOS")
            output="No PCOS"
            list_box.insert("end", "Result")
            list_box.insert("end", output)
            list_box.insert("end", "\n")

        out_label.config(text=output)



def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#d8bcab")
    f1.place(x=0, y=0, width=760, height=250)
    f1.config()

    input_label = Label(f1, text="Enter Test Data", font="arial 16", bg="#d8bcab")
    input_label.pack(padx=0, pady=10)

    
    global message
    message = StringVar()

    global text1
    text1=Text(f1,height=8,width=70)
    text1.pack()


    msg_label = Label(f1, text=
        "", textvariable=message,
                      bg='#d8bcab').place(x=330, y=185)

    predict_button = Button(
        f1, text="Prediction", command=prediction, bg="#d4e09b")
    predict_button.pack(side="bottom", pady=16)
    global f2
    f2 = Frame(f, bg="#a0ecd0")
    f2.place(x=0, y=250, width=760, height=500)
    f2.config(pady=20)

    result_label = Label(f2, text="Prediction Result", font="arial 16", bg="#a0ecd0")
    result_label.pack(padx=0, pady=0)

    global out_label
    out_label = Label(f2, text="", bg="#a0ecd0", font="arial 16")
    out_label.pack(pady=70)

    f3 = Frame(f, bg="#3a606e")
    f3.place(x=760, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Backend Process", font="arial 14",fg="white", bg="#3a606e")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()



def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="light goldenrod")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Icons/home.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="PCOS Detection System",
                       font="arial 35", bg="light goldenrod")
    home_label.place(x=240, y=200)


f = Frame(a, bg="light goldenrod")
f.pack(side="top", fill="both", expand=True)

front_image1 = Image.open("Icons/home.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((1000,650), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="PCOS Detection System",
                   font="arial 35", bg="light goldenrod")
home_label.place(x=240, y=200)

m = Menu(a)
m.add_command(label="Homepage", command=Home)
checkmenu = Menu(m)
m.add_command(label="Testpage", command=Check)

plotmenu=Menu(m)
a.config(menu=m)


a.mainloop()
