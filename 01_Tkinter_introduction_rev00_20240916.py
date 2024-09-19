#
# TITLE: Tkinter (introduction)
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=0zrq2N2qAVo&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4
#
#


# modules
from tkinter import *
from PIL import ImageTk, Image

# creating window
root = Tk()

# title
root.title('FEM')

# window size
root.geometry('1800x800')

# labels
label1 = Label(root, text='Finite Element Method', font=('Verdana, 20'))
label1.grid(row=0, column=0, padx=40, pady=40)

label2 = Label(root, text='How to write script')
label2.grid(row=1, column=0, padx=40, pady=40)

# frame
frame = LabelFrame(root)
frame.grid(row=0, column=1, padx=40, pady=40, ipadx=80, ipady=80)

label1_frame = Label(frame, text='Finite Element Method', font=('Verdana, 20'))
label1_frame.grid(row=0, column=0, padx=40, pady=40)

# button
n = 1

def button_print(n):
    print(str(n) + ' > This button works')

button1 = Button(root, text='Button', command=lambda: button_print(n),
                 font=('Verdana', 35), fg='blue', bg='black', activebackground='red')
button1.grid(row=2, column=0)

# entry box
entry = Entry(root, width=20, font=('Verdana', 30), justify='center', state='readonly')
entry.grid(row=0, column=2)

entry.configure(state='normal')
entry.insert(0, '23')
entry.configure(state='readonly')

value = entry.get()
print(value)

# check box

def check_function(var):
    print(var)

varCheck = StringVar()
varCheck.set('on')

checkbox = Checkbutton(root, text='switch', onvalue='on', offvalue='off',
                       variable=varCheck, command=lambda: check_function(varCheck.get()),
                       font=('Verdana', 30))
checkbox.grid(row=1, column=1)

# sliders

def scaler(val):
    global slider
    global sliderentry

    sliderentry.configure(state='normal')
    sliderentry.delete(0, END)
    sliderentry.insert(0, f"{slider.get():.2f}")
    sliderentry.configure(state='readonly')

sliderframe = LabelFrame(root)
sliderframe.grid(row=0, column=3, padx=30, pady=30)

slider = Scale(sliderframe, from_=-20, to=20, orient='horizontal',
               width=20, length=200, showvalue=0,
               command=scaler)
slider.grid(row=0, column=0)

sliderentry = Entry(sliderframe, width=5, borderwidth=5, font=('Verdana', 20),
                    justify='center',state='readonly')
sliderentry.grid(row=1, column=0)

sliderentry.configure(state='normal')
sliderentry.insert(0, 1.00)
sliderentry.configure(state='readonly')

# radiobuttons

radioframe = LabelFrame(root)
radioframe.grid(row=2, column=1, padx=30, pady=30)

colormapVar = StringVar()
colormapVar.set('jet')

radioJet = Radiobutton(radioframe, variable=colormapVar,text='Jet', value='jet', font=('Verdana', 10))
radioCopper = Radiobutton(radioframe, variable=colormapVar,text='Copper', value='copper', font=('Verdana', 10))
radioCool = Radiobutton(radioframe, variable=colormapVar,text='Cool', value='cool', font=('Verdana', 10))

radioJet.grid(row=0, column=0)
radioCopper.grid(row=1, column=0)
radioCool.grid(row=2, column=0)

imageJet = Image.new('RGB', (100, 100), color=(0,0,200))
imageCopper = Image.new('RGB', (100, 100), color=(150, 0, 0))
imageCool = Image.new('RGB', (100, 100), color=(0,100,100))

imageJet = ImageTk.PhotoImage(imageJet.resize((100, 20)))
imageCopper = ImageTk.PhotoImage(imageCopper.resize((100, 20)))
imageCool = ImageTk.PhotoImage(imageCool.resize((100, 20)))

radioJet.configure(image=imageJet)
radioCopper.configure(image=imageCopper)
radioCool.configure(image=imageCool)

#
root.mainloop()

