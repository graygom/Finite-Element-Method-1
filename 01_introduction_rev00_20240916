#
# TITLE: Tkinter (introduction)
# AUTHOR: Hyunseung Yoo
# PURPOSE:  
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=0zrq2N2qAVo&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4
#
#


from tkinter import *


# creating window
root = Tk()

# title
root.title('FEM')

# window size
root.geometry('1600x800')

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



#
root.mainloop()


