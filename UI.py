import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model  

model = load_model('fruit_prediction_transfer.h5')

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

        img = cv2.imread(file_path)
        img = cv2.resize(img, (200, 200))  
        img = np.expand_dims(img, axis=0)  
        img = img / 255.0 
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        output = {0: 'manzana', 1: 'banana', 2: 'mixto', 3: 'naranja'}
        prediction_label.config(text=f"Es una: {output[predicted_class]}")
        print(output[predicted_class])

app = tk.Tk()
app.title("Reconocimiento de frutas")

button = tk.Button(app, text="Predecir imagen", command=open_image)
button.pack(pady=10)

label = tk.Label(app)
label.pack()

prediction_label = tk.Label(app, text="")
prediction_label.pack()

app.mainloop()