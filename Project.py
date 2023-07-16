import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keyboard

model = tf.keras.models.load_model('handwritten.model')

while (True):
    
        img = cv2.imread(f"Input_image.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        t3_indices = np.argsort(prediction[0])[::-1][:3]
        t3_probabilities = prediction[0][t3_indices]
        t3_probabilities_ = t3_probabilities*100
        print("Top 3 Predictions:")
        for i in range(3):
            digit = t3_indices[i]
            probability = t3_probabilities_[i]
            print(f"Digit: {digit}, Probability: {probability:.4f}")
        col = ['red', 'green', 'blue']
        plt.bar(range(3), t3_probabilities_, tick_label=t3_indices, color = col)
        plt.xlabel('Digit')
        plt.ylabel('Probability (%)')
        plt.title('Top 3 Predictions')
        plt.ylim(0,100)
        plt.show()

        if keyboard.is_pressed('esc') or keyboard.is_pressed('f5'):
            print("Loop terminated.")
            break
        