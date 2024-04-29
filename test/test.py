import requests

# Specify the absolute path to your image file
image_path = 'C:\Users\giftc\mydlproject\static\0530_1492626047222176976_0.jpg'

# Open the file with the absolute path
with open(image_path, 'rb') as img:

    resp = requests.post("http://localhost:5000/predict", files={'file': img})

print(resp.text)
