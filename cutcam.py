from PIL import Image

import matplotlib.pyplot as plt
for i in range(1,1226):
    img = Image.open('./test/img (%d).jpg'%i)

    Img2=img.resize((224,224))
    Img2.save("./225/%04d.jpg"%i)


