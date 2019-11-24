import numpy as np
from glob import glob
from PIL import Image

def one_hot_encode(y, num_classes=101):
    return np.squeeze(np.eye(num_classes)[y.reshape(-1)])


class DataLoader:
    def __init__(self, img_res=(224, 224)):
        self.img_res = img_res

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "indicator_data"
        path = glob('./%s/*' % (225))

        self.n_batches = int(len(path) / batch_size) #計算要幾批的batches才能把圖片load完

        n_batches = int(len(path) / batch_size)  # 計算要幾批的batches才能把圖片load完
        temp = one_hot_encode((np.load("./indicator_data/rg.npy")[:-1]).astype(int)).tolist()


        for i in range(n_batches):
            batch = path[i * batch_size:(i + 1) * batch_size]  # 0~(batch_size*1)-1，batch_size*1~(batch_size*2)-1......
            imgs_A = []
            imgs_B = temp[i * batch_size:(i + 1) * batch_size]
            for img in batch:
                img_A = Image.open(img).convert("L")
                img_A = img_A.resize((100, 100))
                img_A = np.reshape(img_A, (100 , 100 , 1))
                imgs_A.append(img_A)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "indicator_data"
        path = glob('./%s/*' % (225))
        indices = (len(path) * np.random.rand(batch_size)).astype(int)
        temp = one_hot_encode(np.load("./indicator_data/rg.npy").astype(int)).tolist()
        imgs_A = []
        imgs_B = []
        name = []

        for i in indices:
            img_A = Image.open(path[i]).convert("L")
            img_A = img_A.resize((100, 100))
            img_A = np.reshape(img_A, (100, 100, 1))

            imgs_A.append(img_A)
            imgs_B.append(temp[i])
            name.append(path[i][6:])

        name = np.array(name)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        return name, imgs_A, imgs_B

def load_test_data(batch_size=1):
    data_type = "indicator_data"
    path = glob('./%s/*' % (225))
    indices=[]
    for i in range(1225):
        indices.append(i)
    indices=np.array(indices)
    print(indices)
    imgs_A = []

    for i in indices:
        img_A = Image.open(path[i]).convert("L")
        img_A = img_A.resize((100, 100))
        img_A = np.reshape(img_A, (100, 100, 1))
        # img_A = np.expand_dims(img_A, axis=2)
        imgs_A.append(img_A)

    imgs_A = np.array(imgs_A)

    return imgs_A


# if __name__ == '__main__':
#     loading = DataLoader("dataset1", img_res=(150, 300))
#     imgs_A, imgs_B=loading.load_data(4, is_testing=True)