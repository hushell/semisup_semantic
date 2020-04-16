import os
import pandas as pd
import numpy as np
import torch.utils.data as data
from PIL import Image

class M2NISTDataset(data.Dataset):
    heightSize = 64
    widthSize = 64
    n_classes = 11
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]

    def __init__(self, root, opt):
        super(M2NISTDataset, self).__init__()

        image_set = 'train' if opt.isTrain else 'test'
        path = os.path.join(root, image_set)
        self.path = path

        if not os.path.exists(path + '/combined.npy'):
            self.generate(root, image_set)

        X = np.load(path + '/combined.npy')
        y = np.load(path + '/segmented.npy')

        X = X.astype(np.float32)
        y = y.astype(np.int32)

        # Normalize
        X -= 127.0
        X /= 127.0

        self.X, self.y = X, y

        # sup indices
        totNum = self.__len__()
        if (opt.sup_portion > 0 and opt.sup_portion <= 1):
            while True:
                self.sup_indices = np.random.randint(0, totNum, int(opt.sup_portion * totNum))
                yy = y[self.sup_indices]
                if yy.sum(axis=(0,1,2)).all():
                    break
        elif opt.sup_portion == 0:
            self.sup_indices = []
        else:
            # sup_portion = 0, 1, ..., 10
            self.sup_indices = np.concatenate([np.arange(i,totNum,10)
                                               for i in range(opt.sup_portion)])

        print('==> supervised portion = %.3f' % (float(len(self.sup_indices)) / totNum))

        # visualization
        self.label2color = np.array([(128, 0, 0),
                                     (192, 192, 128),
                                     (128, 64, 128),
                                     (0, 0, 192),
                                     (128, 128, 0),
                                     (192, 128, 128),
                                     (64, 64, 128),
                                     (64, 0, 128),
                                     (64, 64, 0),
                                     (0, 128, 192),
                                     (0, 0, 0)])

        self.label2name = np.array(['0', '1', '2', '3', '4',
                                    '5', '6', '7', '8', '9'])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.X[index], self.y[index]

        #img = img[np.newaxis, :]
        img = img * 0.5 + 0.5
        img = np.stack([img, img, img], axis=0)

        ## doing this so that it is consistent with all other datasets
        ## to return a PIL Image
        #img = Image.fromarray(img, mode='L')

        #if self.transform is not None:
        #    img = self.transform(img)

        #target = torch.cat([target, torch.zeros(target.shape[0], target.shape[1], 1)], -1)
        #target = target.argmax(axis=-1)
        target = np.concatenate((target, np.ones((target.shape[0], target.shape[1], 1))), axis=-1)
        target = target.argmax(axis=-1)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        return {'A': img, 'B': target, 'issup': True if index in self.sup_indices else False}

    def __len__(self):
        return len(self.y)

    def name(self):
        return 'M2NISTDataset'

    def get_next_batch(self, batch_size):
        for start in range(0,len(self.y),batch_size):
            end = min(len(self.y), start+batch_size)
            yield self.X[start:end] , self.y[start:end]

    def generate(self, root, image_set):
        training_file = 'train.csv'
        test_file = 'test.csv'
        data = pd.read_csv(os.path.join(self.path, training_file)).as_matrix()

        img_shape=(64, 64)
        dataset_sz = 5000
        max_digits_per_image = 2

        np.random.seed(1234)

        def combine(single_images, max_digits_per_image, canvas_sz):
            # Number (random) of digits to be combined between 1 and max_digits_per_image.
            nb_digits = np.random.randint(low=2,high=max_digits_per_image+1)
            # Indices (random) of digit images to be combined.
            rand_indices = np.random.randint(0,len(single_images),nb_digits)

            src_images = single_images[rand_indices,1:]
            src_labels = single_images[rand_indices,0 ]

            # Segmented output image once channel per digit.
            labels  = np.zeros([*canvas_sz,10],dtype=single_images.dtype)

            for i in range(nb_digits):
                x_off_start = np.random.randint(i*28,i*28+10)
                y_off_start = np.random.randint(0,canvas_sz[0]-28+1)

                x_off_end = x_off_start + 28
                y_off_end = y_off_start + 28

                if x_off_end <= canvas_sz[1] and y_off_end <= canvas_sz[0]:
                    src_img = src_images[i].reshape([28,28])
                    src_digit = src_labels[i]
                    labels[y_off_start:y_off_end, x_off_start:x_off_end,src_digit] = src_img

            canvas = np.max(labels, axis=2)
            labels = np.clip(labels,a_min=0,a_max=1)

            return canvas, labels

        combined = []
        segmented = []

        for i in range(dataset_sz):
            img, segments = combine(data, max_digits_per_image, img_shape)
            combined.append(img)
            segmented.append(segments)

        os.makedirs(os.path.join(root, image_set), exist_ok=True)
        np.save(os.path.join(root, image_set, 'combined.npy'), combined)
        np.save(os.path.join(root, image_set, 'segmented.npy'), segmented)

