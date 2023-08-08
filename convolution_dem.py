import numpy as np
import matplotlib.pyplot as plt

stride = 2


def apply_conv(x, y, image, filt):
    sumr = 0
    sumg = 0
    sumb = 0


    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= x + i < image.shape[0] and 0 <= y + j < image.shape[1]:
                factor = filt[i,j]
                sumr += (image[x + i, y + j][0]) * factor
                sumg += (image[x + i, y + j][1]) * factor
                sumb += (image[x + i, y + j][2]) * factor

    return np.array((sumr, sumg, sumb))


def filtered_image(dx,dy, image, filt):
    n_im = []
    for i in range(dx):
        for j in range(dy):
            n_im.append(apply_conv(i+1, j+1, image, filt))

    n_im = np.array(n_im).reshape(dx, dy, 3)
    return n_im


def max_pooled_image(dx,dy, image):
    n_im = []
    for i in range(0,image.shape[0],stride):
        for j in range(0,image.shape[1], stride):
            n_im.append(pooling(i, j, image))

    n_im = np.array(n_im).reshape(dx, dy, 3)
    return n_im


def pooling(x, y, image, pool_size=stride):
    pool_r = []
    pool_g = []
    pool_b = []

    for i in range(pool_size):
        for j in range(pool_size):
            pool_r.append(image[x + i, y + j][0])
            pool_g.append(image[x + i, y + j][1])
            pool_b.append(image[x + i, y + j][2])

    max_r = max(pool_r)
    max_g = max(pool_g)
    max_b = max(pool_b)

    return np.array((max_r, max_g, max_b))


def main():
    image = 255 - np.load("cat.npy")[100]
    image = image.reshape(28, 28)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    filt = np.array([[-1, -1, -1],
                      [1, 1, 1],
                      [0, 0, 0]])
    filt = filt.T


    filtered_img = filtered_image(26,26,image, filt).reshape(26, 26, 3)


    pooled_img = max_pooled_image(26//stride,26//stride,filtered_img).reshape(26//stride,26//stride, 3)


    fig, axs = plt.subplots(1, 3)


    axs[0].imshow(image)
    axs[0].set_title('Image')

    axs[1].imshow(filtered_img)
    axs[1].set_title('Filtered image')


    axs[2].imshow(pooled_img)
    axs[2].set_title('Pooled image')

    plt.show()




if __name__ == "__main__":
    main()
