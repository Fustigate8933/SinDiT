import matplotlib.pyplot as plt
import cv2
def showimgs(*args):
    for i in range(0, len(args), 2):
        cv2.imshow(args[i], args[i + 1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showrgb(img, figsize=(5, 3), save=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off");
    if save is not None:
        plt.savefig(f"{save}.jpg")
    plt.show()