import cv2
import sys
import os
import pathlib
from sklearn.model_selection import train_test_split
import shutil

mode = ""

ix,iy = -1,-1
colors = {'green': (0, 255, 0), 'red': (0, 0, 255)}
xmin_vals_rectangle = []
ymin_vals_rectangle = []
xmax_vals_rectangle = []
ymax_vals_rectangle = []

x_vals_point = []
y_vals_point = []

def option(event, x, y, flags, param):
    if (mode == "rectangle"):
        draw_rectangle(event, x, y, flags, param)
    elif (mode == "point"):
        draw_point(event, x, y, flags, param)

def draw_rectangle(event, x, y, flags, param):
    global ix,iy

    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        xmin_vals_rectangle.append(ix)
        ymin_vals_rectangle.append(iy)
    elif event == cv2.EVENT_LBUTTONUP:
        xmax_vals_rectangle.append(x)
        ymax_vals_rectangle.append(y)
        cv2.rectangle(image,(ix,iy),(x,y), colors['red'], 2)
        print("Do a rectangle or press 'echap' for next image...\n")

def draw_point(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image, (x, y), 5, colors['green'], -1)
        x_vals_point.append(x)
        y_vals_point.append(y)
        print("Double click another pixel or press 'echap' for next image...\n")

if __name__ == "__main__":

    folder = sys.argv[1]
    print("Welcome to the Image Annotation Program!\n")
    val = input("Name the folder to retrieve the annotations: ")
    os.makedirs(val)
    print("Press echap for next image\n")
    print("Choose your mode: press p to annotate a point or press r to annotate a rectangle")

    for (folder, subfolder, files) in os.walk(folder):
        size = len(files)
        for file in files:
            xmin_vals_rectangle = []
            ymin_vals_rectangle = []
            xmax_vals_rectangle = []
            ymax_vals_rectangle = []
            image = cv2.imread(os.path.join(folder, file), -1)
            # image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            height = image.shape[0]
            width = image.shape[1]
            cv2.namedWindow('Annotation')
            cv2.setMouseCallback('Annotation', option)

            while True:
                cv2.imshow('Annotation', image)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('p'):
                    print("point mode selected")
                    print("Double click anywhere inside the image to annotate that point...\n")
                    mode = "point"
                elif k == ord('r'):
                    print("rectangle mode selected")
                    mode = "rectangle"
                elif k == 27:
                    fname = os.path.join(val, pathlib.Path(file).stem + '.txt')
                    with open(fname, "w") as f:
                        print("image: ", size, "/", len(files))
                        size -= 1
                        for i, el in enumerate(xmin_vals_rectangle):
                            x_center = ((xmax_vals_rectangle[i] + xmin_vals_rectangle[i]) / 2) / width
                            y_center = ((ymax_vals_rectangle[i] + ymin_vals_rectangle[i]) / 2) / height
                            width_image = (xmax_vals_rectangle[i] - xmin_vals_rectangle[i]) / width
                            height_image = (ymax_vals_rectangle[i] - ymin_vals_rectangle[i]) / height
                            f.write(f"0 {x_center} {y_center} {width_image} {height_image}\n")
                    break
            cv2.destroyAllWindows()

    images = folder
    annotations = val

    def move_files_to_folder(list_of_files, destination_folder):
        for f in list_of_files:
            try:
                shutil.move(f, destination_folder)
            except:
                print(f)
                assert False
            
    imgs = [os.path.join(images, x) for x in os.listdir(images)]
    annots = [os.path.join(annotations, x) for x in os.listdir(annotations) if x[-3:] == "txt"]

    imgs.sort()
    annots.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(imgs, annots, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    if not os.path.exists(images + '/train'):
        os.makedirs(images + '/train')
    if not os.path.exists(images + '/val'):
        os.makedirs(images + '/val')
    if not os.path.exists(images + '/test'):
        os.makedirs(images + '/test')
    if not os.path.exists(annotations + '/train'):
        os.makedirs(annotations + '/train')
    if not os.path.exists(annotations + '/val'):
        os.makedirs(annotations + '/val')
    if not os.path.exists(annotations + '/test'):
        os.makedirs(annotations + '/test')
    move_files_to_folder(train_images, images + '/train')
    move_files_to_folder(val_images, images + '/val')
    move_files_to_folder(test_images, images + '/test')
    move_files_to_folder(train_annotations, annotations + '/train')
    move_files_to_folder(val_annotations, annotations + '/val')
    move_files_to_folder(test_annotations, annotations + '/test')
    if not os.path.exists('detect_data'):
        os.makedirs('detect_data')
    shutil.move(images, "./detect_data/images")
    shutil.move(annotations, "./detect_data/labels")