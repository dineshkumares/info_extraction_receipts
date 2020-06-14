import cv2
import os
import shutil


image_path = "../../data/raw/img/"
box_path= "../../data/raw/box/"
figures_path = "../../figures/"


def visualize_textboxes(img_name):
    image = image_path + img_name + '.jpg'
    box = box_path + img_name + '.csv'
    img = cv2.imread(image)
    with open(box) as topo_file:
        for line in topo_file:
            coor = line.split(',')
            print(coor)
            x1,y1,x3,y3 = int(coor[0]),int(coor[1]),int(coor[4]),int(coor[5])
            text = coor[8].strip('\n').strip('\'')
            print(x1,y1,x3,y3,text)

            img = cv2.rectangle(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
            img = cv2.putText(img, text, (x1, y1 - 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(figures_path + img_name + '.jpg', img)
    topo_file.close()
visualize_textboxes('000') 





# def draw():
#     f = open(box)

#     # read each image and its label
#     print(len(f))

#     line = f.readline()
    
#     line_num = 0 
    
#     while line:
#         line_num += 1 
#         img = cv2.imread(image)

#         coor = line.split(',')
#         print(coor)
#         x1 = int(coor[0])
#         y1 = int(coor[1])
#         x3 = int(coor[4])
#         y3 = int(coor[5])
#         text = coor[8].strip('\n').strip('\'')
#         print(x1,y1,x3,y3,text)

#         img = cv2.rectangle(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
#         img = cv2.putText(img, text, (x1, y1 - 1),
#                     cv2.FONT_HERSHEY_TRIPLEX, 0.35, (0, 0, 255), 1)

#         # cv2.imwrite(box, img)
#         # line = f.readline()
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# draw() 