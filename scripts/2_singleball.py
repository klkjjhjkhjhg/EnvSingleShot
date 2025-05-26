import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

class detcircles:
    def __init__(self,img_path):
        self.img_path = img_path
    def detect_and_select_circles(self,img_path, circleFile):
        img_ori = cv2.imread(img_path)
        H, W = img_ori.shape[:2]
        if H > 2000:
            img_ori = cv2.resize(img_ori, dsize=(0,0), fx=0.5, fy=0.5)
        H, W = img_ori.shape[:2]
        img_org = img_ori.copy()
        img_orgi = img_ori.copy()
        img = cv2.cvtColor(img_ori ,cv2.COLOR_BGR2GRAY)
        plt.rcParams["figure.figsize"] = (16,9)
        plt.imshow(img,cmap='gray')
        #将模糊应用在图片编辑上，结果会标定的更清晰，边缘平滑，减少噪声影响
        img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
        plt.rcParams["figure.figsize"] = (16,9)
        plt.imshow(img,cmap='gray')
        if os.path.exists(circleFile):
            return 0, 0
        os.makedirs('tmp/balldetect',exist_ok=True)
        #是在MirrorBall_imgs1文件夹下换不同的镜球图片在这里更改名字即可
        param1, param2 = 50, 30
        minR = int(H/30)
        maxR = int(H/10)
        dis = 100
        while True:
            all_circs = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, dis,param1 = param1 , param2 = param2, minRadius=minR, maxRadius=maxR)

            # 边缘检测
            edges = cv2.Canny(img, param1//2, param1)
            cy, cx = H/2.0, W/2.0
            # check if its radius is smaller than 1/6 of imgH and center is close to center
            if all_circs is None:
                print('No circle detected!')
                param1, param2, minR, maxR, dis = old
                continue
            all_circs = np.squeeze(all_circs, axis=0)
            select2 = (all_circs[:, 0] < (cx+W*0.2)) * (all_circs[:, 0] > (cx-W*0.2))
            select3 = (all_circs[:, 1] < (cy+H*0.2)) * (all_circs[:, 1] > (cy-H*0.2))
            select = select2 * select3
            select_circs = all_circs[select]
            img_orgi = img_ori.copy()
            for j, circ in enumerate(all_circs):
                cv2.putText(img_orgi, text=str(j), org=(int(circ[0]-circ[2]), int(circ[1]-circ[2])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,0,255), thickness=2)
                cv2.circle(img_orgi, (int(circ[0]), int(circ[1])), int(circ[2]), (200, 0, 0), 2)
            plt.imshow(img_orgi[:,:,::-1])
            plt.savefig('tmp/balldetect/markres')
            cv2.imwrite('tmp/balldetect/edges.png', edges)
            print(f'Current params: {param1}, {param2}, {minR}, {maxR}, {dis}. Insert which circle to accept, if unacceptable, input parameters')
            params = input().strip().split()
            if len(params) == 1:
                decision = int(params[0])
            else:
                old = (param1, param2, minR, maxR, dis)
                param1, param2, minR, maxR, dis = list(map(int, params))
                continue
            select_circs = all_circs[int(decision)][np.newaxis,...]
            if select_circs.shape[0]<1:
                print(' Ball detection failed!')
            elif select_circs.shape[0]>1:
                print(' Find two circles near the center!')
            else:
                print(' Find one circle in the center!')
                coor = select_circs[0] # x, y, radius
                top = int(coor[1]-coor[2])
                bottom = int(coor[1]+coor[2])
                left = int(coor[0]-coor[2])
                right = int(coor[0]+coor[2])
                print('Top:', top, ' Bottom: ', bottom, end=' ')
                print('Left:', left, ' Right: ', right)
                circle = np.uint16(np.around(coor))
                cv2.circle(img_org, (int(coor[0]), int(coor[1])), int(coor[2]), (200, 0, 0), 5)
                print('Insert y if accept this circle estimation, otherwise input anything:')
                decision = input()
                if decision == 'y':
                    with open(circleFile, 'a') as locFile:
                        locFile.write('{} {} {} {} {}\n'.format(1, top, bottom, left, right))
                    break
        return H, W
    
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0:  # 为了避免除以零的错误
            return v
        return v / norm

    def get_extrinsic_matrix(self,target,origin,up, txt_path, coord_type='mitsuba'):
        target = np.array(target)
        origin = np.array(origin)
        up = detcircle.normalize(np.array(up))
        new_z = detcircle.normalize(target-origin)
        if (new_z == up).all() or (new_z == -up).all():
            raise ValueError('new_z and up can not be the same')
        if coord_type == 'opencv':
            new_x = detcircle.normalize(np.cross(new_z,up))
        elif coord_type == 'mitsuba':
            new_x = detcircle.normalize(np.cross(up,new_z))
        elif coord_type == 'blender':
            new_z = -new_z
            new_x = detcircle.normalize(np.cross(up,new_z))
        else:
            raise ValueError('coord_type must be "opencv", "mitsuba" or "blender"')
        new_y = detcircle.normalize(np.cross(new_z,new_x))
        extrinsic = np.zeros([3,4])
        R = np.zeros([3,3])
        R[0,0:3] = new_x
        R[1,0:3] = new_y
        R[2,0:3] = new_z
        t = -np.dot(R,origin)
        extrinsic[:,0:3] = R
        extrinsic[:,3] = t
        q = detcircle.rotation_matrix_to_quaternion(R)
        with open(txt_path, 'w') as file:
            file.write("# Image list with two lines of data per image: \n")
            file.write("IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n")
            file.write("POINTS2D[] as (X, Y, POINT3D_ID) \n")
            file.write("Number of images: 1 \n")
            number_part = [1,q[0],q[1],q[2],q[3], t[0],t[1],t[2],1]
            file.write(' '.join(map(str, number_part)) )
            file.write(' '+ img_name + "\n")
            print(f"Data successfully written to {txt_path}")
        return extrinsic,q

    def fov_to_intrinsic_mat(self,fov,fov_axis,w,h,txt_path):
        if fov_axis == 'x':
            df = w / 2 / np.tan(fov/2*np.pi/180.0)
        else:
            df = h / 2 / np.tan(fov/2*np.pi/180.0)
        intrinsic_matrix = np.array([[df,0,w/2],[0,df,h/2],[0,0,1]])
        f = intrinsic_matrix[0][0]
        k = 0
        with open(txt_path, 'w') as file:
            file.write("# Camera list with one line of data per camera: \n")
            file.write("CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n")
            file.write("Number of cameras:1. \n")
            string_part = "1 SIMPLE_RADIAL"
            number_part = [w, h, f, int(w/2), int(h/2), k]
            data_str = f"{string_part} {' '.join(map(str, number_part))}"
            file.write(data_str)
            print(f"Data successfully written to {txt_path}")
        return intrinsic_matrix

    def rotation_matrix_to_quaternion(self,R):
        if R.shape != (3, 3):
            raise ValueError("Input must be a 3x3 rotation matrix.")
        q = np.empty(4)
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2  # S=4*qw
            q[0] = 0.25 * s
            q[1] = (R[2, 1] - R[1, 2]) / s
            q[2] = (R[0, 2] - R[2, 0]) / s
            q[3] = (R[1, 0] - R[0, 1]) / s
        else:
            # Find the largest diagonal element
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:  # R[0, 0] is largest
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
                q[0] = (R[2, 1] - R[1, 2]) / s
                q[1] = 0.25 * s
                q[2] = (R[0, 1] + R[1, 0]) / s
                q[3] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:  # R[1, 1] is largest
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
                q[0] = (R[0, 2] - R[2, 0]) / s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = 0.25 * s
                q[3] = (R[1, 2] + R[2, 1]) / s
            else:  # R[2, 2] is largest
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
                q[0] = (R[1, 0] - R[0, 1]) / s
                q[1] = (R[0, 2] + R[2, 0]) / s
                q[2] = (R[1, 2] + R[2, 1]) / s
                q[3] = 0.25 * s
        return q


if __name__ == '__main__':
    #镜球检测
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', default='expDir')
    opt = parser.parse_args()
    expDir = opt.expDir
    imgFolder = os.path.join(expDir, "raw_images")
    print("The intermediate images are stored in tmp/balldetect. Please check these images and select the best detected ball.")
    for p in sorted(os.listdir(imgFolder)):
        if '.' == p[0]:
            continue
        img_path = os.path.join(imgFolder,p) #从这里改变图片的名字（图片均在SinglenvBall）
        img_name = os.path.basename(img_path)
        img_nameno = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs(os.path.join(expDir, 'sparse'),exist_ok=True)
        circleFile = os.path.join(expDir, f'sparse/{img_nameno}_ballLoc.txt')
        detcircle = detcircles(img_path)
        h, w = detcircle.detect_and_select_circles(img_path, circleFile)
        #相机内外参检测
        # fov = 
        if h == w == 0:
            continue
        fov = np.arctan(36/2/24)/np.pi * 180 * 2
        origin = [1.375, 0, 3.3919]
        target = [1.375, 0, 0]
        up = [0, 1, 0]
        fov_axis = "x"
        imgtxt = os.path.join(expDir, f'sparse/{img_nameno}_images.txt')
        cameratxt = os.path.join(expDir,f'sparse/{img_nameno}_camera.txt')
        detcircle.fov_to_intrinsic_mat(fov,fov_axis,w,h, cameratxt)
        detcircle.get_extrinsic_matrix(target,origin,up,imgtxt,coord_type='opencv')
