import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from os import listdir
from os.path import isfile, join
from random import randrange
import random
def gen_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    rgb = [r,g,b]
    return rgb
color_list = []
for i in range(60):
    color_list.append(gen_color())

class MainWindow():
    def __init__(self, window): 
        
        # Vars
        self.window = window
        self.window.geometry("1500x1200")
        self.interval = 500 # Interval in ms to get the latest frame
        self.width = 450
        self.height = 450
          
        # First Frame zero
        zero_img = np.zeros([self.height, self.width, 3],dtype=np.uint8)
        zero_img.fill(255)
        zero_img = self.resize_image(zero_img)
        self.image_zero = zero_img
        self.image_1 = self.img_to_photo(self.image_zero)
        self.image_2 = self.img_to_photo(self.image_zero)
        self.image_3 = self.img_to_photo(self.image_zero)
        
        # Labels
        self.label = tk.Label(self.window, text="CELL TRACKING", font=("Calibri", 16))
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Original Frame", font=("Calibri", 12), relief="groove", width=57)
        self.label.grid(row=4, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Seg & Loc", font=("Calibri", 12), relief="groove", width=57)
        self.label.grid(row=4, column=2, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Tracking", font=("Calibri", 12), relief="groove", width=57)
        self.label.grid(row=4, column=4, padx=10, pady=5, sticky="sw")
        
        # Videos canvas
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas2 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas3 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=5, column=0, padx=10)
        self.canvas2.grid(row=5, column=2, padx=10) 
        self.canvas3.grid(row=5, column=4, padx=10) 
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_1)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_2)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_3)
        
        # Btn start
        self.btn_start = tk.Button(self.window, text="START", width=20, command=self.start_video)
        self.btn_start.grid(row=1, column=0, padx=10)
        self.btn_reset = tk.Button(self.window, text="RESTART", width=20, command=self.restart)
        self.btn_reset.grid(row=2, column=0, padx=10)
        
        
        # DATA list
        def get_files_in_dir(dir_path):
            file_list = [dir_path+f for f in listdir(dir_path) if isfile(join(dir_path, f))]
            print(f'Dir {dir_path} - len {len(file_list)} - Example file {file_list[0]}')
            return file_list
        def read_content_im(l):
            res_list = []
            for el in l:
                im = cv2.imread(el)
                res_list.append(im)    
            return res_list
        path_im = 'Cells-Images/'
        self.im = read_content_im(get_files_in_dir(path_im)[0:50])
        ## Init values
        self.restart()
        
        
        
    ############################## Core
    def start_video(self):
        self.update_image() # Update Frames on canvas
    def restart(self):
        self.index = 2
        #bc = cv2.imread('bc.tif')
        self.bc = self.im[0]
    def update_image(self):
        # Get the latest frame and convert image format
        self.image_1 = self.im[self.index].copy() 
        self.image_2 = self.image_1.copy() #np.ones([self.image_1.shape[0], self.image_1.shape[1], 3],dtype=np.uint8)   
        self.image_3 = self.image_1.copy()
        
        # Background on image
        bc = self.bc
        bc_tresh = cv2.imread('bc-tresh2.jpg')
        bc_tresh = cv2.cvtColor(bc_tresh, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(bc, self.image_2)
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Tresholding: isolate cells
        ret, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        
        # Background on treshold
        ret, bc_tresh = cv2.threshold(bc_tresh, 0, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        thresh = cv2.absdiff(bc_tresh, thresh)
        
        # Cleaning
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations = 1)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations = 3)
        self.image_2 = thresh
        
        # Identify cells
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_output = cv2.cvtColor(np.zeros(np.shape(thresh), dtype='uint8'), cv2.COLOR_GRAY2BGR) 
        clean_filled = cv2.cvtColor(np.zeros(np.shape(thresh), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        new_pos = []
        for i, cont in enumerate(contours):
            # compute the center of the contour
            try:
                M = cv2.moments(cont)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(cont)
                if area > 200 and cY > 35:
                    new_pos.append([cX,cY])
                    boundrec = cv2.boundingRect(contours[i])
                    cv2.drawContours(self.image_3, contours, i, (0, 0, 255), 2)
                    cv2.rectangle(self.image_3, boundrec, (0, 255, 0), 2)
            except:
                continue
        new_pos = np.asarray(new_pos)
        
        # INIT: initialize every track
        if self.index == 2:
            self.tracks = []
            self.time_alive = []
            track_init = []
            for pos in new_pos:
                track_init.append(np.asarray([pos]))
                self.time_alive.append(0) # used for new tracks
            self.tracks = np.array(track_init) # must build a 3d array {track, time step, coordinate 2d}
            self.time_alive = np.asarray(self.time_alive)
        
        # NEXT
        else:
            # COMPUTE COST
            #print(f'--------len self.tracks {len(self.tracks)}')
            #print(f'--------len new_pos {len(new_pos)}')
            prev_pos_list = []
            cost_matrix = np.zeros((len(self.tracks), len(new_pos))) # tracks; measurement
            for i in range(len(self.tracks)): # for each track
                # Position at t
                prev_pos = self.tracks[i][self.index-3]
                prev_pos_list.append(prev_pos)
                cost = self.run_euc(prev_pos, new_pos)
                cost_matrix[i] = cost # prev;new
            
            # CLEAN COST MATRIX: search min per col and row at the same time
            clean_cost_matrix = cost_matrix.copy()
            nan = 100000000
            clean_cost_matrix[:] = nan
            for i, track in enumerate(cost_matrix):
                for _ in track:
                    # find min for current track (loop because I have to check until I find the min)
                    min_cost = np.min(track)
                    min_ind = np.argmin(track)
                    if min_cost != nan:
                        # find min track wise for that loc found (compaire found min with all others tracks)
                        other_min = np.min(cost_matrix.T[min_ind])
                        other_min_ind = np.argmin(cost_matrix.T[min_ind])
                        if other_min_ind != i:
                            # Another tracks is more min on that value
                            # nan that value for the track and repeat the process
                            track[min_ind] = nan
                            continue # if no value found leave clean cost values for that track to nan
                        else:
                            # great we found the loc
                            cost_matrix[i][:] = nan # like deleting the track
                            cost_matrix[:, min_ind] = nan # like deleting the position
                            clean_cost_matrix[i][min_ind] = min_cost # add cost to matrix
                            break
                    else:
                        break
        
            # EXTRACT POTENTIAL NEW TRACKS
            not_assigned_loc = []
            for i, loc in enumerate(clean_cost_matrix.T):
                min_cost = np.min(loc)
                if min_cost == nan:
                    x, y = new_pos[i]
                    not_assigned_loc.append([[x,y]])
        
            # ASSIGN NEW POS FOR EACH TRACK: by timestep and cost values
            track_next = []
            for i, row in enumerate(clean_cost_matrix):
                min_cost = np.min(row)
                min_ind = np.argmin(row) 
                #print(f'min_cost {min_cost}')
                if min_cost > 60:
                    x, y = prev_pos_list[i]
                    x += randrange(6)
                    y += randrange(6)
                else:
                    x, y = new_pos[min_ind]
                track_next.append([[x,y]])
            a = np.array(track_next)
            self.tracks = np.hstack((self.tracks,a)) 
        
        
        
        # ADD NEW TRACKS: num measurements is more than num tracks
        num_new_tracks = len(new_pos) - len(self.tracks) 
        #print(self.tracks)
        if num_new_tracks > 0: 
            for el in not_assigned_loc:
                x, y = el[0]
                a = self.tracks[0].copy()
                a[:] = np.array([[x,y]]) 
                a = np.array([a]) 
                self.tracks = np.vstack((self.tracks,a))
                self.time_alive = np.append(self.time_alive, 0) 
        
        # VISUALIZE 
        cv2.putText(self.image_3, 'Num of tracks: {}'.format(len(self.tracks)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 1, cv2.LINE_AA)
        for i, track in enumerate(self.tracks):
            for t in track[-20:]:
                x, y = t
                cv2.circle(self.image_3, (x, y), 7, color_list[i], -1) #(0, 0, 255), -1)
        
        
        # Rescale and convert to photo
        self.image_1 = self.img_to_photo(self.resize_image(self.image_1))
        self.image_2 = self.img_to_photo(self.resize_image(self.image_2))
        self.image_3 = self.img_to_photo(self.resize_image(self.image_3))
        
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_1)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_2)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_3)
        
        # Repeat every 'interval' ms
        if self.index >= len(self.im) -2 :
            self.restart()
        else:
            self.index += 1
        self.window.after(self.interval, self.update_image)
     
    ############################## Tools
    @staticmethod
    def run_euc(a,b):
        return np.array([np.linalg.norm(a-x) for x in b])
    @staticmethod
    def euc(a,b):
        return np.linalg.norm(a-b)
    @staticmethod
    def draw_cross(im, x, y, color):
        for i in range(-8,8):
            try:
                im[x+i,y] = color
                im[x,y+i] = color
                im[x+i,y+1] = color
                im[x+1,y+i] = color  
            except:
                continue
    def estimate(self, mp, init, kalman):
        kalman.correct(mp-init)
        tp = kalman.predict()
        est_x = int(int(tp[0])+int(init[0]))
        est_y = int(int(tp[1])+int(init[1]))
        return est_x, est_y, kalman
    @staticmethod
    def init_kalman():
        kalman = cv2.KalmanFilter(4,2)
        # 1 only whereI have 2d measurement
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # transition consider both velocity and psoition
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        # This is really important because map how quikly the estimation fit the signal (very fast object needs higher coef)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32) #* 0.8 #0.03
        #kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
        return kalman
    @staticmethod
    def resize_image(img):
        # Resize
        #img = np.asarray(img)
        height, width, *_ = img.shape
        HEIGHT = 450   
        imgScale = HEIGHT/height
        newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        img = cv2.resize(img, (int(newX), int(newY)))
        #img = img[0:int(img.shape[1]), 0:int(img.shape[0])]
        return img
    @staticmethod
    def img_to_photo(img):
        # Transform to photo
        photo = Image.fromarray(img) # to PIL format
        photo = ImageTk.PhotoImage(photo) # to ImageTk format
        return photo

if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()