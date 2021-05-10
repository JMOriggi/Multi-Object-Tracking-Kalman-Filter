import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from os import listdir
from os.path import isfile, join
import random
def gen_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    rgb = [r,g,b]
    return rgb
    
class MainWindow():
    def __init__(self, window): 
        
        # Vars
        self.window = window
        self.window.geometry("1500x1200")
        self.interval = 20 # Interval in ms to get the latest frame
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
        self.label = tk.Label(self.window, text="BAT TRACKING", font=("Calibri", 16))
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
        
        # Option sensibility
        self.cost_limit = 14
        OPTIONS = [5,10,14,30]         
        self.variable = tk.StringVar(self.window)
        self.variable.set(OPTIONS[2])
        self.sel_bat_id = tk.OptionMenu(self.window, self.variable, *OPTIONS)
        self.sel_bat_id.grid(row=1, column=4, padx=10)
        self.btn_start = tk.Button(self.window, text="Apply", width=20, command=self.apply)
        self.btn_start.grid(row=2, column=4, padx=10)
        
        # DATA list
        def get_files_in_dir(dir_path):
            file_list = [dir_path+f for f in listdir(dir_path) if isfile(join(dir_path, f))]
            print(f'Dir {dir_path} - len {len(file_list)} - Example file {file_list[0]}')
            return file_list
        def read_content_im(l):
            res_list = []
            for el in l:
                res_list.append(np.asarray(Image.open(el)))    
            return res_list
        def read_content_txt_1(l):
            res_list = []
            for el in l:
                with open(el) as f:
                    bat_loc = [(int(line.strip().split(',')[0]), int(line.strip().split(',')[1])) for line in f]
                res_list.append(bat_loc)
            return res_list
        def read_content_txt_2(l):
            res_list = []
            for el in l:
                with open(el) as f:
                    bat_loc = [line.strip() for line in f]
                bat_loc = [[int(x) for x in el.split(',')] for el in bat_loc]
                res_list.append(np.asarray(bat_loc))
            return res_list
        path_false_im = 'Bat-Images/FalseColor/'
        path_gray_im = 'Bat-Images/Gray/'
        path_loc_txt = 'Bat-Loc/'
        path_seg_txt = 'Bat-Seg/'
        self.false_im = read_content_im(get_files_in_dir(path_false_im)[0:50])
        self.gray_im = read_content_im(get_files_in_dir(path_gray_im)[0:50])
        self.loc = read_content_txt_1(get_files_in_dir(path_loc_txt)[0:50])
        self.seg = read_content_txt_2(get_files_in_dir(path_seg_txt)[0:50])
        
        ## Init values
        self.restart()
        #ok=4,5,9,12,14,21,22,32,35,43   big=40,49,54  small=6,44,45    difficulty=17,18,30
        
        
        
    ############################## Core
    def start_video(self):
        self.update_image() # Update Frames on canvas
    def restart(self):
        self.index = 0
    def apply(self):
        self.cost_limit = int(self.variable.get())
        print (f"NEW TRESHOLD: {self.cost_limit}")
        self.restart()
    def update_image(self):
                
        # Get the latest frame and convert image format
        self.image_1 = self.false_im[self.index]
        gray = self.gray_im[self.index] 
        self.image_2 = gray.copy()    
        self.image_3 = gray.copy() #np.ones([self.image_2.shape[0], self.image_2.shape[1], 3],dtype=np.uint8) 
        
        # Background cancelling
        self.image_2[self.seg[self.index] == 0] = [0,0,0]
        #self.image_3[self.seg[self.index] == 0] = [0,0,0]
                
        # Localization
        loc_list = self.loc[self.index]
        for loc in loc_list:
            x, y = loc[0],loc[1]
            self.draw_cross(self.image_2, y, x, [255,0,0])
         
        # INIT: initialize every track
        if self.index == 0:
            #print('*********INIT')
            self.tracks = []
            self.initialState = []
            self.estimation = []
            self.kalman_tracks = []
            self.lost_tracks = []
            self.time_alive = []
            self.color_list = []
            track_init = []
            for bat_id in range(len(loc_list)): #[22,32,35,40,43,49,54]:
                x, y = loc_list[bat_id]
                mp = np.array([[np.float32(x)],[np.float32(y)]])
                track_init.append([[x,y]]) 
                self.kalman_tracks.append(self.init_kalman()) # each track must have his proper kalman fitler
                self.initialState.append(mp) # used in kalman
                self.lost_tracks.append(0) # used for overlapping track control
                self.time_alive.append(0) # used for new tracks
                self.color_list.append(gen_color())
                self.draw_cross(self.image_3, y, x, [255,0,0]) 
            a = np.array(track_init)
            self.color_list = np.asarray(self.color_list)
            self.tracks = a # must build a 3d array {track, time step, coordinate 2d}
            self.estimation = a 
            self.lost_tracks = np.asarray(self.lost_tracks)
            self.time_alive = np.asarray(self.time_alive)
            self.kalman_tracks = np.asarray(self.kalman_tracks)
            self.initialState = np.asarray(self.initialState)
            #print(self.initialState)
            print('AFTER INIT')
            #print(self.color_list)  
                
        # NEXT STEP: estimate with kalman, and find closest measurement (if present)
        else:
            #print('*********NEXT')
            # ESTIMATE: from previous position
            est_matrix = []
            for i in range(len(self.tracks)): # for each track
                # Position at t
                x, y = self.tracks[i][self.index-1]
                # Estimate position at t+1 (kalman)
                mp = np.array([[np.float32(x)],[np.float32(y)]])
                est_x, est_y, self.kalman_tracks[i] = self.estimate(mp, self.initialState[i], self.kalman_tracks[i])
                est_matrix.append([[est_x, est_y]])
            est_matrix = np.asarray(est_matrix)
            self.estimation = np.hstack((self.estimation,est_matrix))
            #print(est_matrix)
            est_matrix = np.squeeze(est_matrix, axis=1)                
            
            # COMPUTE COST: between computed estimate and real current measurement
            track_next = []
            track_del = []
            loc_matrix = np.asarray(loc_list)
            cost_matrix = np.zeros((len(est_matrix), len(loc_matrix))) # tracks; measurement
            #print(f'Mapped tracks = {len(self.tracks)} - current measures = {len(loc_matrix)}')
            for i, est in enumerate(est_matrix):
                cost = self.run_euc(est, loc_matrix)
                cost_matrix[i] = cost
            
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
                            cost_matrix[i][:] = nan # deleting the track
                            cost_matrix[:, min_ind] = nan # deleting the position
                            clean_cost_matrix[i][min_ind] = min_cost # add cost to matrix
                            break
                    else:
                        break
            
            # EXTRACT POTENTIAL NEW TRACKS
            not_assigned_loc = []
            for i, loc in enumerate(clean_cost_matrix.T):
                min_cost = np.min(loc)
                if min_cost == nan:
                    x, y = loc_matrix[i]
                    not_assigned_loc.append([[x,y]])
                    
            
            # ASSIGN NEW POS FOR EACH TRACK: by timestep and cost values
            for i, row in enumerate(clean_cost_matrix):
                # Init: I have to wait for the kalman to fit the object to then apply stronger filtering
                self.time_alive[i] += 1
                if self.time_alive[i] < 4:
                    min_cost = np.min(row)
                    min_ind = np.argmin(row) 
                    if min_cost < 50:
                        # great we found the loc
                        x, y = loc_matrix[min_ind]
                        track_next.append([[x,y]])
                    else:
                        # no clear measurement
                        x, y = est_matrix[i]
                        track_next.append([[x,y]])
                        '''if (min_cost < nan) and (min_cost > 100):
                            x, y = loc_matrix[min_ind]
                            not_assigned_loc.append([[x,y]])'''
                else:
                    # Next: now that kalman is ok I cna apply my logic
                    min_cost = np.min(row)
                    min_ind = np.argmin(row) 
                    #print(f'Cost: {min_cost}')     
                    if min_cost <= self.cost_limit:
                        # great we found the loc
                        x, y = loc_matrix[min_ind]
                        track_next.append([[x,y]])
                        self.lost_tracks[i] = 0 # reset (in case before was maybe lsot and now reassigned
                    else:
                        # no clear measurement
                        # if kalman settle and cost high means the track maybe lost
                        self.lost_tracks[i] += 1
                        if self.lost_tracks[i] > 5:
                            # if lost for more than 5 frame mean completely lost (trow away the track)
                            #print('TRACK LOST')
                            track_del.append(i)
                        else :
                            # I use for the moment the estimation as real measure
                            #print('TRACK MAYBE LOST')
                            x, y = est_matrix[i]
                            track_next.append([[x,y]])
                            '''if (min_cost < nan) and (min_cost > 100):
                                x, y = loc_matrix[min_ind]
                                not_assigned_loc.append([[x,y]])'''
                    
            # DEL LOST TRACKS
            track_del.sort(reverse=True)
            for i in track_del:
                #print(f'i {i} - len {len(self.tracks)}')
                self.tracks = np.delete(self.tracks, (i), axis=0)
                self.estimation = np.delete(self.estimation, (i), axis=0)
                self.lost_tracks = np.delete(self.lost_tracks, (i), axis=0)
                self.initialState = np.delete(self.initialState, (i), axis=0)
                self.kalman_tracks = np.delete(self.kalman_tracks, (i), axis=0)
                self.time_alive = np.delete(self.time_alive, (i), axis=0)
                self.color_list = np.delete(self.color_list, (i), axis=0)
                    
            # Update tracks with new timestep (columns)
            a = np.array(track_next)
            self.tracks = np.hstack((self.tracks,a)) 
               
            
            # ADD NEW TRACKS: num measurements is more than num tracks
            num_new_tracks = len(loc_list) - len(self.tracks) 
            #print(self.tracks)
            if num_new_tracks > 0:     
                #print(f'++++++{num_new_tracks} --- {len(not_assigned_loc)}')
                for el in not_assigned_loc:
                    x, y = el[0]
                    mp = np.array([[np.float32(x)],[np.float32(y)]])
                    a = self.tracks[0].copy()
                    a[:] = np.array([[x,y]]) 
                    a = np.array([a]) 
                    self.tracks = np.vstack((self.tracks,a))
                    self.estimation= np.vstack((self.estimation,a))
                    self.kalman_tracks = np.append(self.kalman_tracks, self.init_kalman()) # each track must have his proper kalman fitler
                    self.initialState = np.vstack((self.initialState, np.array([mp]))) # used in kalman
                    self.time_alive = np.append(self.time_alive, 0) # used in kalman
                    self.lost_tracks = np.append(self.lost_tracks, 0) # used for overlapping track control
                    self.color_list = np.vstack((self.color_list, gen_color()))
                    #self.draw_cross(self.image_3, y, x, [0,255,0])
            
            # VISUALIZE RESULTS
            for track in self.tracks: # for each track
                # Position at t
                x, y = track[self.index]
                self.draw_cross(self.image_3, y, x, [255,0,0]) 
            for i, track in enumerate(self.estimation): # for each track
                for tmsp in track[-30:]: # for each step
                    x, y = tmsp
                    self.draw_cross(self.image_3, y, x, self.color_list[i]) #[0,0,255])
                    
        #cv2.namedWindow("Background", cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("Background", cv2.cvtColor(self.image_3, cv2.COLOR_RGB2BGR))
        
        # Rescale and convert to photo
        self.image_1 = self.img_to_photo(self.resize_image(self.image_1))
        self.image_2 = self.img_to_photo(self.resize_image(self.image_2))
        self.image_3 = self.img_to_photo(self.resize_image(self.image_3))
        
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_1)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_2)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_3)
        
        # Repeat every 'interval' ms
        if self.index >= len(self.false_im) -2 :
            self.restart()
        else:
            self.index += 1
        self.window.after(self.interval, self.update_image)
     
    ############################## Tools
    @staticmethod
    def run_euc(a,b):
        return np.array([np.linalg.norm(a-x) for x in b])
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
