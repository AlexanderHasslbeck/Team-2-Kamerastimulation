import rosbags.convert
import rosbags.highlevel
import rosbags.interfaces
from rosbags.rosbag1 import Reader
import os
import binascii
import cv2
import skimage
import numpy as np
import time
import math
from matplotlib import pyplot as plt
from matplotlib import patches as mpachtes

class imgcompare:
    def __init__(self, bag_real, bag_sim, tpc_real, tpc_sim, 
                 result_dir = None, realpic_dir = None, simpic_dir = None, diff_dir = None, cntrs_dir = None, plot_dir = None):
        self.cwd = os.getcwd()
        self.bagpath_real = bag_real 
        self.bagpath_sim = bag_sim
        self.topic_real = tpc_real
        self.topic_sim = tpc_sim
        
        self.result_path = self.cwd
        self.real_picture_path = self.cwd + "\\real_pictures"
        self.sim_picture_path = self.cwd + "\\sim_pictures"
        self.diff_picture_path = self.cwd + "\\diff_pictures"
        self.cntrs_picture_path = self.cwd + "\\cntrs_pictures"
        self.plot_picture_path = self.cwd + "\\plots"
        
        if result_dir != None:
            self.result_path = result_dir
        
        if realpic_dir != None:
            self.real_picture_path = realpic_dir
            
        if simpic_dir != None:
            self.sim_picture_path = simpic_dir
            
        if diff_dir != None:
            self.diff_picture_path = diff_dir
            
        if cntrs_dir != None:
            self.cntrs_picture_path = cntrs_dir
            
        if plot_dir != None:
            self.plot_picture_path = plot_dir
        
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        if not os.path.exists(self.real_picture_path):
            os.mkdir(self.real_picture_path)
        
        if not os.path.exists(self.sim_picture_path):
            os.mkdir(self.sim_picture_path)
        
        if not os.path.exists(self.diff_picture_path):
            os.mkdir(self.diff_picture_path)
        
        if not os.path.exists(self.cntrs_picture_path):
            os.mkdir(self.cntrs_picture_path)
        
        if not os.path.exists(self.plot_picture_path):
            os.mkdir(self.plot_picture_path)

    
    def create_fig_ssim_rmse(self, frame_cnt, y_min, y_max):
        fig, ax = plt.subplots()
        ax.set_xlim(0, frame_cnt+1)
        ax.set_xlabel("Frame")
        if y_min or y_max != None:
            ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Score")
        ax.set_title("SSIM und RMSE")
        blue_patch = mpachtes.Patch(color="blue", label = "SSIM")
        red_patch = mpachtes.Patch(color="red", label = "RMSE")
        ax.legend(handles=[blue_patch, red_patch])
        return (fig, ax)
    
    def create_fig_rmse(self, frame_cnt, y_min, y_max):
        fig, ax = plt.subplots()
        ax.set_xlim(0, frame_cnt+1)
        ax.set_xlabel("Frame")
        if y_min or y_max != None:     
            ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Root Mean Square Error")
        ax.set_title("RMSE of Histogram per Frame")
        return (fig, ax)
    
    def image_compare_ssim(self, im1, im2, thr, full):                                              #vergleicht die bilder und bewertet diese im1 ist realbild, im2 ist sim-bild
        if full:
            (score_ssim, diff) = skimage.metrics.structural_similarity(im1, im2, full=True)         #erzeugt ein score zur bewertung der gleichheit und das differenzbild
            diff = (diff * 255).astype("uint8")                                                     #das differenzbild wird nur im bereich von 0..1 erstellt, wird auf den wertebereich 0..255 skaliert
            diff_tresh = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY_INV)[1]                    #ein binärbild wird aus dem diff-bild erstellt, weiß = unterschiede, schwarz = gleich, thr legt den schwellwert fest
            return score_ssim, diff, diff_tresh
        else:
            score_ssim = skimage.metrics.structural_similarity(im1, im2, full=False)                #wenn full = False wird nur der score ausgegeben, bei True alles
            return score_ssim
        
    def image_compare_rmse(self, im1, im2, normalized = None):
        if normalized != None:
            return skimage.metrics.normalized_root_mse(im1, im2, normalization=normalized)
        return skimage.metrics.mean_squared_error(im1, im2)                                          #im1 ist realbild, im2 ist sim-bild


    def plot_point(self, ax, point, frame_cnt, style):
        ax.plot(frame_cnt, point, style)
        

    def image_compare_hist(self, im1, im2, method):                                                  #im1 ist das realbild, im2 ist das sim-bild
        hist1 = cv2.calcHist(im1, [0], None, [256], [0, 256])
        hist2 = cv2.calcHist(im2, [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        return cv2.compareHist(hist1, hist2, method), hist1, hist2

    def plot_histogram(self, im1, im2, plot_path, frame_cnt):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.set_title(f"real_frame_{frame_cnt}")
        ax2.set_title(f"sim_frame_{frame_cnt}")
        ax1.hist(im1.ravel(),256,[0,256])
        ax2.hist(im2.ravel(),256,[0,256])
        ax1.set(ylabel = "No. of Pixels")
        ax2.set(xlabel = "Pixel Values", ylabel = "No. of Pixels")
        fig.savefig(plot_path + f"\\plot_hist_frame_{frame_cnt}.jpg", dpi=320)
        plt.close(fig)
        return
    
    def rmse(self, hist1, hist2):
        length = len(hist1)
        sum = 0
        for i in range(0, length):
            diff = hist2[i] - hist1[i]
            diff_sqrd = diff * diff
            sum += diff_sqrd
        mse = sum/length
        rmse = math.sqrt(mse)     
        return rmse, mse

    def image_compare_psnr(self, im1, im2):
        return skimage.metrics.peak_signal_noise_ratio(im1, im2)                                #im1 ist realbild, im2 ist sim-bild

    def image_compare_orb(self, im1, im2, nT):
        orb = cv2.ORB.create(edgeThreshold=100, patchSize=100)
        kpA, desA = orb.detectAndCompute(im1, None)
        kpB, desB = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher.create(normType=nT, crossCheck=True)
        matches = bf.match(desA, desB)
        matches = sorted(matches, key = lambda x: x.distance)
        return kpA, kpB, len(matches), matches

    def find_cntrs(self, img):                                                                        #findet die konturen der unterschiede
        contours= cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]   #gibt die eckpunkte der konturen wieder 
        return contours

    def draw_cntrs(self, img, cntrs):                                                                 #zeichnet konturen in ein bild
        return cv2.drawContours(img, cntrs, -1, (0, 0, 255))               

    def get_topics(self, path):
        with Reader(path) as reader:
            for connection in reader.connections:
                print(connection.topic)

    def create_img_carla(self, bagpath, picture_path, topic, width, height):                                   
        with Reader(bagpath) as reader:
            connections = [x for x in reader.connections if x.topic == topic]  
            frame_cnt = 0
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                frame = rawdata.hex()                                                           #rawdata wird in einem hex-string abgespeichert 
                index = frame.index("6267726138")
                if index != -1:
                    frame_cnt += 1
                    edited_frame = frame[index+28:]
                    image_bgra = np.frombuffer(binascii.a2b_hex(edited_frame), dtype=np.uint8).reshape((height, width, 4))
                    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(picture_path + f"\\frame_{frame_cnt}.jpg", image_bgr)
                else:
                    print(f"Fehler bei Lesen von Frame mit dem Timestamp: {timestamp}")
            return frame_cnt

    def create_img_real(self, bagpath, picture_path, topic):                                   
        with Reader(bagpath) as reader:
            connections = [x for x in reader.connections if x.topic == topic]  
            frame_cnt = 0
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                frame = rawdata.hex()                                                           #rawdata wird in einem hex-string abgespeichert 
                index = frame.find("ffd8")
                if index != -1:
                    frame_cnt += 1 
                    edited_frame = frame[index:]                                                #alle daten vor dem startbyte des jpeg-frames werden entfernt
                    f = open(picture_path + f"\\frame_{frame_cnt}.jpg", "wb")                     #jpeg-datei wird angelegt "w" = schreibender zugriff, "b" = geschriebene werte sind im byte format
                    f.write(binascii.a2b_hex(edited_frame))                                     #bearbeiteter hexstring wird in einen byte-string umformatiert und in die jpeg-datei geschrieben 
                    f.close()                                                                   #die jpeg datei wird geschlossen                                                                  
                else:
                    print(f"Fehler bei Lesen von Frame mit dem Timestamp: {timestamp}, Frame: {frame_cnt}")
            return frame_cnt

    def img_cmp(self, width, height, 
                plot_hist = False, diff = False, cntrs = False, plot_ssim_rmse = False, plot_rmse = False, img_exist = False, frame_cnt_real = None, frame_cnt_sim = None):
        
        t_start = time.time()
        #*********************************************************init start***************************************************************
        
        
        if img_exist == False:
            frame_cnt_real = self.create_img_real(self.bagpath_real, self.real_picture_path, self.topic_real)
            frame_cnt_sim = self.create_img_carla(self.bagpath_sim, self.sim_picture_path, self.topic_sim, width, height)                     #erstellt jpgs aus der .bag-file
        
        print(f"FrameCnt Real: {frame_cnt_real}")
        print(f"FrameCnt Sim: {frame_cnt_sim}")
        
        frame_cnt = min(frame_cnt_real, frame_cnt_sim)

        if plot_rmse:
            fig_rmse_hist, ax_rmse_hist = self.create_fig_rmse(frame_cnt, None, None)

        if plot_ssim_rmse:
            fig_ssim_mse, ax_ssim_rmse = self.create_fig_ssim_rmse(frame_cnt, None, None)

        
        results_txt = self.result_path + f"\\results_{time.gmtime()[3]}_{time.gmtime()[4]}_{time.gmtime()[5]}.txt"
        f = open(results_txt, "w")
        f.write(f"Score:\n")
        f.close()
        
        
        
        #*********************************************************init end*****************************************************************
        
        for i in range(1, frame_cnt+1):

            im_real_path = self.real_picture_path + f"\\frame_{i}.jpg"
            im_sim_path = self.sim_picture_path + f"\\frame_{i}.jpg"

            im1_color = cv2.imread(im_real_path)                                                                  #bilder werden als MatLike abgespeichert
            im2_color= cv2.imread(im_sim_path)

            im1_gray = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

            if diff:
                (score_ssim, im_diff, im_diff_tresh) = self.image_compare_ssim(im1_gray, im2_gray, 3, True)  #ein score, ein diff-bild und ein binäres diff-bild werden erstellt
            else:
                score_ssim = self.image_compare_ssim(im1_gray, im2_gray, 3, False)   # nur der Score wird berechnet
            
            (kp_im1, kp_im2, matches_count, matches) = self.image_compare_orb(im1_gray, im2_gray, cv2.NORM_HAMMING)                       
            score_rmse = self.image_compare_rmse(im1_gray, im2_gray, "euclidean")
            score_pnsr = self.image_compare_psnr(im1_gray, im2_gray)

            if diff:
                img_keypoints = cv2.drawMatches(im1_color, kp_im1, im2_color, kp_im2, matches, None, (0, 255, 0), cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(self.diff_picture_path + f"\\diff_frame_{i}.jpg", im_diff)                                    #diff-bild und binär diff-bild werden als .jpg abgespeichert
                cv2.imwrite(self.diff_picture_path + f"\\diff_tresh_frame_{i}.jpg", im_diff_tresh)
                cv2.imwrite(self.diff_picture_path + f"\\keypoints_frame_{i}.jpg", img_keypoints) 
                
            if cntrs:
                im_cntrs= self.find_cntrs(im_diff_tresh)                                                      #die konturen werden aus dem binären differenzbild gesucht
                img1_done = self.draw_cntrs(im1_color, im_cntrs)                                                          #die gefunden konturen werden in die originalbilder geschrieben
                img2_done = self.draw_cntrs(im2_color, im_cntrs)
                cv2.imwrite(self.cntrs_picture_path + f"\\real_frame{i}.jpg", img1_done)                                         #speichert die bilder mit änderungen ab
                cv2.imwrite(self.cntrs_picture_path + f"\\sim_frame_{i}.jpg", img2_done)
            
            score_hist, hist1, hist2 = self.image_compare_hist(im1_gray, im2_gray, cv2.HISTCMP_CORREL)

            if plot_hist:
                self.plot_histogram(im1_gray, im2_gray, self.plot_picture_path, i)
            
            if plot_rmse:    
                rmse, mse = self.rmse(hist1, hist2)
                self.plot_point(ax_rmse_hist, rmse, i, ".m")

            if plot_ssim_rmse:
                self.plot_point(ax_ssim_rmse, score_ssim, i, ".b")
                self.plot_point(ax_ssim_rmse, score_rmse, i, ".r")

            f = open(results_txt, "a")
            f.write(f"Frame {i}:\nSSIM: {score_ssim:.5f}, Orb-Matches: {matches_count}, RMSE: {score_rmse:.5f}, PNSR: {score_pnsr:.5f}, Histogram: {score_hist:.5f}\n")
            f.close
        
        
        #*********************************************************deinit start*****************************************************************
        
        if plot_rmse:
            fig_rmse_hist.savefig(self.plot_picture_path + "\\plot_rmse_hist.png", dpi = 320)

        if plot_ssim_rmse:
            fig_ssim_mse.savefig(self.plot_picture_path + "\\plot_ssim_rmse.png", dpi = 320)
        
        
        
        #*********************************************************deinit end*****************************************************************
        
        t_stop = time.time()
        print(f"Program finished in {t_stop-t_start}")
