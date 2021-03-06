# -*- coding: utf-8 -*-

import numpy as np

class Line():
    
    def __init__(self, max_buf_iter=20, pix_col=[0,255,0]):
        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.current_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.best_xfitted = None
        
        #polynomial coefficients for the most recent fit
        self.current_polyfit = [np.array([False])]  
        #polynomial coefficients averaged over the last n iterations
        self.best_polyfit = None  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        #x position of the line base in the warped image in pixels
        self.line_base_xpos_pixels = 0
        #lane pixels color
        self.lane_pixels_color = pix_col
        
        #max number of iteration to calculate best_xfitted
        self.max_buf_iter=max_buf_iter        
        # x_fitted values buffer
        self.xfit_buffer=[]
        # polynomial coefficients values buffer
        self.polyfit_buffer = []
        #buffer_index
        self.xfit_buffer_index = 0
        self.poly_buffer_index = 0
        self.current_poly_coeffs_goodness = False
        #acceptable tolerance betwen to polynomial coefficient
        self.delta_poly = 3
        

    def clear_line_data(self):
        self.detected = False        
        self.current_xfitted = [] 
        self.best_xfitted = None
        self.current_polyfit = [np.array([False])]  
        self.best_polyfit = None        
        self.radius_of_curvature = None 
        self.line_base_pos = None 
        self.diffs = np.array([0,0,0], dtype='float')       
        self.allx = None  
        self.ally = None
        self.line_base_xpos_pixels = 0                
        self.xfit_buffer=[]
        self.polyfit_buffer = []
        self.xfit_buffer_index = 0
        self.poly_buffer_index = 0
        self.current_poly_coeffs_goodness = False
        
    
    def set_detected(self, flag):
        if type(flag) == bool:
            self.detected = flag            
        
    
    def is_detected(self):
        return self.detected


    # set the pixel's position of the line
    # after the histogram detection procedure
    def set_xline_base_pixels(self, base_xpos_pix):
        self.line_base_xpos_pixels = base_xpos_pix
        
    
    def get_xline_base_pixels(self):
        return self.line_base_xpos_pixels
    
    
    def set_all_pix_values(self, allx_pix_vals, ally_pix_vals):
        if allx_pix_vals.shape[0] == 0 or ally_pix_vals.shape[0] == 0:
            self.set_detected(False)
            print('ERR: lane not detected')
            return False
        else:
            self.set_detected(True)
            self.allx = allx_pix_vals
            self.ally = ally_pix_vals
            return True
        
    
    def get_all_pix_values(self):
        return self.allx, self.ally
    

    # external call
    def set_curr_poly_coeffs(self, poly_coeffs):
        self.current_polyfit = poly_coeffs        
        
    
    def get_curr_poly_coeffs(self):
        return self.current_polyfit


    # called from set_curr_poly_coeffs
    def push_polyfit_coeff(self, curr_poly_coeff):        
        if (self.poly_buffer_index < self.max_buf_iter):
            self.polyfit_buffer.append(curr_poly_coeff)
            self.poly_buffer_index +=1
        else:
            self.polyfit_buffer.pop(0)
            self.polyfit_buffer.append(curr_poly_coeff)
       

    def get_best_polyfit_coeff(self):
        pass
    
        
    # external call
    def set_current_xfitted(self, fitx):
        self.current_xfitted = fitx        
        
    
    def get_current_xfitted(self):
        return self.current_xfitted
    

    #
    def push_xfitted(self, curr_xfit):
        if (self.xfit_buffer_index < self.max_buf_iter):
            self.xfit_buffer.append(curr_xfit)
            self.xfit_buffer_index +=1
        else:
            self.xfit_buffer.pop(0)
            self.xfit_buffer.append(curr_xfit)
            
        self.best_xfitted = np.mean(self.xfit_buffer, axis=0)
    
    
    def get_best_xfitted(self):
        return self.best_xfitted    


    
    #if the poly is similar to the poly mean return True
    def check_line_data_goodness(self):
        if len(self.polyfit_buffer) > 0:
            pass
        else: # buffer empty
            return True
    
    
    def set_lane_pixels_color(self, lane_pixels_color):
        self.lane_pixels_color = lane_pixels_color
        
    
    def get_lane_pixels_color(self):
        return self.lane_pixels_color
    
    
    def set_radius_of_curvature(self, new_radius):
        self.radius_of_curvature = new_radius
    
    
    def get_radius_of_curvature(self):
        return self.radius_of_curvature
    
    
    
    
    
if __name__ == '__main__':

    vecx = np.random.randint(0,1200,10000)
    vecy = np.random.randint(0,1200,10000)
    vecy = np.array([])
    
    test_lane_line = Line()
    test_lane_line.set_all_pix_values(vecx,vecy)
    
    print(type(vecx))
    
    

    for i in range(30):    
        item = np.random.randint(0,10,100).reshape(10,10)
        test_lane_line.push_xfitted(item)
        print('item:\n', item)
        print(len(test_lane_line.xfit_buffer))
        print(test_lane_line.get_best_xfitted())