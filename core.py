import os
import sys
import time
import numpy as np
from sage.base_app import BaseApp
from .const import IMU_LIST, IMU_FIELDS, ACC_ALL, GYR_ALL, MAX_BUFFER_LEN, GRAVITY, WEIGHT_LOC, HEIGHT_LOC

third_party_path = os.path.abspath(os.path.join(__file__, '../third_party'))
sys.path.insert(0, third_party_path)

if __name__ == '__main__':
    from GRFPredictor import GRFPredictor
    
else:
    from .GRFPredictor import GRFPredictor


sys.path.remove(third_party_path)

class Core(BaseApp):
    def __init__(self, my_sage):
        BaseApp.__init__(self, my_sage, __file__)
        self.grf_predictor = GRFPredictor(self.config['weight'], self.config['height'])
        self.time_now = 0
        
    def run_in_loop(self):
        data = self.my_sage.get_next_data()
        data_and_grfs_list = self.grf_predictor.update_stream(data)
        print(f"Data and GRFs list: {data_and_grfs_list}")
            
        if data_and_grfs_list:
            print("Processing data and GRFs")
            for data_and_grfs in data_and_grfs_list:
                # Print the data_and_grfs variable
                print(f"data_and_grfs: {data_and_grfs}")

                data = data_and_grfs[0]
                grfs = data_and_grfs[1]
                self.time_now += 0.01
                my_data = {'time': [self.time_now], 
                        'plate_1_force_x': [grfs[0]], 
                        'plate_1_force_y': [grfs[1]], 
                        'plate_1_force_z': [grfs[2]], 
                        'plate_2_force_x': [grfs[3]], 
                        'plate_2_force_y': [grfs[4]], 
                        'plate_2_force_z': [grfs[5]]}

                print(f"My data: {my_data}")
                self.my_sage.send_stream_data(data, my_data)
                print(f"grfs for plate_1_force_x: {grfs[0]}")
                self.my_sage.save_data(data, my_data)
                # Print statement after saving data
                print("Data saved.")
        else:
            print("No data or GRFs")
            
        return True




if __name__ == '__main__':
    # This is only for testing. make sure you do the pairing first in web api
    from sage.sage import Sage
    app = Core(Sage())
    app.test_run()
