import numpy as np

def build_clip():
            # cap = cv2.VideoCapture(vid_path)
            # cap.set(1, 0)
            frame_count = 120
            if frame_count <= 56:
                # print(f'Video {vid_path} has insufficient frames')
                return None, None, None, None, None, None, None, None, None, None, None, None
            ############################# frame_list maker start here#################################
            num_frames = 16
            sr_ratio = 4
            min_temporal_span_sparse = num_frames*sr_ratio
            print('frame_count:', frame_count)
            print('min_temporal_span_sparse:', min_temporal_span_sparse)
            if frame_count > min_temporal_span_sparse:
                start_frame = np.random.randint(0,frame_count-min_temporal_span_sparse)
                
                #Dynamic skip rate experiment
                # skip_max = int((frame_count - start_frame)/params.num_frames)
                # # here 4 is the skip rate ratio = 4 chunks
                # if skip_max >= 16:
                #     sr_sparse = np.random.choice([4,8,12,16])
                # elif (skip_max<16) and (skip_max>=12):
                #     sr_sparse = np.random.choice([4,8,12])
                # elif (skip_max<12) and (skip_max>=8):
                #     sr_sparse = np.random.choice([4,8])
                # else:

                sr_sparse = 4
            else:
                start_frame = 0
                sr_sparse = 4
            sr_dense = int(sr_sparse/4)
            erase_size = 19
            
            frames_sparse = [start_frame] + [start_frame + i*sr_sparse for i in range(1, num_frames)]
            frames_dense = [[frames_sparse[j*4]]+[frames_sparse[j*4] + i*sr_dense for i in range(1, num_frames)] for j in range(4)]            

            print('frames_sparse:', frames_sparse)
            print('frames_dense:', frames_dense)

            ################################ frame list maker finishes here ###########################
            
            ################################ actual clip builder starts here ##########################

            sparse_clip = []
            dense_clip0 = []
            dense_clip1 = []
            dense_clip2 = []
            dense_clip3 = []

            a_sparse_clip = []
            a_dense_clip0 = []
            a_dense_clip1 = []
            a_dense_clip2 = []
            a_dense_clip3 = []

            list_sparse = []
            list_dense = [[] for i in range(4)]
            count = -1
            
            random_array = np.random.rand(10,8)
            x_erase = np.random.randint(0,112, size = (10,))
            y_erase = np.random.randint(0,112, size = (10,))


            cropping_factor1 = np.random.uniform(0.6, 1, size = (10,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = [np.random.randint(0, 320 - 320*cropping_factor1[ii] + 1) for ii in range(10)]          
            y0 = [np.random.randint(0, 240 - 240*cropping_factor1[ii] + 1) for ii in range(10)]

            contrast_factor1 = np.random.uniform(0.75,1.25, size = (10,))
            hue_factor1 = np.random.uniform(-0.1,0.1, size = (10,))
            saturation_factor1 = np.random.uniform(0.75,1.25, size = (10,))
            brightness_factor1 = np.random.uniform(0.75,1.25,size = (10,))
            gamma1 = np.random.uniform(0.75,1.25, size = (10,))


            erase_size1 = np.random.randint(int(erase_size/2),erase_size, size = (10,))
            erase_size2 = np.random.randint(int(erase_size/2),erase_size, size = (10,))
            random_color_dropped = np.random.randint(0,3,(10))

            print('random_array:', random_array)
            print('x_erase:', x_erase)
            print('y_erase:', y_erase)
            print('erase_size:', erase_size1)
            print('x0:', x0)
            print('y0:', y0)
            print('cropping_factor1:', cropping_factor1)
            print('contrast_factor1:', contrast_factor1)

            print('hue_factor1:', hue_factor1)
            print('saturation_factor1:', saturation_factor1)
            print('brightness_factor1:', brightness_factor1)
            print('gamma1:', gamma1)
            print('erase_size1:', erase_size1)
            print('erase_size2:', erase_size2)
            

if __name__ == "__main__":
    build_clip()