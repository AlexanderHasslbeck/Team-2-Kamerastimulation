from imagecompare import imgcompare

bag_real = "C:/projekt/90deg_camera.bag"
bag_sim = "C:/projekt/2023-06-16-15-00-01.bag"
topic_real = "/arena_camera_node/image_raw/compressed"
topic_sim = "/carla/ego_vehicle/rgb_front/image"

i = imgcompare(bag_real, bag_sim, topic_real, topic_sim)

i.img_cmp(1936, 1464, True, True, False, True, True, img_exist=True,frame_cnt_real=298,frame_cnt_sim=401)