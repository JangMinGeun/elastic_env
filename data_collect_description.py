import os
import random
import tqdm
import numpy as np
from PIL import Image
from datetime import datetime
import yaml
import h5py
import gym
import elastic2d
import pickle


def randval():
    return (random.random() / 2. + 0.5) * random.choice([-1, 1])

# load config.yaml file
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not os.path.exists("results"):
    os.makedirs("results/odn")
    os.makedirs("results/orn")
    os.makedirs("results/sensor")

if config['num_objects'] == 1:
    pass
    # TODO
else:
    start_time = datetime.now().strftime("%Y_%m_%d %H:%M:%S")
    os.makedirs("results/orn/{}/train/input".format(config["NAME"]))
    os.makedirs("results/orn/{}/train/output".format(config["NAME"]))
    os.makedirs("results/orn/{}/test/input".format(config["NAME"]))
    os.makedirs("results/orn/{}/test/output".format(config["NAME"]))

    env = gym.make('elastic2d-v1', 
            colors=config['colors'], objs=config['objs'], 
            numofobjs=config['numofobjs'], sizes=config['sizes'])
    
    _ = env.reset()
    img_num = 0
    log_images = []

    actions = np.zeros((config['num_train_images'], 2))

    cur_datas = np.zeros((config['num_train_images'], config['numofobjs'] * 3))
    next_datas = np.zeros((config['num_train_images'], config['numofobjs'] * 3))

    cur_descript = []
    next_descript = []


    with tqdm.tqdm(total = config['num_train_images']) as pbar:
        while img_num < config['num_train_images']:
            if img_num % config['episode_steps'] == 0:
                env.reset()
                img, _, _, _= env.step([0,0])
                img = Image.fromarray(img)
                obs = env.render(mode="sensor")
                description = env.render(mode="descript")

                # OP3 Data
                episode_images = []
                episode_actions = []
                
            xaxis, yaxis = randval(), randval()
            actions[img_num][0], actions[img_num][1] = xaxis, yaxis
            cur_datas[img_num] = obs
            cur_descript.append(description)

            img.save("results/orn/{}/train/input/{}_{}.png".format(config["NAME"],config["NAME"], img_num))
            img, _, _, _= env.step(actions[img_num])
            img = Image.fromarray(img)
            img.save("results/orn/{}/train/output/{}_{}.png".format(config["NAME"], config["NAME"], img_num))
            obs = env.render(mode="sensor")
            description = env.render(mode="descript")
            # print(description)
            next_datas[img_num] = obs
            next_descript.append(description)

            img_num = img_num + 1
            pbar.update(1)   
            
    np.savetxt("results/orn/{}/train/{}_actions.csv".format(config["NAME"], config["NAME"]), actions, delimiter=',')
    np.savetxt("results/orn/{}/train/{}_cur_data.csv".format(config["NAME"], config["NAME"]), cur_datas, delimiter=',')
    np.savetxt("results/orn/{}/train/{}_next_data.csv".format(config["NAME"], config["NAME"]), next_datas, delimiter=',')

    with open("results/orn/{}/train/{}_descript.pickle".format(config["NAME"], config["NAME"]), "wb") as fw:
        pickle.dump([cur_descript, next_descript], fw)



    _ = env.reset()
        
    img_num = 0
    actions = np.zeros((config['num_test_images'], 2))
    cur_datas = np.zeros((config['num_test_images'], config['numofobjs'] * 3))
    next_datas = np.zeros((config['num_test_images'], config['numofobjs'] * 3))

    cur_descript = []
    next_descript = []

    with tqdm.tqdm(total = config['num_test_images']) as pbar:
        while img_num < config['num_test_images']:
            if img_num % config['episode_steps'] == 0:
                env.reset()
                img, _, _, _= env.step([0,0])
                img = Image.fromarray(img)
                obs = env.render(mode="sensor")
                description = env.render(mode="descript")

            xaxis, yaxis = randval(), randval()
            actions[img_num][0], actions[img_num][1] = xaxis, yaxis

            img.save("results/orn/{}/test/input/{}_{}.png".format(config["NAME"], config["NAME"], img_num))
            cur_datas[img_num] = obs
            cur_descript.append(description)

            img, _, _, _ = env.step(actions[img_num])
            img = Image.fromarray(img)
            img.save("results/orn/{}/test/output/{}_{}.png".format(config["NAME"], config["NAME"], img_num))
            obs = env.render(mode="sensor")
            description = env.render(mode="descript")
            next_datas[img_num] = obs
            next_descript.append(description)
            img_num = img_num + 1
            pbar.update(1)   

    np.savetxt("results/orn/{}/test/{}_actions.csv".format(config["NAME"],config["NAME"]), actions, delimiter=',')
    np.savetxt("results/orn/{}/test/{}_cur_data.csv".format(config["NAME"], config["NAME"]), cur_datas, delimiter=',')
    np.savetxt("results/orn/{}/test/{}_next_data.csv".format(config["NAME"], config["NAME"]), next_datas, delimiter=',')

    with open("results/orn/{}/test/{}_descript.pickle".format(config["NAME"], config["NAME"]), "wb") as fw:
        pickle.dump([cur_descript, next_descript], fw)

