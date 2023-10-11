import torch
torch.multiprocessing.set_start_method('spawn')
import imageio
import matplotlib.pyplot as plt
from mlxtend.image import extract_face_landmarks
import cv2
import glob
import os
import numpy as np
import tqdm
import re
import pickle
from magface_model import iresnet100, load_dict_inf
from facenet_pytorch import InceptionResnetV1
from sphereface_pytorch.net_sphere import sphere20a
from CosFace_pytorch.net import sphere
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--embedding-source")
parser.add_argument("--num-samples")
parser.add_argument("--data-path")
parser.add_argument("--results-path")
parser.add_argument("--checkpoint-path")
args = parser.parse_args()
config = vars(args)

EMBEDDING_SOURCE = config["embedding_source"]
NUMBER_OF_SAMPLES_TO_GENERATE = int(config["num_samples"])
DATA_PATH = config["data_path"]
RESULTS_PATH = config["results_path"]


if EMBEDDING_SOURCE == "magface":
    # https://github.com/IrvingMeng/MagFace
    magface = iresnet100(pretrained=False, num_classes=512)
    magface = load_dict_inf(magface, config["checkpoint_path"])
    magface = magface.to("cuda")
    magface.eval()
elif EMBEDDING_SOURCE == "arcface":
    # https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
    arcface = iresnet100(pretrained=False, num_classes=512)
    arcface.load_state_dict(torch.load(config["checkpoint_path"]))
    arcface = arcface.to("cuda")
    arcface.eval()
elif EMBEDDING_SOURCE == "sphereface":
    # https://github.com/clcarwin/sphereface_pytorch
    sphereface = sphere20a(feature=True)
    sphereface.load_state_dict(torch.load(config["checkpoint_path"]))
    sphereface = sphereface.to("cuda")
    sphereface.eval()
elif EMBEDDING_SOURCE == "cosface":
    # https://github.com/MuggleWang/CosFace_pytorch
    cosface = sphere(type=20)
    cosface.load_state_dict(torch.load(config["checkpoint_path"]))
    cosface = cosface.to("cuda")
    cosface.eval()
elif EMBEDDING_SOURCE == "facenet":
    # https://github.com/timesler/facenet-pytorch
    facenet = InceptionResnetV1(pretrained='vggface2', device="cuda")
    facenet.eval()

def get_landmarks(image_paths, image_size):
    database = {}
    for image_path in tqdm.tqdm(image_paths):
        filename = image_path.split("/")[-1].split(".")[0]
        if not bool(re.match(r"\w{2}\d{2}\w{2}\w+", filename)):
            continue

        name = re.findall(r"(\w{2}\d{2})\w{2}\w+", filename)[0]
        pose_label = re.findall(r"\w{2}\d{2}\w{2}(\w+)", filename)[0]
        emotion_label = re.findall(r"\w{2}\d{2}(\w{2})\w+", filename)[0]

        img = cv2.imread(image_path.replace("CROPPED_ALIGNED", "KDEF")) # get real images from KDEF for landmarks
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        pose_img = get_pose_image(img, image_size)

        if pose_img is None:
            continue

        person_db = database.get(name, [])
        person_db.append((emotion_label, pose_label, pose_img, image_path))
        database[name] = person_db
        
    return database

def get_pose_image(img, image_size):
    landmarks = extract_face_landmarks(img)

    if np.all(landmarks == 0) or landmarks is None:
        return None

    landmarks[:, 0][landmarks[:, 0] >= image_size] = image_size - 1
    landmarks[:, 0][landmarks[:, 0] < 0] = 0
    landmarks[:, 1][landmarks[:, 1] >= image_size] = image_size - 1
    landmarks[:, 1][landmarks[:, 1] < 0] = 0

    pose_img = np.zeros((image_size, image_size))
    pose_img[landmarks[:, 1], landmarks[:, 0]] = 1
    pose_img = pose_img[:, :, np.newaxis]
    pose_img = pose_img[landmarks[:, 1].min():landmarks[:, 1].max(), landmarks[:, 0].min():landmarks[:, 0].max()]
    pose_img = cv2.resize(pose_img, (image_size, image_size))
    pose_img = np.where(pose_img > 0.5, 1, 0)

    pose_img = pose_img[:, :, np.newaxis]

    return pose_img

def get_magface_embedding(image_path, image_size):
    img = cv2.imread(image_path)

    if img.shape[:2] != list(image_size):
        img = cv2.resize(img, image_size)

    img = img[np.newaxis] / 255.
    img = torch.Tensor(img).to("cuda").to(torch.float32)
    img = img.permute(0, 3, 1, 2)
    embedding = magface(img)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding)

    return embedding

def get_arcface_embedding(image_path, image_size):
    img = cv2.imread(image_path)

    if img.shape[:2] != list(image_size):
        img = cv2.resize(img, image_size)

    img = img[np.newaxis] / 255.
    img = torch.Tensor(img).to("cuda").to(torch.float32)
    img = img.permute(0, 3, 1, 2)
    embedding = arcface(img)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding)

    return embedding

def get_sphereface_embedding(image_path, image_size):
    img = cv2.imread(image_path)

    if img.shape[:2] != list(image_size):
        img = cv2.resize(img, image_size)

    img = (img[np.newaxis] - 127.5) / 128.0
    img = torch.Tensor(img).to("cuda").to(torch.float32)
    img = img.permute(0, 3, 1, 2)
    embedding = sphereface(img)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding)

    return embedding

def get_cosface_embedding(image_path, image_size):
    img = cv2.imread(image_path)

    if img.shape[:2] != list(image_size):
        img = cv2.resize(img, image_size)

    img = (img[np.newaxis] - 127.5) / 128.0
    img = torch.Tensor(img).to("cuda").to(torch.float32)
    img = img.permute(0, 3, 1, 2)
    embedding = cosface(img)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding)

    return embedding

def get_facenet_embedding(image_path, image_size):
    img = cv2.imread(image_path)

    if img.shape[:2] != list(image_size):
        img = cv2.resize(img, image_size)

    img = (img[np.newaxis] - 127.5) / 128.0
    img = torch.Tensor(img).to("cuda").to(torch.float32)
    img = img.permute(0, 3, 1, 2)
    embedding = facenet(img)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding)

    return embedding

def prepare_final_database(database):
    db_keys = list(database.keys())

    final_database = []
    for person1_name in db_keys:
        person1 = database[person1_name]
        for triple_to_reconstruct_index, triple_to_reconstruct in enumerate(person1):
            available_poses_input = []
            available_poses_output = []
            for person2_name in db_keys:
                person2 = database[person2_name]
                available_poses_input = list(filter(lambda index: person2[index][0] == triple_to_reconstruct[0], range(len(person2))))
                available_poses_output = list(filter(lambda index: person2[index][0] == triple_to_reconstruct[0] and person2[index][1] == triple_to_reconstruct[1], range(len(person2))))

                for triple_to_change_input_index in available_poses_input:
                    for triple_to_change_output_index in available_poses_output:
                        final_database.append((person1_name, triple_to_reconstruct_index, person2_name, triple_to_change_input_index, triple_to_change_output_index))

                if len(final_database) >= NUMBER_OF_SAMPLES_TO_GENERATE:
                    break

            if len(final_database) >= NUMBER_OF_SAMPLES_TO_GENERATE:
                break

        if len(final_database) >= NUMBER_OF_SAMPLES_TO_GENERATE:
            break
            
    return final_database

def save_files(database, final_database, type_, image_size):
    for idx in tqdm.tqdm(range(len(final_database))):
        tup = final_database[idx]

        if os.path.isfile(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_pose_img_to_reconstruct.npy') and os.path.isfile(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_input.npy') and os.path.isfile(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_output.npy'):
            continue

        pose_img_to_reconstruct = database[tup[0]][tup[1]][2]
        image_path_input = database[tup[2]][tup[3]][3]
        image_path_output = database[tup[2]][tup[4]][3]

        if EMBEDDING_SOURCE == "magface":
            embedding_input = get_magface_embedding(image_path_input, image_size)
            embedding_output = get_magface_embedding(image_path_output, image_size)
        elif EMBEDDING_SOURCE == "facenet":
            embedding_input = get_facenet_embedding(image_path_input, image_size)
            embedding_output = get_facenet_embedding(image_path_output, image_size)
        elif EMBEDDING_SOURCE == "arcface":
            embedding_input = get_arcface_embedding(image_path_input, image_size)
            embedding_output = get_arcface_embedding(image_path_output, image_size)
        elif EMBEDDING_SOURCE == "sphereface":
            embedding_input = get_sphereface_embedding(image_path_input, image_size)
            embedding_output = get_sphereface_embedding(image_path_output, image_size)
        elif EMBEDDING_SOURCE == "cosface":
            embedding_input = get_cosface_embedding(image_path_input, image_size)
            embedding_output = get_cosface_embedding(image_path_output, image_size)

        with open(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_pose_img_to_reconstruct.npy', 'wb') as f:
            np.save(f, pose_img_to_reconstruct)

        with open(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_input.npy', 'wb') as f:
            np.save(f, embedding_input)

        with open(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/{type_}/{tup[0]}_{tup[1]}_{tup[2]}_{tup[3]}_{tup[4]}_embedding_output.npy', 'wb') as f:
            np.save(f, embedding_output)

    with open(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_database_{type_}.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if os.path.isdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices"):
        shutil.rmtree(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices")

    os.mkdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices")
    os.mkdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/training")
    os.mkdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_matrices/evaluation")

    people_paths = glob.glob(os.path.join(f"{DATA_PATH}/CROPPED_ALIGNED/*"))
    people_paths = np.array(people_paths)
    np.random.shuffle(people_paths)
    people_paths = people_paths.tolist()

    # index = int(0.8 * len(people_paths))
    trainset = people_paths#[:index]
    evalset = people_paths#[index:]

    if EMBEDDING_SOURCE == "magface" or EMBEDDING_SOURCE == "arcface":
        image_size = 112, 112
    elif EMBEDDING_SOURCE == "facenet":
        image_size = 160, 160
    elif EMBEDDING_SOURCE == "sphereface" or EMBEDDING_SOURCE == "cosface":
        image_size = 112, 96

    print("Generating training pairs")
    image_paths = [img_path for person_path in trainset for img_path in glob.glob(person_path + "/*.JPG")]
    database = get_landmarks(image_paths, 112)
    final_database = prepare_final_database(database)
    save_files(database, final_database, "training", image_size)

    print("Generating evaluation pairs")
    image_paths = [img_path for person_path in evalset for img_path in glob.glob(person_path + "/*.JPG")]
    database = get_landmarks(image_paths, 112)
    final_database = prepare_final_database(database)
    save_files(database, final_database, "evaluation", image_size)
