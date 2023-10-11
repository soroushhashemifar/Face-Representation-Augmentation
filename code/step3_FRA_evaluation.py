import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from dataset import CustomDataSetEvaluation
from step2_train_FRA import ConvAutoencoder
import time
import os
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
import pickle
import seaborn as sns
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embedding-source")
parser.add_argument("--results-path")
parser.add_argument("--load-pickles", action='store_true')
parser.add_argument("--plot-autoencoder", action='store_true')
args = parser.parse_args()
config = vars(args)

EMBEDDING_SOURCE = config["embedding_source"]
load_pickles = bool(config["load_pickles"])
plot_autoencoder_results = bool(config["plot_autoencoder"])
RESULTS_PATH = config["results_path"]

def plot_metrics(val_embeddings, val_labels, val_labels_pred, clf, target, objective_name):
    if isinstance(val_labels, list):
        val_labels = np.array(val_labels)
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    val_labels_oh_encoded = enc.fit_transform(val_labels[:, np.newaxis])
    y_score = clf.predict_proba(val_embeddings)

    if target == "pose":
        plot_labels = list(map(lambda item: list(filter(lambda item2: item2[1]==item, list(val_dataset.poses_dict.items())))[0][0], np.unique(val_labels)))
        plot_labels_inverse_dict = {v: k for k, v in val_dataset.poses_dict.items()}
    elif target == "id":
        plot_labels = list(map(lambda item: list(filter(lambda item2: item2[1]==item, list(val_dataset.ids_dict.items())))[0][0], np.unique(val_labels)))
        plot_labels_inverse_dict = {v: k for k, v in val_dataset.ids_dict.items()}
    elif target == "emo":
        plot_labels = list(map(lambda item: list(filter(lambda item2: item2[1]==item, list(val_dataset.emotions_dict.items())))[0][0], np.unique(val_labels)))
        plot_labels_inverse_dict = {v: k for k, v in val_dataset.emotions_dict.items()}

    # precision recall curve
    fig = plt.figure(figsize=(10, 10))
    precision = dict()
    recall = dict()
    for i in range(val_labels_oh_encoded.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(val_labels_oh_encoded[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(plot_labels_inverse_dict[i]))
        
    # plt.ylim([0, 1.2])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title(f"precision vs. recall curve ({target})")
    plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_prc_{target}_{objective_name}.png')

    # roc curve
    fig = plt.figure(figsize=(10, 10))
    fpr = dict()
    tpr = dict()
    for i in range(val_labels_oh_encoded.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(val_labels_oh_encoded[:, i], y_score[:, i])
        auc_score = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {} (AUC = {})'.format(plot_labels_inverse_dict[i], round(auc_score, 4)))

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="No skill")
    
    # plt.ylim([0, 1.2])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title(f"ROC curve ({target})")
    plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_roc_{target}_{objective_name}.png')

    fig = plt.figure(figsize=(10, 10))
    cf_matrix = confusion_matrix(val_labels, val_labels_pred)
    sns.heatmap(cf_matrix, annot=False, xticklabels=plot_labels, yticklabels=plot_labels)
    plt.title(f"Confusion matrix ({target})")
    plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_conf_{target}_{objective_name}.png')


val_dataset = CustomDataSetEvaluation(EMBEDDING_SOURCE=EMBEDDING_SOURCE, RESULTS_PATH=RESULTS_PATH)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)

net = ConvAutoencoder(EMBEDDING_SOURCE, 0.0)
net = net.to("cpu")
state_dict = torch.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_augmentor.pt")
net.load_state_dict(state_dict)
net.eval()

if plot_autoencoder_results:
    # Saving a sample of reconstructed pose styles by the AutoEncoder
    with torch.no_grad():
        fig = plt.figure(figsize=(2, 10))
        fig.subplots_adjust(hspace=0.05, wspace=0.)

        samples_to_plot = np.random.randint(0, len(val_dataset), 10)
        for index, i in enumerate(samples_to_plot):
            data = val_dataset[i]
            
            pose_img = data[0]
            pose_img = torch.tensor(pose_img)
            pose_img = pose_img.unsqueeze(0)
            pose_img = pose_img.permute(0, 3, 1, 2)
            pose_img = pose_img.to(torch.float32)
            
            features = net.encoder(pose_img)
            reconstructed_data = net.decoder_pose_reconstructor(features)
            
            reconstructed_data = reconstructed_data.permute(0, 2, 3, 1).cpu().numpy()
            orig_data = pose_img.permute(0, 2, 3, 1).cpu().numpy()
            
            plt.subplot(10, 2, index*2+1)
            plt.imshow(1-orig_data[0, :, :, 0], cmap="binary")
            plt.axis('off')
            plt.subplot(10, 2, index*2+2)
            plt.imshow(1-reconstructed_data[0, :, :, 0], cmap="binary")
            plt.axis('off')
        
        plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_AE_reconstrcution_{int(time.time())}.png')


########################
# Objective 1
########################
if not load_pickles:
    preaugment_traindata_embs = []
    preaugment_traindata_labels_pose = []
    preaugment_traindata_labels_id = []
    preaugment_traindata_labels_emo = []
    preaugment_valdata_embs = []
    preaugment_valdata_labels_pose = []
    preaugment_valdata_labels_id = []
    preaugment_valdata_labels_emo = []

    for data in tqdm.tqdm(val_dataloader):
        pose_img = data[0]
        pose_img = pose_img.permute(0, 3, 1, 2)
        pose_img = pose_img.to(torch.float32)
            
        input_emb = data[1]
        inpuy_emb_np = input_emb.detach().cpu().numpy()
        preaugment_traindata_embs.append(inpuy_emb_np)
        input_emb = input_emb.to(torch.float32)
        
        output_emb = data[2]
        
        expected_pose_label = data[3]
        
        input_pose_label = data[4]
        preaugment_traindata_labels_pose.append(input_pose_label)

        expected_id_label = data[5]
        preaugment_traindata_labels_id.append(expected_id_label)

        expected_emo_label = data[6]
        preaugment_traindata_labels_emo.append(expected_emo_label)

        # ensure to not mixing the train and validation sets
        output_emb = output_emb.detach().cpu().numpy()
        
        features = net.encoder(pose_img)
        generated_embedding = net.decoder_emb_generator(features, input_emb)
        
        generated_embedding = generated_embedding.detach().cpu().numpy()
        
        # objective 1: measuring how linear separable are the representations
        preaugment_valdata_embs.append(generated_embedding) 
        preaugment_valdata_labels_pose.append(expected_pose_label) 
        preaugment_valdata_labels_id.append(expected_id_label) 
        preaugment_valdata_labels_emo.append(expected_emo_label)
        
    preaugment_traindata_embs = np.concatenate(preaugment_traindata_embs)
    preaugment_traindata_labels_pose = np.concatenate(preaugment_traindata_labels_pose)
    preaugment_traindata_labels_id = np.concatenate(preaugment_traindata_labels_id)
    preaugment_traindata_labels_emo = np.concatenate(preaugment_traindata_labels_emo)
    preaugment_valdata_embs = np.concatenate(preaugment_valdata_embs)
    preaugment_valdata_labels_pose = np.concatenate(preaugment_valdata_labels_pose)
    preaugment_valdata_labels_id = np.concatenate(preaugment_valdata_labels_id)
    preaugment_valdata_labels_emo = np.concatenate(preaugment_valdata_labels_emo)

    scaler = StandardScaler()
    scaler.fit(preaugment_traindata_embs)
    preaugment_traindata_embs = scaler.transform(preaugment_traindata_embs)

    scaler2 = StandardScaler()
    scaler2.fit(preaugment_valdata_embs)
    preaugment_valdata_embs = scaler2.transform(preaugment_valdata_embs)

    preaugment_traindata_labels_encoded_pose = preaugment_traindata_labels_pose
    preaugment_traindata_labels_encoded_id = preaugment_traindata_labels_id
    preaugment_traindata_labels_encoded_emo = preaugment_traindata_labels_emo
    preaugment_valdata_labels_encoded_pose = preaugment_valdata_labels_pose
    preaugment_valdata_labels_encoded_id = preaugment_valdata_labels_id
    preaugment_valdata_labels_encoded_emo = preaugment_valdata_labels_emo

    if not os.path.isdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys"):
        os.mkdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys")

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_embs.npy", preaugment_traindata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_pose.npy", preaugment_traindata_labels_encoded_pose)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_embs.npy", preaugment_valdata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_pose.npy", preaugment_valdata_labels_encoded_pose)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_id.npy", preaugment_traindata_labels_encoded_id)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_id.npy", preaugment_valdata_labels_encoded_id)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_emo.npy", preaugment_traindata_labels_encoded_emo)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_emo.npy", preaugment_valdata_labels_encoded_emo)

else:
    preaugment_traindata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_embs.npy")
    preaugment_traindata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_pose.npy")
    preaugment_valdata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_embs.npy")
    preaugment_valdata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_pose.npy")

    preaugment_traindata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_id.npy")
    preaugment_valdata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_id.npy")

    preaugment_traindata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_traindata_labels_encoded_emo.npy")
    preaugment_valdata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/preaugment_valdata_labels_encoded_emo.npy")

accuracy_pose = []
accuracy_id = []
accuracy_emotion = []
val_accuracy_pose = []
val_accuracy_id = []
val_accuracy_emotion = []

clf_pose = SVC(probability=True)
clf_pose.fit(preaugment_traindata_embs, preaugment_traindata_labels_encoded_pose)
accuracy_pose.append(np.mean(clf_pose.predict(preaugment_traindata_embs) == preaugment_traindata_labels_encoded_pose))
preaugment_valdata_labels_encoded_pose_pred = clf_pose.predict(preaugment_valdata_embs)
val_accuracy_pose.append(np.mean(preaugment_valdata_labels_encoded_pose_pred == preaugment_valdata_labels_encoded_pose))

clf_id = SVC(probability=True)
clf_id.fit(preaugment_traindata_embs, preaugment_traindata_labels_encoded_id)
accuracy_id.append(np.mean(clf_id.predict(preaugment_traindata_embs) == preaugment_traindata_labels_encoded_id))
preaugment_valdata_labels_encoded_id_pred = clf_id.predict(preaugment_valdata_embs)
val_accuracy_id.append(np.mean(preaugment_valdata_labels_encoded_id_pred == preaugment_valdata_labels_encoded_id))

clf_emo = SVC(probability=True)
clf_emo.fit(preaugment_traindata_embs, preaugment_traindata_labels_encoded_emo)
accuracy_emotion.append(np.mean(clf_emo.predict(preaugment_traindata_embs) == preaugment_traindata_labels_encoded_emo))
preaugment_valdata_labels_encoded_emo_pred = clf_emo.predict(preaugment_valdata_embs)
val_accuracy_emotion.append(np.mean(preaugment_valdata_labels_encoded_emo_pred == preaugment_valdata_labels_encoded_emo))

print("Objective: Pre-Augmentation")
print("Avg ACC pose:", np.mean(accuracy_pose), "Avg ACC id:", np.mean(accuracy_id), "Avg ACC emotion:", np.mean(accuracy_emotion))
print("Avg ACC pose (Val):", np.mean(val_accuracy_pose), "Avg ACC id (Val):", np.mean(val_accuracy_id), "Avg ACC emotion (Val):", np.mean(val_accuracy_emotion))

plot_metrics(preaugment_valdata_embs, preaugment_valdata_labels_encoded_pose, preaugment_valdata_labels_encoded_pose_pred, clf_pose, "pose", "preaugment")
plot_metrics(preaugment_valdata_embs, preaugment_valdata_labels_encoded_id, preaugment_valdata_labels_encoded_id_pred, clf_id, "id", "preaugment")
plot_metrics(preaugment_valdata_embs, preaugment_valdata_labels_encoded_emo, preaugment_valdata_labels_encoded_emo_pred, clf_emo, "emo", "preaugment")


####################
# Objectives 2 and 3
####################
if not load_pickles:
    gen_embed_traindata_embs = []
    gen_embed_traindata_labels_pose = []
    gen_embed_traindata_labels_id = []
    gen_embed_traindata_labels_emo = []
    gen_embed_valdata_embs = []
    gen_embed_valdata_labels_pose = []
    gen_embed_valdata_labels_id = []
    gen_embed_valdata_labels_emo = []

    postaugment_traindata_embs_gens = []
    postaugment_traindata_labels_pose_gens = []
    postaugment_traindata_labels_id_gens = []
    postaugment_traindata_labels_emo_gens = []

    for data in tqdm.tqdm(val_dataset):
        pose_img = data[0]
        pose_img = torch.Tensor(pose_img)
        pose_img = pose_img.unsqueeze(0)
        pose_img = pose_img.permute(0, 3, 1, 2)
        pose_img = pose_img.to(torch.float32)
        
        input_emb = data[1]
        
        output_emb = data[2]
        output_emb = torch.Tensor(output_emb)
        output_emb = output_emb.unsqueeze(0)
        output_emb = output_emb.to(torch.float32)
        
        expected_pose_label = data[3]
        input_pose_label = data[4]
        expected_id_label = data[5]
        expected_emo_label = data[6]
        
        if (len(gen_embed_traindata_embs) > 1 and not np.any(np.array(gen_embed_traindata_embs) == input_emb) and len(gen_embed_valdata_embs) > 1 and not np.any(np.array(gen_embed_valdata_embs) == input_emb)) or len(gen_embed_traindata_embs) in [0, 1]:
            gen_embed_traindata_embs.append(input_emb)
            gen_embed_traindata_labels_pose.append(input_pose_label)
            gen_embed_traindata_labels_id.append(expected_id_label)
            gen_embed_traindata_labels_emo.append(expected_emo_label)

        input_emb = torch.Tensor(input_emb)
        input_emb = input_emb.unsqueeze(0)
        input_emb = input_emb.to(torch.float32)

        features = net.encoder(pose_img)
        generated_embedding = net.decoder_emb_generator(features, input_emb)
        generated_embedding = generated_embedding.detach().cpu().numpy()
        postaugment_traindata_embs_gens.append(generated_embedding[0])
        postaugment_traindata_labels_pose_gens.append(expected_pose_label)
        postaugment_traindata_labels_id_gens.append(expected_id_label)
        postaugment_traindata_labels_emo_gens.append(expected_emo_label)
        
        output_emb = output_emb.detach().cpu().numpy()[0]

        if (len(gen_embed_valdata_embs) > 1 and not np.any(np.array(gen_embed_valdata_embs) == output_emb) and len(gen_embed_traindata_embs) > 1 and not np.any(np.array(gen_embed_traindata_embs) == output_emb)) or len(gen_embed_valdata_embs) in [0, 1]:
            gen_embed_valdata_embs.append(output_emb)
            gen_embed_valdata_labels_pose.append(expected_pose_label)
            gen_embed_valdata_labels_id.append(expected_id_label)
            gen_embed_valdata_labels_emo.append(expected_emo_label)
        
    postaugment_traindata_embs = gen_embed_traindata_embs + postaugment_traindata_embs_gens
    postaugment_traindata_labels_pose = gen_embed_traindata_labels_pose + postaugment_traindata_labels_pose_gens
    postaugment_traindata_labels_id = gen_embed_traindata_labels_id + postaugment_traindata_labels_id_gens
    postaugment_traindata_labels_emo = gen_embed_traindata_labels_emo + postaugment_traindata_labels_emo_gens

    gen_embed_traindata_embs = np.array(gen_embed_traindata_embs)
    gen_embed_valdata_embs = np.array(gen_embed_valdata_embs)

    scaler = StandardScaler()
    scaler.fit(gen_embed_traindata_embs)
    gen_embed_traindata_embs = scaler.transform(gen_embed_traindata_embs)

    scaler2 = StandardScaler()
    scaler2.fit(gen_embed_valdata_embs)
    gen_embed_valdata_embs = scaler2.transform(gen_embed_valdata_embs)

    gen_embed_traindata_labels_encoded_pose = gen_embed_traindata_labels_pose
    gen_embed_traindata_labels_encoded_id = gen_embed_traindata_labels_id
    gen_embed_traindata_labels_encoded_emo = gen_embed_traindata_labels_emo
    gen_embed_valdata_labels_encoded_pose = gen_embed_valdata_labels_pose
    gen_embed_valdata_labels_encoded_id = gen_embed_valdata_labels_id
    gen_embed_valdata_labels_encoded_emo = gen_embed_valdata_labels_emo

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_embs.npy", gen_embed_traindata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_pose.npy", gen_embed_traindata_labels_encoded_pose)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_embs.npy", gen_embed_valdata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_pose.npy", gen_embed_valdata_labels_encoded_pose)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_id.npy", gen_embed_traindata_labels_encoded_id)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_id.npy", gen_embed_valdata_labels_encoded_id)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_emo.npy", gen_embed_traindata_labels_encoded_emo)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_emo.npy", gen_embed_valdata_labels_encoded_emo)

else:
    gen_embed_traindata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_embs.npy")
    gen_embed_traindata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_pose.npy")
    gen_embed_valdata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_embs.npy")
    gen_embed_valdata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_pose.npy")

    gen_embed_traindata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_id.npy")
    gen_embed_valdata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_id.npy")

    gen_embed_traindata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_traindata_labels_encoded_emo.npy")
    gen_embed_valdata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/gen_embed_valdata_labels_encoded_emo.npy")

# print("gen_embed_traindata_embs.shape:", gen_embed_traindata_embs.shape)
# print("gen_embed_valdata_embs.shape:", gen_embed_valdata_embs.shape)

accuracy_pose = []
accuracy_id = []
accuracy_emotion = []
val_accuracy_pose = []
val_accuracy_id = []
val_accuracy_emotion = []

clf_pose = SVC(probability=True)
clf_pose.fit(gen_embed_traindata_embs, gen_embed_traindata_labels_encoded_pose)
accuracy_pose.append(np.mean(clf_pose.predict(gen_embed_traindata_embs) == gen_embed_traindata_labels_encoded_pose))
gen_embed_valdata_labels_encoded_pose_pred = clf_pose.predict(gen_embed_valdata_embs)
val_accuracy_pose.append(np.mean(gen_embed_valdata_labels_encoded_pose_pred == gen_embed_valdata_labels_encoded_pose))

clf_id = SVC(probability=True)
clf_id.fit(gen_embed_traindata_embs, gen_embed_traindata_labels_encoded_id)
accuracy_id.append(np.mean(clf_id.predict(gen_embed_traindata_embs) == gen_embed_traindata_labels_encoded_id))
gen_embed_valdata_labels_encoded_id_pred = clf_id.predict(gen_embed_valdata_embs)
val_accuracy_id.append(np.mean(gen_embed_valdata_labels_encoded_id_pred == gen_embed_valdata_labels_encoded_id))

clf_emo = SVC(probability=True)
clf_emo.fit(gen_embed_traindata_embs, gen_embed_traindata_labels_encoded_emo)
accuracy_emotion.append(np.mean(clf_emo.predict(gen_embed_traindata_embs) == gen_embed_traindata_labels_encoded_emo))
gen_embed_valdata_labels_encoded_emo_pred = clf_emo.predict(gen_embed_valdata_embs)
val_accuracy_emotion.append(np.mean(gen_embed_valdata_labels_encoded_emo_pred == gen_embed_valdata_labels_encoded_emo))

print("Objective: Generated Embeddings")
print("Avg ACC pose:", np.mean(accuracy_pose), "Avg ACC id:", np.mean(accuracy_id), "Avg ACC emotion:", np.mean(accuracy_emotion))
print("Avg ACC pose (Val):", np.mean(val_accuracy_pose), "Avg ACC id (Val):", np.mean(val_accuracy_id), "Avg ACC emotion (Val):", np.mean(val_accuracy_emotion))

plot_metrics(gen_embed_valdata_embs, gen_embed_valdata_labels_encoded_pose, gen_embed_valdata_labels_encoded_pose_pred, clf_pose, "pose", "gen_embed")
plot_metrics(gen_embed_valdata_embs, gen_embed_valdata_labels_encoded_id, gen_embed_valdata_labels_encoded_id_pred, clf_id, "id", "gen_embed")
plot_metrics(gen_embed_valdata_embs, gen_embed_valdata_labels_encoded_emo, gen_embed_valdata_labels_encoded_emo_pred, clf_emo, "emo", "gen_embed")


if not load_pickles:
    postaugment_traindata_embs = np.array(postaugment_traindata_embs

    scaler = StandardScaler()
    scaler.fit(postaugment_traindata_embs)
    postaugment_traindata_embs = scaler.transform(postaugment_traindata_embs)

    postaugment_valdata_embs = gen_embed_valdata_embs
    
    postaugment_traindata_labels_encoded_pose = postaugment_traindata_labels_pose
    postaugment_traindata_labels_encoded_id = postaugment_traindata_labels_id
    postaugment_traindata_labels_encoded_emo = postaugment_traindata_labels_emo
    postaugment_valdata_labels_encoded_pose = gen_embed_valdata_labels_encoded_pose
    postaugment_valdata_labels_encoded_id = gen_embed_valdata_labels_encoded_id
    postaugment_valdata_labels_encoded_emo = gen_embed_valdata_labels_encoded_emo

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_embs.npy", postaugment_traindata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_pose.npy", postaugment_traindata_labels_encoded_pose)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_embs.npy", postaugment_valdata_embs)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_pose.npy", postaugment_valdata_labels_encoded_pose)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_id.npy", postaugment_traindata_labels_encoded_id)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_id.npy", postaugment_valdata_labels_encoded_id)

    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_emo.npy", postaugment_traindata_labels_encoded_emo)
    np.save(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_emo.npy", postaugment_valdata_labels_encoded_emo)

else:
    postaugment_traindata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_embs.npy")
    postaugment_traindata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_pose.npy")
    postaugment_valdata_embs = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_embs.npy")
    postaugment_valdata_labels_encoded_pose = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_pose.npy")

    postaugment_traindata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_id.npy")
    postaugment_valdata_labels_encoded_id = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_id.npy")

    postaugment_traindata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_traindata_labels_encoded_emo.npy")
    postaugment_valdata_labels_encoded_emo = np.load(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_objectives_npys/postaugment_valdata_labels_encoded_emo.npy")

# print("postaugment_traindata_embs.shape", postaugment_traindata_embs.shape)
# print("postaugment_valdata_embs.shape", postaugment_valdata_embs.shape)

accuracy_pose = []
accuracy_id = []
accuracy_emotion = []
val_accuracy_pose = []
val_accuracy_id = []
val_accuracy_emotion = []

clf_pose = SVC(probability=True)
clf_pose.fit(postaugment_traindata_embs, postaugment_traindata_labels_encoded_pose)
accuracy_pose.append(np.mean(clf_pose.predict(postaugment_traindata_embs) == postaugment_traindata_labels_encoded_pose))
postaugment_valdata_labels_encoded_pose_pred = clf_pose.predict(postaugment_valdata_embs)
val_accuracy_pose.append(np.mean(postaugment_valdata_labels_encoded_pose_pred == postaugment_valdata_labels_encoded_pose))

clf_id = SVC(probability=True)
clf_id.fit(postaugment_traindata_embs, postaugment_traindata_labels_encoded_id)
accuracy_id.append(np.mean(clf_id.predict(postaugment_traindata_embs) == postaugment_traindata_labels_encoded_id))
postaugment_valdata_labels_encoded_id_pred = clf_id.predict(postaugment_valdata_embs)
val_accuracy_id.append(np.mean(postaugment_valdata_labels_encoded_id_pred == postaugment_valdata_labels_encoded_id))

clf_emo = SVC(probability=True)
clf_emo.fit(postaugment_traindata_embs, postaugment_traindata_labels_encoded_emo)
accuracy_emotion.append(np.mean(clf_emo.predict(postaugment_traindata_embs) == postaugment_traindata_labels_encoded_emo))
postaugment_valdata_labels_encoded_emo_pred = clf_emo.predict(postaugment_valdata_embs)
val_accuracy_emotion.append(np.mean(postaugment_valdata_labels_encoded_emo_pred == postaugment_valdata_labels_encoded_emo))

print("Objective: Post-Augmentation")
print("Avg ACC pose:", np.mean(accuracy_pose), "Avg ACC id:", np.mean(accuracy_id), "Avg ACC emotion:", np.mean(accuracy_emotion))
print("Avg ACC pose (Val):", np.mean(val_accuracy_pose), "Avg ACC id (Val):", np.mean(val_accuracy_id), "Avg ACC emotion (Val):", np.mean(val_accuracy_emotion))

plot_metrics(postaugment_valdata_embs, postaugment_valdata_labels_encoded_pose, postaugment_valdata_labels_encoded_pose_pred, clf_pose, "pose", "postaugment")
plot_metrics(postaugment_valdata_embs, postaugment_valdata_labels_encoded_id, postaugment_valdata_labels_encoded_id_pred, clf_id, "id", "postaugment")
plot_metrics(postaugment_valdata_embs, postaugment_valdata_labels_encoded_emo, postaugment_valdata_labels_encoded_emo_pred, clf_emo, "emo", "postaugment")


with open(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_loss_log.txt") as file:
    logs = file.readlines()
    logs = list(map(lambda item: re.findall(r"Epoch\s(\d+)\s/\s\d+\sLoss\s([\d.]+)\s\(([\d.]+),\s([\d.]+)\)\sVal_Loss\s([\d.]+)\s\(([\d.]+),\s([\d.]+)\)\sLR\s(.+)\n", item)[0], logs))
    logs = [list(map(float, item)) for item in logs]
    
logs = np.array(logs)

fig = plt.figure(figsize=(10, 10))
plt.plot(logs[:, 0], logs[:, 1], label="Train loss")
plt.plot(logs[:, 0], logs[:, 4], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss Error")
plt.legend()
plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_total_loss.png')

fig = plt.figure(figsize=(10, 10))
plt.plot(logs[:, 0], logs[:, 2], label="Train BCE loss")
plt.plot(logs[:, 0], logs[:, 5], label="Validation BCE loss")

plt.xlabel("Epoch")
plt.ylabel("Loss Error")
plt.legend()
plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_bce_loss.png')

fig = plt.figure(figsize=(10, 10))
plt.plot(logs[:, 0], logs[:, 3], label="Train Npair loss")
plt.plot(logs[:, 0], logs[:, 6], label="Validation Npair loss")

plt.xlabel("Epoch")
plt.ylabel("Loss Error")
plt.legend()
plt.savefig(f'{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_npair_loss.png')