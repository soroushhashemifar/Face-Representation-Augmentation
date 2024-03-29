from dataset import CustomDataSet
from model_utils import EmbeddingGeneratorDecoder_VIT
import torch.nn.functional as F
import datetime
import numpy as np
import torch
import tqdm
import argparse
import os


class ConvAutoencoder(torch.nn.Module):
    
    def __init__(self, EMBEDDING_SOURCE, dropout_rate):
        super(ConvAutoencoder, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.decoder_emb_generator = EmbeddingGeneratorDecoder_VIT(EMBEDDING_SOURCE=EMBEDDING_SOURCE)
        
        # Magface
        self.encoder_layer1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.encoder_layer2 = torch.nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.encoder_layer3 = torch.nn.Conv2d(16, 64, 3, stride=1, padding=1)
        self.encoder_layer4 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder_layer5 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder_layer6 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = torch.nn.Upsample(scale_factor=3, mode='nearest')
        self.upsample4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        
        self.decoder_layer1 = torch.nn.Conv2d(512, 256, 2, stride=3, padding=2)
        self.decoder_layer2 = torch.nn.Conv2d(256, 128, 3, stride=1, padding=0)
        self.decoder_layer3 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder_layer4 = torch.nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.decoder_layer5 = torch.nn.Conv2d(16, 4, 3, stride=1, padding=1)
        self.decoder_layer6 = torch.nn.Conv2d(4, 1, 3, stride=1, padding=1)

        # if EMBEDDING_SOURCE == "sphereface" or EMBEDDING_SOURCE == "cosface":
        #     # self.prelu = torch.nn.PReLU(512)
        #     self.fc = torch.nn.Linear(512, 512)
        #     self.normalizer = torch.nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def encoder(self, x): # 1, 112, 112
        x = self.encoder_layer1(x) # 4, 112, 112
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 4, 56, 56
        
        x = self.encoder_layer2(x) # 16, 56, 56
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 16, 28, 28
        
        x = self.encoder_layer3(x) # 64, 28, 28
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 64, 14, 14
        
        x = self.encoder_layer4(x) #128, 14, 14
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 128, 7, 7
        
        x = self.encoder_layer5(x) #256, 7, 7
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 256, 3, 3
        
        x = self.encoder_layer6(x) #512, 3, 3
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # 512, 1, 1
        
        x = torch.reshape(x, (-1, 512))

        return x
        
    def decoder_pose_reconstructor(self, x):
        x = torch.reshape(x, (-1, 512, 1, 1))
        
        x = self.upsample4(x) # 512, 4, 4
        x = self.decoder_layer1(x) # 256, 3, 3
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        
        x = self.upsample3(x) # 256, 9, 9
        x = self.decoder_layer2(x) # 128, 7, 7
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        
        x = self.upsample2(x) # 128, 14, 14
        x = self.decoder_layer3(x) # 64, 14, 14
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        
        x = self.upsample2(x) # 64, 28, 28
        x = self.decoder_layer4(x) # 16, 28, 28
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        
        x = self.upsample2(x) # 16, 56, 56
        x = self.decoder_layer5(x) # 4, 56, 56
        x = F.dropout(x, p=self.dropout_rate)
        x = F.relu(x)
        
        x = self.upsample2(x) # 4, 112, 112
        x = self.decoder_layer6(x) # 1, 112, 112
        x = torch.sigmoid(x)
        
        return x
        
    def forward(self, pose_img, original_embedding):
        coded = self.encoder(pose_img)
        reconstructed_pose = self.decoder_pose_reconstructor(coded)
        
        # Feature fusion
        generated_embedding = self.decoder_emb_generator(coded, original_embedding)

        # if EMBEDDING_SOURCE == "sphereface" or EMBEDDING_SOURCE == "cosface":
        #     # generated_embedding = self.prelu(generated_embedding)
        #     generated_embedding = self.fc(generated_embedding)
        #     generated_embedding = self.normalizer(generated_embedding)
        
        id_predictions = None
        emo_predictions = None
        
        return reconstructed_pose, generated_embedding, (id_predictions, emo_predictions)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-source")
    parser.add_argument("--num-epochs")
    parser.add_argument("--batch-size")
    parser.add_argument("--results-path")
    parser.add_argument("--learning-rate")
    parser.add_argument("--dropout-rate")
    args = parser.parse_args()
    config = vars(args)

    EMBEDDING_SOURCE = config["embedding_source"]
    RESULTS_PATH = config["results_path"]
    NUM_EPOCHS = int(config["num_epochs"])
    BATCHSIZE = int(config["batch_size"])

    train_dataset = CustomDataSet(type_="train", EMBEDDING_SOURCE=EMBEDDING_SOURCE, RESULTS_PATH=RESULTS_PATH)
    val_dataset = CustomDataSet(type_="val", EMBEDDING_SOURCE=EMBEDDING_SOURCE, RESULTS_PATH=RESULTS_PATH)
    print(len(train_dataset), len(val_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True)

    net = ConvAutoencoder(EMBEDDING_SOURCE, float(config["dropout_rate"]))
    net = net.to("cuda")
    
    criterion_bce = torch.nn.BCELoss(reduction='mean')
    criterion_triplet_loss = torch.nn.TripletMarginLoss(margin=10.0, p=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-7)

    if not os.path.isdir(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results"):
        os.makedirs(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results")
    
    historical_loss = []
    historical_val_loss = []
    best_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        # training
        losses1 = []
        losses2 = []
        losses = []
        for data in tqdm.tqdm(train_dataloader):
            pose_img = data[0]
            #print(pose_img.shape)
            pose_img = pose_img.permute(0, 3, 1, 2)
            pose_img = pose_img.to("cuda").to(torch.float32)

            input_emb = data[1]
            input_emb = input_emb.to("cuda").to(torch.float32)
            # print(input_emb.shape)

            output_emb = data[2]
            output_emb = output_emb.to("cuda").to(torch.float32)

            negative_id_embedding = data[3]
            negative_id_embedding = negative_id_embedding.to("cuda").to(torch.float32)

            negative_pose_embedding = data[4]
            negative_pose_embedding = negative_pose_embedding.to("cuda").to(torch.float32)

            negative_emo_embedding = data[5]
            negative_emo_embedding = negative_emo_embedding.to("cuda").to(torch.float32)

            optimizer.zero_grad()

            reconstructed_pose, generated_emb, (id_predictions, emo_predictions) = net(pose_img, input_emb)

            # BCE Loss
            loss1 = criterion_bce(reconstructed_pose, pose_img)
            losses1.append(loss1.item())

            # N-pair Loss
            loss2 = criterion_triplet_loss(generated_emb, output_emb, negative_id_embedding) + criterion_triplet_loss(generated_emb, output_emb, negative_pose_embedding) + criterion_triplet_loss(generated_emb, output_emb, negative_emo_embedding)
            losses2.append(loss2.item())

            loss = loss1 + loss2
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        with torch.no_grad():
            val_losses1 = []
            val_losses2 = []
            val_losses = []
            for val_data in tqdm.tqdm(val_dataloader):
                pose_img = val_data[0]
                pose_img = pose_img.permute(0, 3, 1, 2)
                pose_img = pose_img.to("cuda").to(torch.float32)

                input_emb = val_data[1]
                input_emb = input_emb.to("cuda").to(torch.float32)

                output_emb = val_data[2]
                output_emb = output_emb.to("cuda").to(torch.float32)

                negative_id_embedding = val_data[3]
                negative_id_embedding = negative_id_embedding.to("cuda").to(torch.float32)

                negative_pose_embedding = val_data[4]
                negative_pose_embedding = negative_pose_embedding.to("cuda").to(torch.float32)

                negative_emo_embedding = val_data[5]
                negative_emo_embedding = negative_emo_embedding.to("cuda").to(torch.float32)

                reconstructed_pose, generated_emb, (id_predictions, emo_predictions) = net(pose_img, input_emb)

                # BCE Loss
                val_loss1 = criterion_bce(reconstructed_pose, pose_img)
                val_losses1.append(val_loss1.item())

                # N-pair Loss
                val_loss2 = criterion_triplet_loss(generated_emb, output_emb, negative_id_embedding) + criterion_triplet_loss(generated_emb, output_emb, negative_pose_embedding) + criterion_triplet_loss(generated_emb, output_emb, negative_emo_embedding)
                val_losses2.append(val_loss2.item())

                val_loss = val_loss1 + val_loss2
                val_losses.append(val_loss.item())

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print("Epoch", epoch+1, "/", NUM_EPOCHS, 
              "Loss", round(np.mean(losses), 4), f"({round(np.mean(losses1), 4)}, {round(np.mean(losses2), 4)})", 
              "Val_Loss", round(np.mean(val_losses), 4), f"({round(np.mean(val_losses1), 4)}, {round(np.mean(val_losses2), 4)})", 
              "LR", current_lr)

        historical_loss.append((np.mean(losses), np.mean(losses1), np.mean(losses2)))
        historical_val_loss.append((np.mean(val_losses), np.mean(val_losses1), np.mean(val_losses2)))

        with open(f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_loss_log.txt", "a") as file:
            file.write(f"Epoch {epoch+1} / {NUM_EPOCHS} Loss {round(np.mean(losses), 4)} ({round(np.mean(losses1), 4)}, {round(np.mean(losses2), 4)}) Val_Loss {round(np.mean(val_losses), 4)} ({round(np.mean(val_losses1), 4)}, {round(np.mean(val_losses2), 4)}) LR {current_lr}\n")

        if historical_val_loss[-1][0] < best_loss:
            best_loss = historical_val_loss[-1][0]
            torch.save(net.state_dict(), f"{RESULTS_PATH}/{EMBEDDING_SOURCE}_results/{EMBEDDING_SOURCE}_augmentor.pt")
            print(f"Model saved @ {datetime.datetime.now()}, Loss = {round(historical_loss[-1][0], 4)}, Val_Loss = {round(historical_val_loss[-1][0], 4)}")
