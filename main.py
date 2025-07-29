from trainers.train import pretrain_contrastive, pretrain_multitask

if __name__ == "__main__":
    # Step 1: Contrastive Pretraining
    pretrain_contrastive()

    # Step 2: Multitask Supervised Pretraining
    pretrain_multitask()
