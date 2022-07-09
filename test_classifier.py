    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []

    # testing loop
    for img, label in loader_test:
        img = img.to(device)
        label = label.to(device)

        # predict
        C_out = C(img)

        # calculate metrics
        accuracy = (C_out, label)
        f1_score = (C_out, label)
        precision = (C_out, label)
        recall = (C_out, label)

        # append batch of metrics
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)

    # calculate average metrics over all batches (single results in the container)
    avg_acc = np.mean(accuracy_list)
    avg_f1 = np.mean(f1_score_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    # save final test metrics
    logger.info(f"Average accuracy {avg_acc}")
    logger.info(f"Average f1_score {avg_f1}")
    logger.info(f"Average precision {avg_precision}")
    logger.info(f"Average recall {avg_recall}")
