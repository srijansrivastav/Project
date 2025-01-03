from __future__ import print_function
import os, time, cv2, sys, math, csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time, datetime
import argparse
import random
import osgeo.gdal
import tf_slim as slim


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="test", help='Select "train" or "test" mode. ')
parser.add_argument('--dataset', type=str, default="L7", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=2048, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=2048, help='Width of cropped input image to network')
parser.add_argument('--step_height', type=int, default=2048, help='Height of cropped input image to network')
parser.add_argument('--step_width', type=int, default=2048, help='Width of cropped input image to network')
parser.add_argument('--num_samples', type=int, default=100, help='Width of cropped input image to network')
parser.add_argument('--dropout', type=float, default=0.5, help='Width of cropped input image to network')
parser.add_argument('--learningrate', type=float, default=0.0001, help='Width of cropped input image to network')
parser.add_argument('--decay', type=float, default=0.95, help='decay')
parser.add_argument('--netindex', type=int, default=13, help='layers and depths')
parser.add_argument('--validate', type=str2bool, default=False, help='Perform validation after each epoch')


args = parser.parse_args()

def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

def compareImage(gt, label, class_labels_list):
    num_classes = len(class_labels_list)
    count = np.zeros((num_classes + 1, num_classes + 1))

    for i in range(num_classes):
        # Compare directly against the predicted labels (0, 1)
        lbi = (label == i).astype(np.uint8)  # Binary mask for predicted class i

        for j in range(num_classes):
            gtj = (gt == j).astype(np.uint8)  # Binary mask for ground truth class j
            cij = cv2.bitwise_and(lbi, gtj)  # Intersection of predictions and ground truth
            count[i][j] = np.sum(cij)  # Count the number of matches
            count[i][num_classes] += count[i][j]
            count[num_classes][j] += count[i][j]

    for j in range(num_classes):
        count[num_classes][num_classes] += count[num_classes][j]

    return count



def outputcount(target,count,class_names_list,id):
    num_classes=len(class_names_list)

    Recall=np.zeros(num_classes)
    Precision=np.zeros(num_classes)
    Fscore=np.zeros(num_classes)

    target.write("%s,"%id)
    for j in range(0, num_classes):
        target.write("%s," % class_names_list[j])
    target.write("Accuracy\n")
    for i in range(0, num_classes):
        target.write("%s," % class_names_list[i])
        for j in range(0, num_classes):
            target.write("%d," % count[i][j])
        if count[i][num_classes]==0:
            Precision[i]==0
        else:
            Precision[i]=1.0*count[i][i]/count[i][num_classes]
        target.write("%f\n" %(Precision[i]))
    target.write("Accuracy,")
    corr=0
    for j in range(0, num_classes):
        corr=corr+count[j][j]
        if count[num_classes][j]==0:
            Recall[j]=0
        else:
            Recall[j]=1.0*count[j][j]/count[num_classes][j]
        target.write("%f," %(Recall[j]))
    target.write("%f\n" %(1.0*corr/count[num_classes][num_classes]))
    target.write("Fscore,")
    for j in range(0, num_classes):
        den=Precision[j]+Recall[j]
        if den < 0.00000001:
            Fscore[j]=0
        else:
            Fscore[j]=2.0*(Precision[j]*Recall[j])/den
        target.write("%f," %(Fscore[j]))
    target.write(",\n")
    return Recall, Precision, Fscore
def compute_accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == labels)
    total_predictions = np.prod(labels.shape)
    return correct_predictions / total_predictions
def Drawcurve(num_epochs, avg_scores_per_epoch, avg_loss_per_epoch, path):
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    num_epochs=len(avg_loss_per_epoch)
    
    ax1.plot(range(num_epochs), avg_scores_per_epoch, linewidth=3)
    ax1.set_title("Average validation accuracy vs epochs", fontsize=24)
    ax1.set_xlabel("Epoch", fontsize=24)
    ax1.set_ylabel("Avg. val. accuracy", fontsize=24)

    plt.savefig("%s/%s"%(path, 'accuracy_vs_epochs.png'))

    plt.clf()

    ax1 = fig.add_subplot(111)
    
    ax1.plot(range(num_epochs), avg_loss_per_epoch, linewidth=3)
    ax1.set_title("Average loss vs epochs", fontsize=24)
    ax1.set_xlabel("Epoch", fontsize=24)
    ax1.set_ylabel("Current loss", fontsize=24)

    plt.savefig("%s/%s"%(path, 'loss_vs_epochs.png'))

    return

def build_net_smooth(img, dropout_p=0.5, scope=None):
    net = img
    net = slim.conv2d(net, 64, [1,1], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(net)
    if dropout_p != 0.0:
        net = slim.dropout(net, keep_prob=(1.0-dropout_p))

    net = slim.conv2d(net, 2, [1, 1], padding='VALID', activation_fn=None, normalizer_fn=None)
    net = slim.conv2d(net, 2, [3, 3], padding='VALID', activation_fn=None, normalizer_fn=None)
    label = tf.argmax(net, 3)

    return net, label
    
#=============================================================================
base = r"C:\Users\wr826375\Downloads\SCNN-main\L8\L8T"
def load_all_images_scene_l8(base, mode="train"):
    if mode == "train":
        data_path = os.path.join(base, "Train")
    elif mode == "test":
        data_path = os.path.join(base, "Test")
    else:
        raise ValueError("Mode should be either 'train' or 'test'.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    covernames = os.listdir(data_path)
    for covername in covernames:
        cover_path = os.path.join(data_path, covername)
        if not os.path.isdir(cover_path):
            continue

        print(f"Processing cover: {covername}")

        mv = []
        image_shapes = []
        for file in sorted(os.listdir(cover_path)):
            if file.endswith(".TIF") and "_BQA" not in file and "mask" not in file:
                imgfile = os.path.join(cover_path, file)
                srcRaster = osgeo.gdal.Open(imgfile)
                img = srcRaster.ReadAsArray()

                # Debug: Check the image size and depth
                #print(f"Loaded image from {file} with shape: {img.shape}, dtype: {img.dtype}")
                image_shapes.append(img.shape)
                mv.append(img)

        # Ensure all images are resized to match the smallest dimensions
        min_height = min([shape[0] for shape in image_shapes])
        min_width = min([shape[1] for shape in image_shapes])

        resized_mv = []
        for img in mv:
            resized_img = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
            resized_mv.append(resized_img)

        if len(set([img.shape for img in resized_mv])) > 1:
            print(f"Error: Inconsistent image sizes even after resizing for cover {covername}. Skipping.")
            continue

        if not resized_mv:
            print(f"No valid image files found for cover {covername}. Skipping.")
            continue

        # Now merge the resized images
        try:
            image = cv2.merge(resized_mv)
        except Exception as e:
            print(f"Failed to merge images for cover {covername}: {str(e)}")
            continue

        # Load BQA (Quality Assessment) file
        bqa_file = [os.path.join(cover_path, file) for file in os.listdir(cover_path) if "_BQA.TIF" in file]
        if not bqa_file:
            #print(f"No BQA file found in {cover_path}. Skipping this cover.")
            continue
        bqa_file = bqa_file[0]
        bqaRaster = osgeo.gdal.Open(bqa_file)
        label_bqa = bqaRaster.ReadAsArray()
        transform_label = bqaRaster.GetGeoTransform()

        # Load the mask file
        mask_file = [os.path.join(cover_path, file) for file in os.listdir(cover_path) if "mask" in file]
        if not mask_file:
            print(f"No mask file found in {cover_path}. Skipping this cover.")
            continue
        mask_file = mask_file[0]
        gtRaster = osgeo.gdal.Open(mask_file)
        label_gt = gtRaster.ReadAsArray()
        transform_gt = gtRaster.GetGeoTransform()
        height, width = label_gt.shape

        # Calculate the spatial alignment between the label and mask
        x0 = int((transform_label[0] - transform_gt[0]) / transform_label[1])
        y0 = int((transform_label[3] - transform_gt[3]) / transform_label[5])
        x1 = x0 + bqaRaster.RasterXSize
        y1 = y0 + bqaRaster.RasterYSize

        c0, c1 = 0, x1 - x0
        r0, r1 = 0, y1 - y0
        if x0 < 0:
            c0 = -x0
            x0 = 0
        if y0 < 0:
            r0 = -y0
            y0 = 0
        if x1 > width:
            c1 += width - x1
            x1 = width
        if y1 > height:
            r1 += height - y1
            y1 = height

        # Yield the loaded image and all necessary labels/coordinates for each covername
        yield image, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1
   
        
def Test_Scene_l8(testpath, testset, mode="train"):
    # Directly specify the test directory
    test_dir = r"C:\Users\wr826375\Downloads\SCNN-main\L8\L8T\Test"

    if not os.path.exists(test_dir):
        print(f"Test directory does not exist: {test_dir}")
        return

    covernames = os.listdir(test_dir)
    print(f"Processing covers in directory: {test_dir}")

    covername = None
    target = open(os.path.join(test_dir, f"{testset}_scores.csv"), 'w')
    covertarget = None
    covercount = np.zeros((num_classes + 1, num_classes + 1))
    totalcount = np.zeros((num_classes + 1, num_classes + 1))
    totalimage = 0
    totaltime = 0
    all_test_predictions = []
    all_test_labels = []
    for covername in covernames:
        cover_path = os.path.join(test_dir, covername)
        if not os.path.isdir(cover_path):
            continue

        print(f"Processing cover: {covername}")

        # Load images using the load_all_images_scene_l7 function
        for image_orig, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1 in load_all_images_scene_l8(base, mode="train"):
            print(f"Loaded image shape: {image_orig.shape}, Ground truth shape: {label_gt.shape}")
            print(f"Unique values in ground truth labels before filtering: {np.unique(label_gt)}")
            print(f"Unique values in BQA labels: {np.unique(label_bqa)}")

            # Resize the labels to match the image dimensions
            label_gt_resized = cv2.resize(label_gt, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
            label_bqa_resized = cv2.resize(label_bqa, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)

            print(f"Resized Ground truth shape: {label_gt_resized.shape}, Resized BQA shape: {label_bqa_resized.shape}")

            h, w, num_bands = image_orig.shape

    # Adjust the target image array to accommodate the number of bands dynamically
            if num_bands < 11:
        # If the image has fewer bands, fill the remaining channels with zeros
                image = np.zeros([h, w, 11], np.float32)
                image[:, :, :num_bands] = image_orig
            elif num_bands > 11:
            # If the image has more bands, truncate it to the first 7 bands
                image = image_orig[:, :, :11]
            else:
        # If the image has exactly 7 bands, no adjustment is needed
                image = image_orig

            pred_label = np.zeros([h, w], np.uint8)
            # Normalize the image
            for j in range(7):
                if j < len(mean) and j < len(sdev):
                    image[:, :, j] = (image[:, :, j] - mean[j]) / sdev[j]           

            # Copy the existing channels into the first channels of the image
         
            yrang = list(range(crop_height_2, h - crop_height - crop_height_2, crop_height))
            yrang.append(h - crop_height - crop_height_2)
            xrang = list(range(crop_width_2, w - crop_width - crop_width_2, crop_width))
            xrang.append(w - crop_width - crop_width_2)


            # Run the model on each cropped section of the image
            for y in yrang:
                for x in xrang:
                    img = image[y - crop_height_2:y + crop_height + crop_height_2, x - crop_width_2:x + crop_width + crop_width_2]
                    image_batch = np.expand_dims(img, axis=0)
                    output_label = sess.run(predlabel, feed_dict={input: image_batch})
                    output_label = np.uint8((output_label[0, :, :] + 1) * 50)
                    pred_label[y:y + crop_height, x:x + crop_width] = output_label

            label_pred = pred_label
            pred_label_flat = pred_label.flatten()
            label_gt_resized_flat = label_gt_resized.flatten()

            # Accumulate predictions and labels
            all_test_predictions.append(pred_label_flat)
            all_test_labels.append(label_gt_resized_flat)
            # Prepare ground truth and prediction labels
            label_gt_resized = (label_gt_resized == 64) * 50 + (label_gt_resized == 128) * 50 + (label_gt_resized == 191) * 100 + (label_gt_resized == 192) * 100 + (label_gt_resized == 255) * 100
            label_gt_resized = np.uint8(label_gt_resized)

            label_bqa_resized = label_bqa_resized[r0:r1, c0:c1]
            label_pred = label_pred[r0:r1, c0:c1]
            label_gt_resized = label_gt_resized[y0:y1, x0:x1]
            label_gt_resized = label_gt_resized * (label_bqa_resized != 1)
            label_pred = label_pred * (label_gt_resized > 0)

            print(f"Unique values in ground truth labels after filtering: {np.unique(label_gt_resized)}")
            print(f"Unique values in predicted labels: {np.unique(label_pred)}")

            valid_pixels = np.count_nonzero(label_gt_resized)
            print(f"Valid pixels after filtering for cover {covername}: {valid_pixels}")
            if valid_pixels == 0:
                print(f"Cover {covername}: No valid pixels after label filtering.")
                continue

            print(f"Valid pixels found, proceeding with evaluation for {covername}.")

            # Calculate time
            st = time.time()
            totaltime += time.time() - st
            totalimage += 1

            # Compare the predicted labels to ground truth
            scenecount = compareImage(label_gt_resized, label_pred, class_labels_list)
            outputcount(target, scenecount, class_names_list, covername)

            if covertarget is None:
                covertarget = open(os.path.join(test_dir, f"{testset}_scores_{covername}.csv"), 'w')
            elif covername != os.path.basename(cover_path):
                outputcount(target, covercount, class_names_list, covername)
                totalcount += covercount
                covercount = np.zeros((num_classes + 1, num_classes + 1))

                covertarget.close()
                covertarget = open(os.path.join(test_dir, f"{testset}_scores_{covername}.csv"), 'w')

            outputcount(covertarget, scenecount, class_names_list, covername)
            covercount += scenecount
            
       
    outputcount(target, covercount, class_names_list, covername)
    totalcount += covercount
    
    all_test_predictions = np.concatenate(all_test_predictions)
    all_test_labels = np.concatenate(all_test_labels)
    overall_test_accuracy = compute_accuracy(all_test_predictions, all_test_labels)
    print(f"Overall Testing accuracy for {testset}: {overall_test_accuracy:.6f}") 
    
    outputcount(target, totalcount, class_names_list, "Total")
    target.write("totaltime,totalimage,totaltime/totalimage\n")
    if totalimage > 0:
        target.write(f"{totaltime},{totalimage},{totaltime/totalimage}\n")
    else:
        target.write("No valid images were processed.\n")
    if covertarget:
        covertarget.close()
    target.close()
    return totalcount
 
        
#=============================================================================

base_path = r"C:\Users\wr826375\Downloads\SCNN-main\L7\L7T"

def load_all_images_scene_l7(base_path, mode="test"):
    if mode == "train":
        data_path = os.path.join(base_path, "Train")
    elif mode == "test":
        data_path = os.path.join(base_path, "Test")
    else:
        raise ValueError("Mode should be either 'train' or 'test'.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    covernames = os.listdir(data_path)
    for covername in covernames:
        cover_path = os.path.join(data_path, covername)
        if not os.path.isdir(cover_path):
            continue

        print(f"Processing cover: {covername}")

        mv = []
        image_shapes = []
        for file in sorted(os.listdir(cover_path)):
            if file.endswith(".TIF") and "_BQA" not in file and "mask" not in file:
                imgfile = os.path.join(cover_path, file)
                srcRaster = osgeo.gdal.Open(imgfile)
                img = srcRaster.ReadAsArray()

                # Debug: Check the image size and depth
                #print(f"Loaded image from {file} with shape: {img.shape}, dtype: {img.dtype}")
                image_shapes.append(img.shape)
                mv.append(img)

        # Ensure all images are resized to match the smallest dimensions
        min_height = min([shape[0] for shape in image_shapes])
        min_width = min([shape[1] for shape in image_shapes])

        resized_mv = []
        for img in mv:
            resized_img = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
            resized_mv.append(resized_img)

        if len(set([img.shape for img in resized_mv])) > 1:
            print(f"Error: Inconsistent image sizes even after resizing for cover {covername}. Skipping.")
            continue

        if not resized_mv:
            print(f"No valid image files found for cover {covername}. Skipping.")
            continue

        # Now merge the resized images
        try:
            image = cv2.merge(resized_mv)
        except Exception as e:
            print(f"Failed to merge images for cover {covername}: {str(e)}")
            continue

        # Load BQA (Quality Assessment) file
        bqa_file = [os.path.join(cover_path, file) for file in os.listdir(cover_path) if "_BQA.TIF" in file]
        if not bqa_file:
            #print(f"No BQA file found in {cover_path}. Skipping this cover.")
            continue
        bqa_file = bqa_file[0]
        bqaRaster = osgeo.gdal.Open(bqa_file)
        label_bqa = bqaRaster.ReadAsArray()
        transform_label = bqaRaster.GetGeoTransform()

        # Load the mask file
        mask_file = [os.path.join(cover_path, file) for file in os.listdir(cover_path) if "mask" in file]
        if not mask_file:
            print(f"No mask file found in {cover_path}. Skipping this cover.")
            continue
        mask_file = mask_file[0]
        gtRaster = osgeo.gdal.Open(mask_file)
        label_gt = gtRaster.ReadAsArray()
        transform_gt = gtRaster.GetGeoTransform()
        height, width = label_gt.shape

        # Calculate the spatial alignment between the label and mask
        x0 = int((transform_label[0] - transform_gt[0]) / transform_label[1])
        y0 = int((transform_label[3] - transform_gt[3]) / transform_label[5])
        x1 = x0 + bqaRaster.RasterXSize
        y1 = y0 + bqaRaster.RasterYSize

        c0, c1 = 0, x1 - x0
        r0, r1 = 0, y1 - y0
        if x0 < 0:
            c0 = -x0
            x0 = 0
        if y0 < 0:
            r0 = -y0
            y0 = 0
        if x1 > width:
            c1 += width - x1
            x1 = width
        if y1 > height:
            r1 += height - y1
            y1 = height

        # Yield the loaded image and all necessary labels/coordinates for each covername
        yield image, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1

        

def Test_Scene_l7(testpath, testset, mode="test"):
    # Directly specify the test directory
    test_dir = r"C:\Users\wr826375\Downloads\SCNN-main\L7\L7T\Test"

    if not os.path.exists(test_dir):
        print(f"Test directory does not exist: {test_dir}")
        return

    covernames = os.listdir(test_dir)
    print(f"Processing covers in directory: {test_dir}")

    covername = None
    target = open(os.path.join(test_dir, f"{testset}_scores.csv"), 'w')
    covertarget = None
    covercount = np.zeros((num_classes + 1, num_classes + 1))
    totalcount = np.zeros((num_classes + 1, num_classes + 1))
    totalimage = 0
    totaltime = 0
    all_test_predictions = []
    all_test_labels = []
    
    for covername in covernames:
        cover_path = os.path.join(test_dir, covername)
        if not os.path.isdir(cover_path):
            continue

        print(f"Processing cover: {covername}")

        # Load images using the load_all_images_scene_l7 function
        for image_orig, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1 in load_all_images_scene_l7(base_path, mode="test"):
            #print(f"Loaded image shape: {image_orig.shape}, Ground truth shape: {label_gt.shape}")
            #print(f"Unique values in ground truth labels before filtering: {np.unique(label_gt)}")
            #print(f"Unique values in BQA labels: {np.unique(label_bqa)}")

            # Resize the labels to match the image dimensions
            label_gt_resized = cv2.resize(label_gt, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
            label_bqa_resized = cv2.resize(label_bqa, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)

            #print(f"Resized Ground truth shape: {label_gt_resized.shape}, Resized BQA shape: {label_bqa_resized.shape}")

            h, w, num_bands = image_orig.shape

    # Adjust the target image array to accommodate the number of bands dynamically
            if num_bands < 7:
        # If the image has fewer bands, fill the remaining channels with zeros
                image = np.zeros([h, w, 7], np.float32)
                image[:, :, :num_bands] = image_orig
            elif num_bands > 7:
            # If the image has more bands, truncate it to the first 7 bands
                image = image_orig[:, :, :7]
            else:
        # If the image has exactly 7 bands, no adjustment is needed
                image = image_orig

            pred_label = np.zeros([h, w], np.uint8)
            # Normalize the image
            for j in range(7):
                if j < len(mean) and j < len(sdev):
                    image[:, :, j] = (image[:, :, j] - mean[j]) / sdev[j]           

            # Copy the existing channels into the first channels of the image
         
            yrang = list(range(crop_height_2, h - crop_height - crop_height_2, crop_height))
            yrang.append(h - crop_height - crop_height_2)
            xrang = list(range(crop_width_2, w - crop_width - crop_width_2, crop_width))
            xrang.append(w - crop_width - crop_width_2)


            # Run the model on each cropped section of the image
            for y in yrang:
                for x in xrang:
                    img = image[y - crop_height_2:y + crop_height + crop_height_2, x - crop_width_2:x + crop_width + crop_width_2]
                    image_batch = np.expand_dims(img, axis=0)
                    output_label = sess.run(predlabel, feed_dict={input: image_batch})
                    output_label = np.uint8((output_label[0, :, :] + 1) * 50)
                    pred_label[y:y + crop_height, x:x + crop_width] = output_label

            label_pred = pred_label
            pred_label_flat = pred_label.flatten()
            label_gt_resized_flat = label_gt_resized.flatten()

            # Accumulate predictions and labels
            all_test_predictions.append(pred_label_flat)
            all_test_labels.append(label_gt_resized_flat)
            # Prepare ground truth and prediction labels
            label_gt_resized = (label_gt_resized == 64) * 50 + (label_gt_resized == 128) * 50 + (label_gt_resized == 191) * 100 + (label_gt_resized == 192) * 100 + (label_gt_resized == 255) * 100
            label_gt_resized = np.uint8(label_gt_resized)

            label_bqa_resized = label_bqa_resized[r0:r1, c0:c1]
            label_pred = label_pred[r0:r1, c0:c1]
            label_gt_resized = label_gt_resized[y0:y1, x0:x1]
            label_gt_resized = label_gt_resized * (label_bqa_resized != 1)
            label_pred = label_pred * (label_gt_resized > 0)

            #print(f"Unique values in ground truth labels after filtering: {np.unique(label_gt_resized)}")
            #print(f"Unique values in predicted labels: {np.unique(label_pred)}")

            valid_pixels = np.count_nonzero(label_gt_resized)
            #print(f"Valid pixels after filtering for cover {covername}: {valid_pixels}")
            if valid_pixels == 0:
                print(f"Cover {covername}: No valid pixels after label filtering.")
                continue

            print(f"Valid pixels found, proceeding with evaluation for {covername}.")

            # Calculate time
            st = time.time()
            totaltime += time.time() - st
            totalimage += 1

            # Compare the predicted labels to ground truth
            scenecount = compareImage(label_gt_resized, label_pred, class_labels_list)
            outputcount(target, scenecount, class_names_list, covername)

            if covertarget is None:
                covertarget = open(os.path.join(test_dir, f"{testset}_scores_{covername}.csv"), 'w')
            elif covername != os.path.basename(cover_path):
                outputcount(target, covercount, class_names_list, covername)
                totalcount += covercount
                covercount = np.zeros((num_classes + 1, num_classes + 1))

                covertarget.close()
                covertarget = open(os.path.join(test_dir, f"{testset}_scores_{covername}.csv"), 'w')

            outputcount(covertarget, scenecount, class_names_list, covername)
            covercount += scenecount
            
       
    outputcount(target, covercount, class_names_list, covername)
    totalcount += covercount
    
    all_test_predictions = np.concatenate(all_test_predictions)
    all_test_labels = np.concatenate(all_test_labels)
    overall_test_accuracy = compute_accuracy(all_test_predictions, all_test_labels)
    print(f"Overall Testing accuracy for {testset}: {overall_test_accuracy:.6f}") 
    
    outputcount(target, totalcount, class_names_list, "Total")
    target.write(f"totaltime,totalimage,totaltime/totalimage\n")
    if totalimage > 0:
        target.write(f"{totaltime},{totalimage},{totaltime/totalimage}\n")
    else:
        target.write("No valid images were processed.\n")
    if covertarget:
        covertarget.close()
    target.close()
    return totalcount

#=============================================================================
def plot_predictions_vs_ground_truth(predictions, ground_truth):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predictions', alpha=0.6)
    plt.plot(ground_truth, label='Ground Truth', alpha=0.6)
    plt.title('Comparison of Predictions and Ground Truth During Training')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.show()





def calculate_coordinates(transform_label, transform_gt, bqaRaster, gtRaster):
    x0 = int((transform_label[0] - transform_gt[0]) / transform_label[1])
    y0 = int((transform_label[3] - transform_gt[3]) / transform_label[5])
    x1 = x0 + bqaRaster.RasterXSize
    y1 = y0 + bqaRaster.RasterYSize
    return x0, y0, x1, y1

def adjust_coordinates(x0, y0, x1, y1, width, height):
    c0, c1 = max(0, -x0), x1 - x0
    r0, r1 = max(0, -y0), y1 - y0
    if x1 > width:
        c1 = c1 + (width - x1)
    if y1 > height:
        r1 = r1 + (height - y1)
    return c0, c1, r0, r1

        
def SaveImages(gt_image, pred_image, filename):
    comptmp=cv2.compare(gt_image,pred_image,cv2.CMP_NE)
    cv2.imwrite("%s_gt.png"%(filename),np.uint8(gt_image))
    cv2.imwrite("%s_pred.png"%(filename),np.uint8(pred_image))
    cv2.imwrite("%s_comp.png"%(filename),np.uint8(comptmp))
    return



exppath = os.path.join(args.dataset, "Experiment")

if not os.path.isdir(exppath):
    os.makedirs(exppath)
checkpointpath = os.path.join(exppath, "checkpoints")
if not os.path.isdir(checkpointpath):
    os.makedirs(checkpointpath)
    
num_classes = 2
class_names_list=list()
class_names_list.append('clear')
class_names_list.append('cloud')
class_labels_list=list()
class_labels_list.append(50)
class_labels_list.append(100)

crop_height=args.crop_height
crop_width=args.crop_width
step_height=args.step_height
step_width=args.step_width

crop_height_2=1
crop_width_2=1

mean=np.loadtxt("%s/%s_Samples/mean.txt"%(args.dataset,args.dataset))
sdev=np.loadtxt("%s/%s_Samples/sdev.txt"%(args.dataset,args.dataset))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
else:
    # Create a OneDeviceStrategy
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

with strategy.scope():
    if args.dataset=="L7":
        numchannel=7
    elif args.dataset=="L8":
        numchannel=10

print("Preparing the model ...")

tf.compat.v1.disable_eager_execution()
input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, numchannel])
gt = tf.compat.v1.placeholder(tf.uint8, shape=[None, None, None])

network = None
init_fn = None

indices = gt
output = tf.one_hot(indices, num_classes)
indices0 = tf.cast(indices, tf.float32)
indices1 = tf.expand_dims(indices0, axis=3)

dropout_p=args.dropout
learning_rate=args.learningrate

network, predlabel = build_net_smooth(input, dropout_p, scope=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))

# =============================================================================
# varall = [var for var in tf.compat.v1.trainable_variables()]
# opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995).minimize(loss, var_list=varall)
# # Get the list of all variables in the current model
# var_list = [v for v in tf.compat.v1.global_variables()]
# print("Model Variables:")
# 
# # Filter the variables to exclude those that are not in the checkpoint
# all_vars = tf.compat.v1.global_variables()
# 
# # Exclude the problematic variables by filtering them out from the list
# var_list = [v for v in all_vars if 'Conv_10' not in v.name]
# 
# 
# # Create a saver that only restores the filtered variables
# 
# 
# 
# sess = tf.compat.v1.Session()
# saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=1000)
# sess.run(tf.compat.v1.initializers.global_variables())
# =============================================================================

varall = [var for var in tf.compat.v1.trainable_variables()]
opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995).minimize(loss, var_list=varall)
# Get the list of all variables in the current model
#var_list = [v for v in tf.compat.v1.global_variables()]
#print("Model Variables:")

# Filter the variables to exclude those that are not in the checkpoint
all_vars = tf.compat.v1.global_variables()

# Exclude the problematic variables by filtering them out from the list
#var_list = [v for v in all_vars if 'Conv_10' not in v.name]
#checkpoint_path = "L7/Experiment/checkpoints/latest_model.ckpt"
if args.dataset == "L8":
    checkpoint_path = "L8/Experiment/checkpoints/latest_model.ckpt"
else:
    checkpoint_path = "L7/Experiment/checkpoints/latest_model.ckpt"
if args.dataset == "L8":
    # Filtering logic for L8
    var_list = [v for v in all_vars if 'Conv_10' not in v.name]  # Adjust this filter as needed
else:
    # Filtering logic for L7
    var_list = [v for v in all_vars if 'Conv_10' not in v.name]

#checkpoint_vars = {name for name, _ in tf.train.list_variables(checkpoint_path)}
 # Ensure you have imported os at the top of your script

# Check if the checkpoint exists
if os.path.exists(checkpoint_path + '.index'):
    checkpoint_vars = {name for name, _ in tf.train.list_variables(checkpoint_path)}
else:
    print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    checkpoint_vars = set()


# Filter variables that are present in both the model and the checkpoint
filtered_vars = [v for v in all_vars if v.name.split(':')[0] in checkpoint_vars]
# Directly use all trainable variables
#filtered_vars = tf.compat.v1.global_variables()

# Log the filtered variables to a file for debugging
with open('model_variables_filtered.txt', 'w') as f:
    f.write("Filtered Model Variables for Checkpoint:\n")
    for v in filtered_vars:
        f.write(v.name + '\n')
#print("Filtered model variables have been written to 'model_variables_filtered.txt'")

# Create a saver that only restores the filtered variables



sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.initializers.global_variables())
saver = tf.compat.v1.train.Saver(var_list=filtered_vars)

# Try to restore from checkpoint
# Try to find the latest checkpoint
model_checkpoint_name = tf.train.latest_checkpoint(checkpointpath)

if model_checkpoint_name:
    print(f"Loaded latest model at checkpoint: {model_checkpoint_name}")
    try:
        saver.restore(sess, model_checkpoint_name)
    except tf.errors.NotFoundError as e:
        #print(f"Error loading checkpoint: {e}")
        #print("Continuing with random initialization for missing variables.")
        sess.run(tf.compat.v1.global_variables_initializer())
else:
    print("No checkpoint found, starting from scratch.")
    sess.run(tf.compat.v1.global_variables_initializer())


# Initialize any missing variables
uninitialized_vars = [
    v for v in tf.compat.v1.global_variables() if v not in sess.run(tf.compat.v1.report_uninitialized_variables())
]
sess.run(tf.compat.v1.variables_initializer(uninitialized_vars))
#print("Initialized missing variables randomly.")


# Logic to handle loading from checkpoints based on mode.
from tensorflow.python.tools import inspect_checkpoint as chkp

# Provide the path to your checkpoint file

#print("\nVariables in the checkpoint:")
#chkp.print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True, all_tensor_names=True)

def count_params():
    total_params = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim
        total_params += variable_params
    print(f'Total params: {total_params}')
count_params()

if init_fn is not None:
    init_fn(sess)


if args.mode == "train":

    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Num Epochs -->", args.num_epochs)
    print("")

    mv=[]
    if args.dataset=="L8":

        for j in range(10):
            imgfile=("%s/%s_Samples/band_%d.tif"%(args.dataset,args.dataset,j))
            srcRaster = osgeo.gdal.Open(imgfile)
            img = srcRaster.ReadAsArray()
            img = (img-mean[j])/sdev[j]
            mv.append(img)
        image=cv2.merge(mv)
        image=image[:,0:args.num_samples*11,0:10]
        image_batch=np.expand_dims(image, axis=0)

        labfile=("%s/%s_Samples/label.tif"%(args.dataset,args.dataset))
        labRaster = osgeo.gdal.Open(labfile)
        labe = labRaster.ReadAsArray()
        labe = (labe==50)*0+(labe==100)*0+(labe==150)*1+(labe==200)*1

    elif args.dataset=="L7":
        for j in range(7):
            imgfile=("%s/%s_Samples/band_%d.tif"%(args.dataset,args.dataset,j))
            srcRaster = osgeo.gdal.Open(imgfile)
            img = srcRaster.ReadAsArray()
            img = (img-mean[j])/sdev[j]
            mv.append(img)
        image=cv2.merge(mv)
        image=image[:,0:args.num_samples*11,:]
        image_batch=np.expand_dims(image, axis=0)

        labfile=("%s/%s_Samples/label.tif"%(args.dataset,args.dataset))
        labRaster = osgeo.gdal.Open(labfile)
        labe = labRaster.ReadAsArray()
        labe = (labe==64)*0+(labe==128)*0+(labe==191)*1+(labe==192)*1+(labe==255)*1


    h=image.shape[0]
    w=image.shape[1]
    yrang=range(0, h, 11)
    xrang=range(0, w, 11)

    imgs=[]
    labs=[]
    for y in yrang:
        for x in xrang:
            img=image[y+5-crop_height_2:y+5+crop_height_2+1, x+5-crop_width_2:x+5+crop_width_2+1]
            lab=labe[y+5:y+6, x+5:x+6]
            imgs.append(img)
            labs.append(lab)
    image_batch = np.stack(imgs, axis=0)
    label_batch = np.stack(labs, axis=0)

    avg_loss_per_epoch = []
    avg_scores_per_epoch = []
    avg_scoresnb_per_epoch = []
    
# =============================================================================
#     # Do the training here
#     for epoch in range(0, args.num_epochs):
#         current_losses = []
# 
#     # Training Phase
#         for i in range(200):
#             st = time.time()
#             _, current = sess.run([opt, loss], feed_dict={input: image_batch, gt: label_batch})
#             current_losses.append(current)
# 
#     # Log the epoch information
#     string_print = "Epoch = %d Current = %.2f Time = %.2f" % (epoch, current, time.time() - st)
#     LOG(string_print)
# 
#     mean_loss = np.mean(current_losses)
#     avg_loss_per_epoch.append(mean_loss)
# 
#     # Create directories if needed
#     val_path = "%s/%04d" % (checkpointpath, epoch)
#     if not os.path.isdir(val_path):
#         os.makedirs(val_path)
# 
#     # Save the model checkpoint
#     saver.save(sess, model_checkpoint_name)
# 
#     # Perform validation at the end of each epoch
#     if args.dataset == "L7":
#         totalcount = Test_Scene_l7(val_path, "Val")
#     elif args.dataset == "L8":
#         totalcount = Test_Scene_l8(val_path, "Val")
# 
#     # Calculate the total number of correct predictions
#     corr = 0
#     for j in range(0, num_classes):
#         corr = corr + totalcount[j][j]
# 
#     # Calculate the average score
#     if totalcount[num_classes][num_classes] > 0:
#         avg_score = 1.0 * corr / totalcount[num_classes][num_classes]
#     else:
#         avg_score = 0
# 
#     avg_scores_per_epoch.append(avg_score)
# 
#     print(f"Epoch {epoch}: avg_score = {avg_score:.6f}, mean loss = {mean_loss:.6f}")
# 
#     # Draw the learning curve
#     Drawcurve(args.num_epochs, avg_scores_per_epoch, avg_loss_per_epoch, checkpointpath)
# =============================================================================


#=============================================================================
    sess.run(tf.compat.v1.global_variables_initializer())
    # Do the training here
    all_train_predictions = []
    all_train_labels = []
    for epoch in range(0, args.num_epochs):
        current_losses = []
            
        # Training Phase
        for i in range(200):
            st = time.time()
            _, current = sess.run([opt, loss], feed_dict={input: image_batch, gt: label_batch})
            current_losses.append(current)
            
            pred_label_batch = sess.run(predlabel, feed_dict={input: image_batch})

        # Flatten predictions and labels and ensure they are collected correctly
            all_train_predictions.extend(pred_label_batch.flatten().tolist())
            all_train_labels.extend(label_batch.flatten().tolist())

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)
        
        # Calculate the score using training data if validation is off
        if not args.validate:
            totalcount = np.zeros((num_classes + 1, num_classes + 1))  # Initialize total count

            for i in range(5):  # For example, evaluate on 5 random batches from the training data
                pred_label_batch = sess.run(predlabel, feed_dict={input: image_batch})

                # Debugging: print the predicted and ground truth labels
                #print(f"Predicted labels (batch {i}):", np.unique(pred_label_batch))
                #print(f"Ground truth labels (batch {i}):", np.unique(label_batch))

                # Compare predictions with ground truth and update the total count
                scenecount = compareImage(label_batch, pred_label_batch, class_labels_list)
                totalcount += scenecount

                # Debugging: check the scenecount array
                #print(f"Scenecount for batch {i}:")
                #print(scenecount)

            # Calculate the average score
            corr = np.sum([totalcount[j][j] for j in range(num_classes)])  # Sum of diagonal elements
            if totalcount[num_classes][num_classes] > 0:
                avg_score = corr / totalcount[num_classes][num_classes]
            else:
                avg_score = 0

            # Debugging: check the totalcount and corr values
            #print(f"Totalcount after evaluation:")
            #print(totalcount)
            print(f"Correct predictions (corr): {corr}")
        else:
            avg_score = 0  # Placeholder when validation is enabled

        avg_scores_per_epoch.append(avg_score)

        print(f"Epoch {epoch}: avg_score = {avg_score:.6f}, mean loss = {mean_loss:.6f}")
        saver.save(sess, os.path.join(checkpointpath, "latest_model.ckpt"))
        # Optionally save the model and update the learning curves
    if len(all_train_predictions) > 0 and len(all_train_labels) > 0:  # Ensure there are elements to concatenate
        all_train_predictions = np.array(all_train_predictions)
        all_train_labels = np.array(all_train_labels)
        overall_train_accuracy = compute_accuracy(all_train_predictions, all_train_labels)
        print(f"Overall Training accuracy: {overall_train_accuracy:.6f}")
    else:
        print("No training data was accumulated; cannot compute overall training accuracy.")

    Drawcurve(args.num_epochs, avg_scores_per_epoch, avg_loss_per_epoch, checkpointpath)
#=============================================================================

        
elif args.mode == "test":
    print("\n***** Begin testing *****")
    print("Dataset -->", args.dataset)
    print("")

    model_checkpoint_name = tf.train.latest_checkpoint(checkpointpath)
    if model_checkpoint_name:
       print("Loaded latest model at checkpoint:", model_checkpoint_name)
       saver.restore(sess, model_checkpoint_name)
    else:
       print("No checkpoint found, starting from scratch.")
    test_path=exppath+"/Test"
    if not os.path.isdir("%s"%(test_path)):
        os.makedirs("%s"%(test_path))
    if args.dataset=="L7":
        Test_Scene_l7(test_path,"Test")
    elif args.dataset=="L8":
        Test_Scene_l8(test_path,"Test")


else:
    ValueError("Invalid mode selected.")
