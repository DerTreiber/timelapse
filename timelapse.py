import os
from shutil import copyfile, rmtree
from time import time

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences


def find_reference_image_old(image_paths, prominence=0.4):
    peaks_list = []
    images = [get_distance_transform(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in image_paths]
    scores = []
    for i, (ref_image, path) in enumerate(zip(images, image_paths)):
        print(path)
        # similarity_scores = np.zeros([len(image_paths)])
        similarity_scores = []
        for image in images:
            similarity_scores.append(similarity_score(ref_image, image))
            print('Similarity score of {}: {:.2f}'.format(i,similarity_scores[-1]))
            image_show = np.vstack([ref_image, image])
            image_show = cv2.resize(image_show, (1000, 1000))
            # cv2.imshow('images', image_show)
            # if int(cv2.waitKey(10)) == 112:
            #     cv2.waitKey()
        peaks_list.append(get_peaks(similarity_scores, prominence=prominence))
        scores.append(np.array(similarity_scores))
    

    scores = np.vstack(scores)
    print(np.array([len(peaks) for peaks in peaks_list]))
    # cv2.waitKey()
    most_hits = np.argmax(np.array([len(peaks) for peaks in peaks_list]))

    return image_paths[most_hits]

def gather_files(sourcedir, file_ext='.JPG'):
    """Returns list of files with speficied file extension from source directory.

    Args:
        sourcedir (str): Source directory.
        file_ext (str, optional): File extension. Defaults to '.JPG'.

    Returns:
        [list(str)]: List of files. 
    """
    return sorted([os.path.join(sourcedir, x) for x in os.listdir(sourcedir) if os.path.splitext(x)[1] == file_ext])

# def get_similarity_scores_matrix(images, image_paths):
#     resize_factor = 4
    
#     similarity_scores = np.zeros([len(images), len(images)])
#     for i in range(len(images)):
#         for j in range(len(images)):
#             similarity_scores[i,j] = similarity_score(images[i], images[j])
#             print('Similarity score {} and {}: {:.2f}'.format(image_paths[i],image_paths[j],similarity_scores[i,j]))
#             display_img = np.vstack([images[i],images[j]])
#             display_img = cv2.resize(display_img, (display_img.shape[1]//resize_factor,display_img.shape[0]//resize_factor))
#             # cv2.imshow('images', display_img)
#             # cv2.waitKey(10)
#     return similarity_scores

def get_new_ref_image(scores, image_paths):
    peak_indices = get_peaks(scores)
    print('Last peak at image: ', peak_indices[-1])
    if len(peak_indices) == 0:
        return None, peak_indices
    ref_image = get_distance_transform(cv2.imread(image_paths[peak_indices[-1]], cv2.IMREAD_GRAYSCALE))
    return ref_image, peak_indices

def display_images(image_ref, image, resize_factor):
    display_img = np.vstack([image_ref,image])
    new_resolution = (display_img.shape[1]//resize_factor,display_img.shape[0]//resize_factor)
    display_img = cv2.resize(display_img, new_resolution)
    # cv2.imshow('images', display_img)
    # cv2.waitKey(10)

def get_similarity_scores_batch(image_ref, image_paths):
    # similarity_scores = np.zeros([len(image_paths)])
    similarity_scores = []
    images = [get_distance_transform(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in image_paths]
    for i, image in enumerate(images):
        similarity_scores.append(similarity_score(image_ref, image))
        image_show = np.vstack([image_ref, image])
        image_show = cv2.resize(image_show, (1000, 1000))
        cv2.imshow('images', image_show)
        if int(cv2.waitKey(10)) == 112:
            cv2.waitKey()
        print('Similarity score of {}: {:.2f}'.format(i,similarity_scores[-1]))
    return similarity_scores

def get_similarity_scores(ref_image, image_paths, prominence=0.2, batch_size=40):
    ref_image = get_distance_transform(ref_image)
    
    # similarity_scores = np.zeros([len(image_paths)])
    similarity_scores = []
    cur_peak_indices = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[batch_start:batch_start+batch_size]
        batch_scores = get_similarity_scores_batch(ref_image, batch_image_paths)
        new_peak_indices = get_peaks(np.array(batch_scores))
        
        ### bad ref image was selected, get second to last one as ref image instead
        if len(new_peak_indices) == 0:
            ref_image = get_distance_transform(cv2.imread(image_paths[cur_peak_indices[-2]], cv2.IMREAD_GRAYSCALE))
            print('new ref image', image_paths[cur_peak_indices[-2]])
            batch_scores = get_similarity_scores_batch(ref_image, batch_image_paths)
            new_peak_indices = get_peaks(np.array(batch_scores), prominence=prominence)

        ref_image = get_distance_transform(cv2.imread(batch_image_paths[new_peak_indices[-1]], cv2.IMREAD_GRAYSCALE))
        print('new ref image', batch_image_paths[new_peak_indices[-1]])
        cur_peak_indices = new_peak_indices+batch_start


        similarity_scores += batch_scores

    return np.array(similarity_scores)

def similarity_score(image1, image2):
    ### TODO korrelation von kantenbild
    # return ssim(image1, image2)
    return 1 - (np.count_nonzero(image1-image2))/(image1.shape[0]*image1.shape[1])


def get_peaks(scores, prominence=0.2):
    normalized_scores = (scores - np.min(scores)) * (1/np.max(scores))
    normalized_scores = normalized_scores * (1/np.max(normalized_scores))
    peaks,_ = find_peaks(normalized_scores, prominence=prominence)
    # prominences = peak_prominences(scores, peaks)[0]
    return peaks

def get_peaks_threshold(scores, threshold=0.3):
    normalized_scores = (scores - np.min(scores)) * (1/np.max(scores))
    normalized_scores = normalized_scores * (1/np.max(normalized_scores))
    return normalized_scores > threshold

def get_distance_transform(img):
    ret,th1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,333,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,333,2)

    
    # edges = cv2.Canny(img,100,200)
    # dist = cv2.distanceTransform(edges, cv2.DIST_L2, 3)
    # image_show = np.vstack([edges, dist])
    # image_show = cv2.resize(image_show, (1000, 1000))
    # cv2.imshow('edges + distance transform', image_show)
    # image_threshold = np.vstack([img,th1])
    # image_threshold = cv2.resize(image_threshold, (1000, 1000))
    # cv2.imshow('threshold global', image_threshold)
    # image_threshold_adaptive = np.vstack([img,th2])
    # image_threshold_adaptive = cv2.resize(image_threshold_adaptive, (1000, 1000))
    # cv2.imshow('threshold adaptive mean', image_threshold_adaptive)
    # image_threshold_adaptive = np.vstack([img,th3])
    # image_threshold_adaptive = cv2.resize(image_threshold_adaptive, (1000, 1000))
    # cv2.imshow('threshold adaptive gaussian', image_threshold_adaptive)

    # cv2.waitKey()
    return th3

def write_video(paths, out_path, codec=cv2.VideoWriter_fourcc(*'MJPG'), fps=20.0):
    print('Writing output to video.')
    img = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    
    resolution = (img.shape[1], img.shape[0])

    vid_writer = cv2.VideoWriter(out_path, codec, fps, resolution)
    for path in paths:
        vid_writer.write(cv2.imread(path,cv2.IMREAD_COLOR))

    vid_writer.release()        

def sort_images(paths, path_reference, prominence=0.2, batch_size=40):

    return similarity_scores

def plot_peaks(scores, peaks):
    plt.plot(np.arange(len(scores)), scores)
    plt.scatter(peaks, scores[peaks], marker='x', color='red')
    plt.show()
    plt.savefig('peaks.png')

### TODO
### find reference
### find image with the most similar image

def find_reference_image(image_paths, prominence=0.4):
    peaks_list = []
    images = [get_distance_transform(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in image_paths]
    scores = []
    for i, (ref_image, path) in enumerate(zip(images, image_paths)):
        print(path)
        # similarity_scores = np.zeros([len(image_paths)])
        similarity_scores = []
        for image in images:
            similarity_scores.append(similarity_score(ref_image, image))
            print('Similarity score of {}: {:.2f}'.format(i,similarity_scores[-1]))
            image_show = np.vstack([ref_image, image])
            image_show = cv2.resize(image_show, (1000, 1000))
            # cv2.imshow('images', image_show)
            # if int(cv2.waitKey(10)) == 112:
            #     cv2.waitKey()
        scores.append(np.array(similarity_scores))

    peaks_list = []
    for prominence in np.arange(start=1, stop=0, step=-0.1):
        for score in scores:
            peaks_list.append(get_peaks(score, prominence=prominence))
        
        len_peaks = np.array([len(peaks) for peaks in peaks_list])
        most_hits = np.argmax(len_peaks)
        if len_peaks[most_hits] >= 3:
            break
    

    scores = np.vstack(scores)
    print(np.array([len(peaks) for peaks in peaks_list]))
    # cv2.waitKey()
    most_hits = np.argmax(np.array([len(peaks) for peaks in peaks_list]))

    return paths[most_hits]

def test_features(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    plt.imshow(img),plt.show()


if __name__ == "__main__":
    """
    Adjustable parameters:
        - batch_size
        - prominence_search
        - prominence_sort
    """
    # test_features('/home/treiber/andras/timelapsedata/126_PANA/P1260179.JPG')
    batch_size = 100
    prominence_search = 0.7
    prominence_sort = 0.2

    # sourcedir = '/home/treiber/andras/raw'
    # targetdir = '/home/treiber/andras/sorted'


    # sourcedir = '/media/treiber/5EC43479C434560D/Users/JulianTreiber/andras/102_PANA'
    # targetdir = '/media/treiber/5EC43479C434560D/Users/JulianTreiber/andras/102_PANA_sorted'
    
    sourcedir = '/home/treiber/andras/timelapsedata/126_PANA'
    targetdir = '/home/treiber/andras/sorted'


    if not os.path.isdir(targetdir):
        os.makedirs(targetdir)

    start_time = time()
    paths = gather_files(sourcedir)[100:]
    start_time_find_ref_image = time()
    # path_reference = find_reference_image(paths[:batch_size], prominence=prominence_search)
    path_reference = find_reference_image_old(paths[:batch_size], prominence=prominence_search)
    # path_reference = '/home/treiber/andras/TIMELAPSEDINO/P1240263.JPG'
    cv2.imshow('reference image', cv2.resize(cv2.imread(path_reference), (1000,500)))
    cv2.waitKey()

    # path_reference = '/media/treiber/5EC43479C434560D/Users/JulianTreiber/andras/102_PANA/P1020041.JPG'
    total_time_find_ref_image = time() - start_time_find_ref_image
    
    # print(path_reference)
    # path_reference = paths[9]

    start_time_similarity_scores = time()
    ### read images
    ref_image = cv2.imread(path_reference, cv2.IMREAD_GRAYSCALE)
    
    ### get similarity scores
    similarity_scores = get_similarity_scores(ref_image, paths, prominence=prominence_sort, batch_size=batch_size)

    ### get peaks, which are matching images
    peak_indices = get_peaks(similarity_scores)
    plot_peaks(similarity_scores, peak_indices)
    end_time_similarity_scores = time() - start_time_similarity_scores


    start_time_write_video = time()
    print('Copying images to {}.'.format(targetdir))
    rmtree(targetdir)
    os.mkdir(targetdir)
    for i in peak_indices:
        copyfile(paths[i], os.path.join(targetdir, os.path.split(paths[i])[-1]))

    write_video(sorted([os.path.join(targetdir, x) for x in os.listdir(targetdir)]), 'res.mp4', codec=cv2.VideoWriter_fourcc(*'DIVX'))
    end_time_write_video = time() - start_time_write_video

    total_time = time() - start_time 

    print('Total time: {:.2f} s'.format(total_time))
    print('Number of images: {}'.format(len(paths)))
    print('Time for finding reference image: {:.2f} s'.format(total_time_find_ref_image))
    print('Time for finding similarity scores: {:.2f} s'.format(end_time_similarity_scores))
    print('Time for writing video: {:.2f} s'.format(end_time_write_video))

