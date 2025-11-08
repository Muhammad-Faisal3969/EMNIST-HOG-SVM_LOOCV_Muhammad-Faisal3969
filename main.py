# MUHAMMAD FAISAL
# 4212301084
# MIDTERM

# INITILIZATIONS
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# CONFIGURATIONS
from skimage.feature import hog
from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

# PROGRESS BAR
from tqdm import tqdm 

# IGNORE COMPUTATIONAL WARNINGS
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# GLOBAL VARIABLES
SAMPLES_PER_CLASS = 500
TOTAL_SAMPLES = 26 * SAMPLES_PER_CLASS
IMAGE_SIZE = 28

# FILE PATH
FILE_PATH = r"C:\Users\Asus-chan\assignments\EMNIST-HOG-SVM_LOOCV_Muhammad-Faisal3969\archive\emnist-letters-train.csv"

# OUTPUT PATH
OUTPUT_DIR = 'C:\Users\Asus-chan\assignments\EMNIST-HOG-SVM_LOOCV_Muhammad-Faisal3969\output'
LOG_FILE = os.path.join(OUTPUT_DIR, 'evaluation_log.txt')

# HOG/SVM PARAMETER
HOG_PARAMS_FINAL = {'orientations': 9, 'ppc': (8, 8), 'cpb': (2, 2)} 
SVM_PARAMS_FINAL = {'C': 10.0, 'kernel': 'linear'} 

# MODE
RUN_TUNING_MODE = False  
RUN_LOOCV_FINAL = True   

# LOGGING AND VISUALIZATIONS
def write_log(message):
    """Writing to console and log file..."""
    print(message)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')

def plot_confusion_matrix(y_true, y_pred, kernel_name):
    """Making and saving confusion matrix plot..."""
    labels_az = [chr(ord('A') + i) for i in range(26)] 
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_az, yticklabels=labels_az,
                cbar=True, cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    title = f'Confusion Matrix (LOOCV + HOG + SVM {kernel_name.capitalize()})'
    plt.title(title, fontsize=16)
    plt.tight_layout() 
    
    filename = f'confusion_matrix_hog_svm_{kernel_name}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    write_log(f"\n[VISUALIZATION] Confusion Matrix saved to: {filepath}")
    plt.close()

# LOADING DATA AND SAMPLING DATA
def load_and_sample_data(file_path):
    """Loading, normalizing and sampling data..."""
    write_log("Loading and sampling data...")
    df = pd.read_csv(file_path, header=None)
    X_full = df.iloc[:, 1:].values.astype('float32') / 255.0 
    y_full = df.iloc[:, 0].values - 1 
    
    X_sampled_list = []
    y_sampled_list = []
    
    for class_label in range(26):
        class_indices = np.where(y_full == class_label)[0]
        if len(class_indices) >= SAMPLES_PER_CLASS:
            selected_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
            X_sampled_list.append(X_full[selected_indices])
            y_sampled_list.append(y_full[selected_indices])
        else:
            X_sampled_list.append(X_full[class_indices])
            y_sampled_list.append(y_full[class_indices])

    X_sampled = np.concatenate(X_sampled_list, axis=0)
    y_sampled = np.concatenate(y_sampled_list, axis=0)
    write_log(f"Total sample: {len(X_sampled)} ({len(np.unique(y_sampled))} class)")
    return X_sampled, y_sampled

# HOG FEATURES PROGRESS BAR
def extract_hog_features(images, orientations, ppc, cpb):
    """Extracting HOG features from image array..."""
    hog_features = []
    write_log(f"\n[HOG] Extraction features: Orient={orientations}, PPC={ppc}, CPB={cpb}...")
    
    for image in tqdm(images, desc="HOG extraction"):
        image_2d = image.reshape(IMAGE_SIZE, IMAGE_SIZE)
        features = hog(image_2d, 
                       orientations=orientations, 
                       pixels_per_cell=ppc,
                       cells_per_block=cpb, 
                       transform_sqrt=True,
                       feature_vector=True)
        hog_features.append(features)
        
    X_features = np.array(hog_features)
    write_log(f"=HOG features extraction: {X_features.shape}")
    return X_features

# PARAMETER TUNING
def tune_parameters(X_features, y_labels, param_grid):
    """Tuning SVM parameter using stratified K-Fold CV..."""
    write_log("\n[TUNING] Start the tuning with Stratified K-Fold CV (k=5)...")
    
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_base = SVC(gamma='scale', max_iter=20000, random_state=42)

    grid_search = GridSearchCV(
        model_base, 
        param_grid, 
        cv=cv_kfold, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_features, y_labels)
    end_time = time.time()
    
    write_log("\n!Tuning result (K-Fold CV)!")
    write_log(f"Tuning took: {(end_time - start_time):.2f} seconds!")
    write_log(f"Best parameter: {grid_search.best_params_}")
    write_log(f"Best K-Fold parameter: {grid_search.best_score_ * 100:.2f}%")
    
    return grid_search.best_params_

# FINAL EVALUATION (LOOCV)
def evaluate_loocv(X_features, y_labels, C_param, kernel_param):
    """Final evaluation using Leave-One-Out Cross-Validation..."""
    write_log(f"\n[LOOCV] LOOCV finalizing (Kernel={kernel_param}, C={C_param})...")
    
    model_svm = SVC(C=C_param, kernel=kernel_param, gamma='scale', random_state=42, max_iter=30000) 
    loocv = LeaveOneOut()
    
    start_time = time.time()
    write_log(f"WARNING: LOOCV on {len(X_features)} will take a long time to sample (hours/days)!")
    
    y_pred = cross_val_predict(model_svm, X_features, y_labels, cv=loocv, n_jobs=-1, verbose=1)
    
    end_time = time.time()
    
    accuracy = accuracy_score(y_labels, y_pred)
    
    write_log("\nLOOCV final matrix evaluation")
    write_log(f"total LOOCV Computational time: {(end_time - start_time) / 3600:.2f} hours")
    write_log(f"LOOCV accuracy: {accuracy * 100:.2f}%")

    plot_confusion_matrix(y_labels, y_pred, kernel_param)

    return accuracy


# MAIN FUNCTION

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    write_log("=======================================================")
    write_log(f"EMNIST classification start at {time.ctime()}")
    write_log(f"Output Log: {LOG_FILE}")
    write_log("=======================================================")
    
    if not os.path.exists(FILE_PATH):
        write_log(f"ERROR: File {FILE_PATH} file not found!")
    else:
        X, y = load_and_sample_data(FILE_PATH)
        
        # PARAMETER TUNING
        if RUN_TUNING_MODE:
            pass 
        
        # FINAL LOOCV EVALUATION
        if RUN_LOOCV_FINAL:
            write_log("\n!Final LOOCV evaluation!")
            
            X_features_final = extract_hog_features(X, **HOG_PARAMS_FINAL)
            
            final_accuracy = evaluate_loocv(X_features_final, y, 
                                            SVM_PARAMS_FINAL['C'], 
                                            SVM_PARAMS_FINAL['kernel']) 
            
            write_log("\n!Proram execution is done!")