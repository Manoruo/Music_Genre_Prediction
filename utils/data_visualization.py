import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
import pandas as pd 
import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_confusion_matrix(y_true, y_pred, genre_labels):
    """
        Plots a confusion matrix heatmap and displays the classification report.

        Parameters:
        - y_true (numpy array): Ground truth labels (int array).
        - y_pred (numpy array): Predicted labels (int array).
        - genre_labels (dict): Mapping of numeric labels to genre names.

        Returns:
        None
    """
    
    # Get unique labels
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap using seaborn
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[genre_labels[label] for label in unique_labels],
                yticklabels=[genre_labels[label] for label in unique_labels][::-1])  # Reverse the order for y-axis

    # Add labels, title, and color bar
    plt.xlabel('True Genre')
    plt.ylabel('Predicted Genre')
    plt.title('Confusion Matrix')

    plt.show()

    # Display the classification report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=[genre_labels[label] for label in unique_labels]))

def plot_multilabel_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a multi-label confusion matrix heatmap.

    Parameters:
    - y_true (numpy array): Ground truth labels (binary array).
    - y_pred (numpy array): Predicted labels (binary array).
    - labels (list): List of label names.

    Returns:
    None
    """
    
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)


    cm = multilabel_confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    total = round(len(labels.keys()) / 3) + 1
    for i, label in labels.items():
        plt.subplot(total, 3, i + 1)
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        plt.title(f'Confusion Matrix for Label: {label}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def visualize_data(genre_labels, multilabels = True, pal="rocket"):
    """
    Visualize the frequency distribution of genres in a given DataFrame.
    
    This function helps detect outliers in the genre distribution by providing a descriptive summary and creating a bar plot
    showing the frequency of each genre. The palette choice can be customized for better visualization.
    
    Parameters:
    - genre_labels (pd.Series): The Series containing lists of genres for each entry.
    - mutlilabels (bool): Tells us if there is a list of possible labels (genres) or a single label  
    - pal (str): Palette to be used for the seaborn count plot. Default is "tab10".

    Returns:
    None
    """
    
    # each row in the genre_labels is a list of the song's genres. We need to unpack it so we can get counts for each genre
    if multilabels:
        genres = [genre for row in genre_labels for genre in row]
    else:
        genres = genre_labels
        
    genres_series = pd.Series(genres) # this series object will have ALL the genres (out of the lists)
  
    print(genres_series.describe())
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create a bar plot for genre frequencies
    plt.figure(figsize=(12, 10))
    # make seaborn count plot 
    ax = sns.countplot(y=genres_series, order=genres_series.value_counts().index, palette=pal)
    # add counts to count plot 
    for label in ax.containers:
        ax.bar_label(label)
    
    plt.title("Genre Frequency", fontsize=14)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Genre", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()