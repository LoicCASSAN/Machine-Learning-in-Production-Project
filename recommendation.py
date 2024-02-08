import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import ast
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process, fuzz
from IPython.display import clear_output
import pickle
# from data_db import user, mdp
import time
import os


def get_latest_file(path_pattern):
    dir_name, file_pattern = os.path.split(path_pattern)
    base, ext = os.path.splitext(file_pattern)
    base = base.replace('*', '')  # Supprimez l'astérisque pour la correspondance
    files = [f for f in os.listdir(dir_name) if f.endswith(base + ext) and '_' in f]
    if not files:  # S'il n'y a pas de fichier correspondant
        raise FileNotFoundError(f"No file found for the pattern {path_pattern}")
    # Trier les fichiers en fonction du numéro avant le tiret bas, s'il existe, sinon 0
    latest_file = max(files, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
    return os.path.join(dir_name, latest_file)

def load_model_and_mappings():
    U_matrix_path = get_latest_file('Model/*U_matrix.pkl')
    S_matrix_path = get_latest_file('Model/*S_matrix.pkl')
    VT_matrix_path = get_latest_file('Model/*VT_matrix.pkl')
    user_id_to_index_path = get_latest_file('Model/*user_id_to_index.pkl')
    product_id_to_index_path = get_latest_file('Model/*product_id_to_index.pkl')
    original_matrix_path = get_latest_file('Model/*original_matrix.pkl')
    U_train_path = get_latest_file('Model/*U_train.pkl')
    VT_train_path = get_latest_file('Model/*VT_train.pkl')
    Book_Dataset_path = get_latest_file('Model/*Book_Dataset.pkl')

    with open(U_matrix_path, 'rb') as f:
        U_matrix = pickle.load(f)
    with open(S_matrix_path, 'rb') as f:
        S_matrix = pickle.load(f)
    with open(VT_matrix_path, 'rb') as f:
        VT_matrix = pickle.load(f)
    with open(user_id_to_index_path, 'rb') as f:
        user_id_to_index = pickle.load(f)
    with open(product_id_to_index_path, 'rb') as f:
        product_id_to_index = pickle.load(f)
    with open(original_matrix_path, 'rb') as f:
        original_matrix = pickle.load(f)
    with open(U_train_path, 'rb') as f:
        U_train = pickle.load(f)
    with open(VT_train_path, 'rb') as f:
        VT_train = pickle.load(f)
    filtered_df = pd.read_pickle(Book_Dataset_path)

    return U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df

# Utilisation de la fonction
try:
    U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = load_model_and_mappings()
except FileNotFoundError as e:
    print(e)



def book_recommendation_system(filtered_df):
    print("Training recommendation system start...")
    start_time = time.time()
    
    # print("Using small Model Version")
    # filtered_df = filtered_df.head(1000)

    from fuzzywuzzy import process

    def find_closest_title(title, titles_list):
        closest_title, score = process.extractOne(title, titles_list)
        return closest_title if score > 90 else None  # Vous pouvez ajuster le seuil de score

    # Parcourir filtered_df pour trouver et associer les ProductId manquants
    for index, row in filtered_df.iterrows():
        if pd.isnull(row['ProductId']) or row['ProductId'] == '':
            closest_title = find_closest_title(row['title'], product_ids_by_title.index)
            if closest_title:
                filtered_df.at[index, 'ProductId'] = product_ids_by_title[closest_title]

    filtered_df = filtered_df.sample(frac=1, random_state=42)
    # Get unique UserIds and ProductIds
    unique_user_ids = filtered_df['UserId'].unique()
    unique_product_ids = filtered_df['ProductId'].unique() #unique ids for books are less

    user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
    product_id_to_index = {product_id: index for index, product_id in enumerate(unique_product_ids)}

    # clean matrix
    matrix = np.zeros((len(unique_user_ids), len(unique_product_ids)))

    # users as rows, books as columns with their ratings
    for _, row in filtered_df.iterrows():
        user_id = row['UserId']
        product_id = row['ProductId']
        score = row['Score']
        
        user_index = user_id_to_index[user_id]
        product_index = product_id_to_index[product_id]
        
        if matrix[user_index][product_index] < score:
            matrix[user_index][product_index] = score
    print(matrix.shape)
    matrix
    # Z-Scoring
    matrix = stats.zscore(matrix, axis=0)
    # Evaluation
    def calculate_mse(predicted_matrix, test_matrix):
        num_users = min(predicted_matrix.shape[0], test_matrix.shape[0])
        num_items = min(predicted_matrix.shape[1], test_matrix.shape[1])
        mse = np.mean((predicted_matrix[:num_users, :num_items] - test_matrix[:num_users, :num_items]) ** 2)
        return mse

    def calculate_f1_score(recall, precision):
        if recall + precision == 0:
            return 0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def precision_at_k(actual_matrix, predicted_matrix, k, threshold):
        binary_predicted_matrix = predicted_matrix >= threshold
        
        precision = []
        for i in range(len(actual_matrix)):
            actual_indices = np.where(actual_matrix[i] >= threshold)[0]
            predicted_indices = np.argsort(~binary_predicted_matrix[i])[:k]
            common_indices = np.intersect1d(actual_indices, predicted_indices)
            precision.append(len(common_indices) / len(predicted_indices))
        
        return np.mean(precision)

    def recall_at_k(true_matrix, pred_matrix, k, threshold):
        pred_matrix_sorted = np.argsort(pred_matrix, axis=1)[:, ::-1][:, :k]
        recall_scores = []
        for i in range(len(true_matrix)):
            true_positives = len(set(pred_matrix_sorted[i]).intersection(set(np.where(true_matrix[i] >= threshold)[0])))
            actual_positives = len(np.where(true_matrix[i] >= threshold)[0])
            if actual_positives > 0:
                recall_scores.append(true_positives / actual_positives)
        recall = np.mean(recall_scores)
        return recall
    # SVD
    def split_train_test(matrix, test_size=0.01, random_state=42):
        train_matrix, test_matrix = train_test_split(matrix, test_size=test_size, random_state=random_state)
        return train_matrix, test_matrix

    def calculate_svd(train_matrix, k=100): #k=600 and k=100 for the low version
        train_sparse = csr_matrix(train_matrix)
        # Perform SVD on the sparse matrix
        U_train, S_train, VT_train = svds(train_sparse, k=k)
        # Reverse the singular values, columns of U_train, and rows of VT_train
        S_train_k = np.diag(S_train[::-1])
        U_train_k = U_train[:, ::-1]
        VT_train_k = VT_train[::-1, :]
        
        return U_train_k, S_train_k, VT_train_k

    train_matrix, test_matrix = split_train_test(matrix)

    # training set
    U_train, S_train, VT_train = calculate_svd(train_matrix)
    U_train_pred = np.dot(train_matrix, VT_train.T)
    train_pred_matrix = np.dot(U_train_pred, VT_train)

    # Make predictions for the test set
    U_test_pred = np.dot(test_matrix, VT_train.T)
    predicted_matrix = np.dot(U_test_pred, VT_train)

    # Calculate MSE 
    train_mse = calculate_mse(train_matrix, train_pred_matrix)
    test_mse = calculate_mse(test_matrix, predicted_matrix)

    print("Train Set Mean Squared Error (MSE):", train_mse)
    print("Test Set Mean Squared Error (MSE):", test_mse)
    # Calculate Precision at k for the test set
    precision = precision_at_k(test_matrix, predicted_matrix, k=10, threshold=3)

    # Calculate Recall at k for the test set
    recall = recall_at_k(test_matrix, predicted_matrix, k=10, threshold=3)

    # Calculate F1 score
    f1_score = calculate_f1_score(recall, precision)
    print("RMSE (training): ", np.sqrt(train_mse) )
    print("RMSE (test): ", np.sqrt(test_mse))
    print("Precision @ 10: ", precision)
    print("Recall @ 10:", recall)
    print("F1 Score:", f1_score)
    
    try:
        df = pd.read_pickle('Dataset/resultats.pkl')
    except FileNotFoundError:
        # Si le fichier pickle n'existe pas, créez un nouveau dataframe
        df = pd.DataFrame()

    # Ajouter les nouvelles informations
    new_data = {
        'RMSE (training)': [np.sqrt(train_mse)],
        'RMSE (test)': [np.sqrt(test_mse)],
        'Precision @ 10': [precision],
        'Recall @ 10': [recall],
        'F1 Score': [f1_score]
    }

    new_df = pd.DataFrame(new_data)

    # Concaténer le nouveau dataframe avec l'ancien (s'il existe)
    df = pd.concat([df, new_df], ignore_index=True)

    # Sauvegarder le dataframe mis à jour dans le fichier pickle
    df.to_pickle('Dataset/resultats.pkl')
    
    
    # Save Model
    # Enregistrement des matrices U, S, et VT
    def save_with_unique_name(path, data):
        dir_name, file_name = os.path.split(path)
        base, ext = os.path.splitext(file_name)
        counter = 1
        new_path = os.path.join(dir_name, f"{base}{ext}")  # Définir le chemin sans préfixe de compteur pour le premier essai
        while os.path.exists(new_path):  # Vérifier si le fichier existe sans préfixe de compteur
            new_path = os.path.join(dir_name, f"{counter}_{base}{ext}")  # Ajouter un préfixe de compteur si nécessaire
            counter += 1
        with open(new_path, 'wb') as f:
            pickle.dump(data, f)
        return new_path  # Retourner le nouveau chemin pour confirmer où le fichier a été sauvegardé

    # Utilisation de la fonction
    save_with_unique_name('Model/U_matrix.pkl', U_train)
    save_with_unique_name('Model/S_matrix.pkl', S_train)
    save_with_unique_name('Model/VT_matrix.pkl', VT_train)
    save_with_unique_name('Model/user_id_to_index.pkl', user_id_to_index)
    save_with_unique_name('Model/product_id_to_index.pkl', product_id_to_index)
    save_with_unique_name('Model/original_matrix.pkl', matrix)
    save_with_unique_name('Model/U_train.pkl', U_train)
    save_with_unique_name('Model/VT_train.pkl', VT_train)
    save_with_unique_name('Model/Book_Dataset.pkl', filtered_df)
    
    print("Recommendation system trained in %s seconds." % (time.time() - start_time))



# book_recommendation_system(filtered_df)


## TEST SUR UTILISATEUR DEJA PRÉSENT
def fetch_relevant_items_for_user(user_id, filtered_df, relevant_items=5):
    # Get the index of the user
    U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, not_used = load_model_and_mappings()
    user_index = user_id_to_index[user_id]
    user_embedding = U_train[user_index, :]
    
    similarity_scores = VT_train.T.dot(user_embedding)

    sorted_indices = similarity_scores.argsort()[::-1]
    top_relevant_indices = sorted_indices[:relevant_items]
    
    relevant_items = [list(product_id_to_index.keys())[list(product_id_to_index.values()).index(idx)] for idx in top_relevant_indices]
    relevant_titles = filtered_df.loc[filtered_df['ProductId'].isin(relevant_items), 'title'].tolist()
    
    # Remove any duplicate titles
    unique_relevant_titles = list(set(relevant_titles))
    
    # Get the final set of relevant items without duplicate titles
    final_relevant_items = []
    for title in unique_relevant_titles:
        final_relevant_items.append(title)
    
    return final_relevant_items

def provide_recommendations_for_user(user_id, filtered_df, top_n=35):
    if user_id not in filtered_df['UserId'].values:
        # Si l'utilisateur n'existe pas, retourner un message ou un DataFrame vide
        return pd.DataFrame()  # DataFrame vide
    relevant_items = fetch_relevant_items_for_user(user_id, filtered_df, top_n)
    relevant_items_df = filtered_df[filtered_df['title'].isin(relevant_items)]
    relevant_items_df = relevant_items_df.drop('UserId', axis=1)
    
    relevant_items_df = relevant_items_df.rename(columns={'Score': 'Predict_Score'})
    relevant_items_df = relevant_items_df.sort_values(by='Predict_Score', ascending=False)

    # Grouper par titre et prendre la première occurrence de chaque titre
    relevant_items_df = relevant_items_df.groupby('title').first().reset_index()

    return relevant_items_df



def display_user_info(user_id, df):
    user_info = df.loc[df['UserId'] == user_id]
    return user_info

def search_books_by_title(word, df):
    # Filtrer les livres dont le titre contient le mot spécifié
    filtered_books = df[df['title'].str.contains(word, case=False, na=False)]

    # Supprimer les doublons basés sur 'ProductId'
    unique_books = filtered_books.drop_duplicates(subset='ProductId')

    return unique_books

def add_book(row, df):
    new_df = pd.DataFrame(row, index=[0])
    return pd.concat([df, new_df], ignore_index=True)