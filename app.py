from flask import Flask, request, render_template, redirect, url_for
import recommendation
import pandas as pd
import os

# Chargement des modèles et des données
try:
    U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = recommendation.load_model_and_mappings()
except FileNotFoundError as e:
    print(e)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    global filtered_df
    user_info = None
    book_search_results = None  # Ajout pour la recherche de livres

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        search_title = request.form.get('search_title')  # Récupérer le titre recherché

        if user_id:
            user_info = recommendation.display_user_info(user_id, filtered_df)
            if user_info.empty:
                user_info = None

        if search_title:  # Si un titre est recherché
            book_search_results = recommendation.search_books_by_title(search_title, filtered_df)

    return render_template('index.html', user_info=user_info, book_search_results=book_search_results)


@app.route('/view_all')
def view_all():
    global filtered_df
    short_df = filtered_df.head(100)
    return render_template('view_all.html', table=short_df.to_html(classes='data', header="true"))

@app.route('/user_recommendation', methods=['GET', 'POST'])
def user_recommendation():
    global filtered_df
    recommendations = None
    user_not_found = False  # Flag pour indiquer si l'utilisateur n'est pas trouvé

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            recommendations = recommendation.provide_recommendations_for_user(user_id, filtered_df)
            # recommendations = recommendations.sort_values('Predict_Score', ascending=False)
            if recommendations is None:
                user_not_found = True
    return render_template('user_recommendation.html', recommendations=recommendations, user_not_found=user_not_found)


# app.py
@app.route('/add_book', methods=['GET', 'POST'])
def add_book():
    global filtered_df  # Ajouter cette ligne pour indiquer que filtered_df est global
    if request.method == 'POST':
        # Récupérer les données du formulaire
        product_id = request.form.get('ProductId')
        user_id = request.form.get('UserId')
        title = request.form.get('title')
        score = request.form.get('Score')
        time = request.form.get('Time')
        authors = request.form.get('authors')
        categories = request.form.get('categories')

        # Validation pour s'assurer que le score est un nombre valide
        try:
            score = float(score)  # Tentative de conversion en float
        except ValueError:
            # Gérer l'erreur si la conversion échoue
            # Par exemple, renvoyer un message d'erreur à l'utilisateur
            return render_template('add_book.html', error="Invalid score. Please enter a numeric value.")

        # Créer une nouvelle ligne sous forme de dictionnaire
        new_row = {'ProductId': product_id, 'UserId': user_id, 'title': title, 
                   'Score': score, 'Time': time, 'authors': authors, 'categories': categories}
        
        # Ajouter la nouvelle ligne au DataFrame
        filtered_df = recommendation.add_book(new_row, filtered_df)

        # Rediriger vers la page d'accueil ou une autre page si souhaité
        return redirect(url_for('index'))

    return render_template('add_book.html')



@app.route('/reload_model')
def reload_model():
    global filtered_df
    global U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train
    update_monitoring_stats(filtered_df)
    # Appeler la fonction pour reconstruire le système de recommandation
    recommendation.book_recommendation_system(filtered_df)

    # Recharger les modèles et les mappings
    try:
        U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = recommendation.load_model_and_mappings()
    except FileNotFoundError as e:
        print(e)

    # Rediriger vers la page d'accueil ou une autre page appropriée
    return redirect(url_for('index'))

def update_monitoring_stats(filtered_df):
    file_path = 'Dataset/Monitoring.pkl'
    
    # Calculer les statistiques
    df_stats = {
        'Timestamp': pd.Timestamp.now(),
        'Nombre_de_lignes': len(filtered_df),
        'Nombre_de_produits_uniques': filtered_df['ProductId'].nunique(),
        'Nombre_d_utilisateurs_uniques': filtered_df['UserId'].nunique(),
    }
    new_data = pd.DataFrame([df_stats])

    # Charger ou créer le DataFrame de monitoring
    if os.path.exists(file_path):
        monitoring_data = pd.read_pickle(file_path)
        monitoring_data = pd.concat([monitoring_data, new_data], ignore_index=True)
    else:
        monitoring_data = new_data

    # Sauvegarder les données
    monitoring_data.to_pickle(file_path)




@app.route('/monitoring')
def monitoring():
    # Charger le modèle de résultats existant depuis le fichier pickle
    model_results_df = pd.read_pickle('Dataset/resultats.pkl')

    # Charger le DataFrame de suivi depuis le fichier pickle
    monitoring_df = pd.read_pickle('Dataset/Monitoring.pkl')
    
    # Tri par Timestamp si nécessaire
    monitoring_df = monitoring_df.sort_values('Timestamp', ascending=False)

    # Calculer les statistiques par catégorie si nécessaire
    category_stats = filtered_df.groupby('categories')['Score'].agg(['count', 'mean']).reset_index()
    category_stats.rename(columns={'count': 'Nombre_de_Livres', 'mean': 'Note_Moyenne'}, inplace=True)

    # Passer les données au template
    return render_template(
        'monitoring.html', 
        model_results=model_results_df, 
        monitoring_df=monitoring_df, 
        category_stats=category_stats
    )




if __name__ == '__main__':
    app.run(debug=True)
