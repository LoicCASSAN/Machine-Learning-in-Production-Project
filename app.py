from flask import Flask, request, render_template, redirect, url_for
import recommendation
import pandas as pd

# Chargement des modèles et des données
try:
    U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = recommendation.load_model_and_mappings()
except FileNotFoundError as e:
    print(e)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    global filtered_df
    user_info = None  # Initialiser user_info à None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            user_info = recommendation.display_user_info(user_id, filtered_df)
            if user_info.empty:  # Vérifier si le résultat est vide
                user_info = None  # Réinitialiser user_info si aucune info n'est trouvée
    return render_template('index.html', user_info=user_info)


@app.route('/user_recommendation', methods=['GET', 'POST'])
def user_recommendation():
    global filtered_df
    recommendations = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            recommendations = recommendation.provide_recommendations_for_user(user_id, filtered_df, top_n=35)
            # print(recommendations)  # Pour déboguer
    return render_template('user_recommendation.html', recommendations=recommendations)


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

    # Appeler la fonction pour reconstruire le système de recommandation
    recommendation.book_recommendation_system(filtered_df)

    # Recharger les modèles et les mappings
    try:
        U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = recommendation.load_model_and_mappings()
    except FileNotFoundError as e:
        print(e)

    # Rediriger vers la page d'accueil ou une autre page appropriée
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
