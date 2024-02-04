from flask import Flask, request, render_template, redirect, url_for
import recommendation

# Chargement des modèles et des données
try:
    U_matrix, S_matrix, VT_matrix, user_id_to_index, product_id_to_index, original_matrix, U_train, VT_train, filtered_df = recommendation.load_model_and_mappings()
except FileNotFoundError as e:
    print(e)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
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
    recommendations = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            recommendations = recommendation.provide_recommendations_for_user(user_id)
    return render_template('user_recommendation.html', recommendations=recommendations)







if __name__ == '__main__':
    app.run(debug=True)
