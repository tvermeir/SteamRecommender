import numpy as np
import ast  import time
from collections import defaultdict
import json
import dask.array as da
import dask.bag as db
from dask import delayed, compute



# Helper function for Dask Bag processing in build_user_game_matrix
def safe_literal_eval(line_str):
    try:
        return ast.literal_eval(line_str.strip())
    except (ValueError, SyntaxError):

        return None

# Helper function for Dask Bag processing in build_user_game_matrix
def extract_interactions(user_data_dict):
    user_id = user_data_dict.get('user_id')
    items = user_data_dict.get('items', [])
    if not user_id:
        return []

    interactions = []
    for item in items:
        game_id = item.get('item_id')
        playtime = item.get('playtime_forever', 0)
        if game_id:
            log_playtime = np.log1p(playtime)
            interactions.append({'user_id': user_id, 'game_id': str(game_id), 'playtime': log_playtime})
    return interactions



def compute_cosine_similarity(X_da): # X_da is a Dask array
    """
    Compute cosine similarity between all pairs of vectors in X_da.
    X_da should be a 2D Dask array where each row is a vector.
    """
    print("LOG: Starting cosine similarity computation (Dask)...")
    # Calculate norms using Dask array functions
    norms_da = da.linalg.norm(X_da, axis=1, keepdims=True)
    norms_da = da.maximum(norms_da, 1e-12)

    # Normalize the vectors
    X_norm_da = X_da / norms_da

    # Compute cosine similarity
    result = da.dot(X_norm_da, X_norm_da.T)
    print("LOG: Finished cosine similarity computation (Dask).")
    return result


def compute_item_similarities(matrix_da, game_ids, output_file, top_n=10): # Takes Dask array
    """
    Compute cosine similarity between games and save the results.

    Args:
        matrix_da: User-game Dask array (users x games)
        game_ids: List of game IDs corresponding to matrix columns
        output_file: Base filename for saving results
        top_n: Number of most similar games to save for each game
    """
    overall_start_time = time.time()

   
    start_time = time.time()


    similarities_da = compute_cosine_similarity(matrix_da.T) 
    print(f"LOG: Dask cosine similarity graph built in {time.time() - start_time:.2f} seconds.")

    print("LOG: Computing full similarity Dask array to NumPy array...")
    start_time = time.time()
    similarities_np = similarities_da.compute() # Convert to NumPy array
    print(f"LOG: Full similarity matrix computed to NumPy in {time.time() - start_time:.2f} seconds. Shape: {similarities_np.shape}")

  
    similar_games = {}
    print(f"LOG: Starting top-N similar games selection for {len(game_ids)} games...")
    start_time = time.time()
    for i in range(len(game_ids)):
        if i > 0 and i % 500 == 0:
            print(f"LOG: Processed top-N selection for {i}/{len(game_ids)} games...")
        game_similarities = similarities_np[i]
        top_indices = np.argsort(game_similarities)[::-1][:top_n + 1]
        top_indices = top_indices[top_indices != i][:top_n]
        similar_games[game_ids[i]] = {
            'similar_games': [game_ids[idx] for idx in top_indices],
            'similarities': [game_similarities[idx] for idx in top_indices]
        }
    print(f"LOG: Finished top-N similar games selection in {time.time() - start_time:.2f} seconds.")

    
    print("LOG: Saving similarity results...")
    start_time = time.time()
    np.save(output_file + '.item_similarities.npy', similarities_np) 

   
    with open(output_file + '.top_similar_games.txt', 'w') as f:
        for game_id, data in similar_games.items():
            f.write(f"Game {game_id}:\n")
            for similar_game, similarity_val in zip(data['similar_games'], data['similarities']):
                f.write(f"  - {similar_game} (similarity: {similarity_val:.4f})\n")
            f.write("\n")
    print(f"LOG: Finished saving similarity results in {time.time() - start_time:.2f} seconds.")
    print(f"LOG: Item-item similarity computation complete in {time.time() - overall_start_time:.2f} seconds.")
    return similar_games


def load_game_details(game_file):
    """
    Load detailed game information from the game data file.
    Returns a dictionary mapping game IDs to their details.
    """
    print(f"LOG: Starting to load game details from {game_file}...")
    start_time = time.time()
    game_details = {}
    try:
        with open(game_file, 'r') as f:
            games_list = json.load(f)
            print(f"LOG: Loaded {len(games_list)} raw game entries from JSON.")
            for idx, game_data in enumerate(games_list):
                if idx == 0:
                    print("LOG: First raw game entry from JSON:")
        
                game_id = str(game_data.get('id', ''))
                if not game_id:
                    print(f"LOG: Warning: Entry {idx+1} in game_file missing 'id' field, skipping.")
                    continue
                details = {
                    'name': game_data.get('name', 'Unknown'),
                    'publisher': game_data.get('publisher', 'Unknown'),
                    'genres': game_data.get('genres', []),
                    'price': float(game_data.get('price', 0.0)),
                    'release_date': game_data.get('release_date', 'Unknown'),
                    'developer': game_data.get('developer', 'Unknown'),
                    'tags': game_data.get('tags', []),
                    'specs': game_data.get('specs', []),
                    'url': game_data.get('url', 'N/A')
                }
                game_details[game_id] = details
        print(f"LOG: Successfully processed and loaded details for {len(game_details)} games.")
        if game_details:
            first_game_id = next(iter(game_details))
            print("LOG: Details for first processed game entry (from dict):")
           
        else:
            print("LOG: No game details were loaded into the dictionary!")
        print(f"LOG: Finished loading game details in {time.time() - start_time:.2f} seconds.")
        return game_details
    except FileNotFoundError:
        print(f"Error: Game file {game_file} not found!")
        return {}
    except Exception as e:
        print(f"LOG: Error loading game details: {str(e)}")
        return {}



def _generate_recommendations_for_user(user_idx, user_id, matrix_np, game_ids, similar_games, game_details, top_n, min_playtime_threshold, game_id_to_idx_map):
    user_playtimes = matrix_np[user_idx]
    played_games_indices = np.where(user_playtimes > min_playtime_threshold)[0]

    if len(played_games_indices) == 0:
        return user_id, None

    game_scores = {}
    for played_game_idx in played_games_indices:
        played_game_id = game_ids[played_game_idx]
        playtime = user_playtimes[played_game_idx]

        
        if played_game_id not in similar_games:
            continue

        similar_data = similar_games[played_game_id]

        for similar_game_id, similarity_val in zip(similar_data['similar_games'], similar_data['similarities']):         
            if similar_game_id not in game_id_to_idx_map:
                continue
            similar_game_idx = game_id_to_idx_map[similar_game_id]

            if user_playtimes[similar_game_idx] > 0:
                continue

            score = similarity_val * playtime
            if similar_game_id in game_scores:
                game_scores[similar_game_id] += score
            else:
                game_scores[similar_game_id] = score

    if game_scores:
        top_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommended_games_details = []
        for game_id_val, score_val in top_games:
            details = game_details.get(str(game_id_val), {})
            game_info = {
                'game_id': str(game_id_val),
                'score': float(score_val),
                'title': details.get('name', 'N/A'),
                'price': details.get('price', 0.0),
                'publisher': details.get('publisher', 'N/A'),
                'genres': details.get('genres', []),
                'url': details.get('url', 'N/A')
            }
            recommended_games_details.append(game_info)
        return user_id, {'recommended_games': recommended_games_details}
    return user_id, None


def generate_recommendations(matrix_np, user_ids, game_ids, similar_games, game_details, output_file, top_n=10,
                             min_playtime_threshold=0.5):  # Takes NumPy matrix
    print(f"LOG: Starting personalized recommendation generation for {len(user_ids)} users...")
    overall_start_time = time.time()

    game_id_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}

    delayed_chunk_recs = [] 
    print(f"LOG: Building Dask delayed graph for user recommendations (in chunks)...")
    start_time = time.time()

    chunk_size = 500  
    user_indices = list(range(len(user_ids)))

    for i in range(0, len(user_indices), chunk_size):
        current_chunk_indices = user_indices[i:i + chunk_size]

        if i > 0 and (i // chunk_size) % (
        max(1, (len(user_ids) // chunk_size) // 20)) == 0:  
            print(f"LOG: Prepared Dask tasks for {i}/{len(user_ids)} users (processing in chunks)...")

        task = delayed(_process_user_chunk)(
            current_chunk_indices,  
            user_ids, 
            matrix_np,
            game_ids,
            similar_games,
            game_details,
            top_n,
            min_playtime_threshold,
            game_id_to_idx
        )
        delayed_chunk_recs.append(task)

    num_chunks = len(delayed_chunk_recs)
    print(
        f"LOG: Dask delayed graph for {len(user_ids)} users (in {num_chunks} chunks) built in {time.time() - start_time:.2f} seconds.")

    print(f"LOG: Computing recommendations for {len(user_ids)} users (via {num_chunks} Dask tasks)...")
    start_time = time.time()
   
    results_from_chunks = compute(*delayed_chunk_recs)
    print(f"LOG: Dask computation for recommendations finished in {time.time() - start_time:.2f} seconds.")

   
    user_recs_list_flat = []
    for chunk_result in results_from_chunks:
        if chunk_result:  
            user_recs_list_flat.extend(chunk_result)

    recommendations = {}
    processed_recs = 0
    for res_user_id, rec_data in user_recs_list_flat:
        if rec_data:
            recommendations[res_user_id] = rec_data
            processed_recs += 1
    print(f"LOG: Aggregated recommendations for {processed_recs} users.")

   
    print("LOG: Saving recommendations...")
    start_time_save = time.time() 
    recommendations_json_file = output_file + '.recommendations.json'
    try:
        with open(recommendations_json_file, 'w') as f_json:
            json.dump(recommendations, f_json, indent=4)
        print(f"LOG: Recommendations saved to {recommendations_json_file}")
    except IOError as e:
        print(f"LOG: Error saving recommendations to JSON file {recommendations_json_file}: {e}")
    except TypeError as e:
        print(f"LOG: Error serializing recommendations to JSON: {e}")

    print(f"LOG: Recommendation generation and saving complete in {time.time() - overall_start_time:.2f} seconds.")
    return recommendations


def _process_user_chunk(user_indices_chunk, all_user_ids, matrix_np, game_ids, similar_games, game_details, top_n,
                        min_playtime_threshold, game_id_to_idx_map):
    """Processes a chunk of users and returns their recommendations."""
    chunk_recommendations = []
    for user_idx_in_chunk, actual_user_idx in enumerate(user_indices_chunk):
        user_id = all_user_ids[actual_user_idx]

        
        processed_user_id, rec_data = _generate_recommendations_for_user(
            actual_user_idx,
            user_id,
            matrix_np,
            game_ids,
            similar_games,
            game_details,
            top_n,
            min_playtime_threshold,
            game_id_to_idx_map
        )
        chunk_recommendations.append((processed_user_id, rec_data))
    return chunk_recommendations

def _evaluate_single_user_task(user_idx, matrix_np, game_ids, similar_games, game_id_to_idx_map, top_n, min_playtime_threshold, seed_offset):
   
    np.random.seed(42 + seed_offset) 

    user_playtimes = matrix_np[user_idx] 
    played_games_indices = np.where(user_playtimes > min_playtime_threshold)[0]

   
    if len(played_games_indices) < 2:
        return None

    hidden_game_idx = np.random.choice(played_games_indices)
    temp_user_playtimes = user_playtimes.copy()
    temp_user_playtimes[hidden_game_idx] = 0

    current_played_games_for_scoring = np.where(temp_user_playtimes > min_playtime_threshold)[0]
    game_scores = defaultdict(float)

    for played_game_idx_for_scoring in current_played_games_for_scoring:
        playtime_for_scoring = temp_user_playtimes[played_game_idx_for_scoring]
        pg_id_str = game_ids[played_game_idx_for_scoring]

        if pg_id_str not in similar_games:
            
            continue
        similar_data = similar_games[pg_id_str]

        for sim_game_id, sim_score in zip(similar_data['similar_games'], similar_data['similarities']):
            if sim_game_id not in game_id_to_idx_map:
            
                continue
            sim_game_idx = game_id_to_idx_map[sim_game_id]
            if temp_user_playtimes[sim_game_idx] == 0:
                game_scores[sim_game_idx] += sim_score * playtime_for_scoring

    sorted_recommended_indices = sorted(game_scores.keys(), key=lambda k: game_scores[k], reverse=True)
    top_n_recommended_indices = sorted_recommended_indices[:top_n]

    hit = 0
    rank = float('inf')
    if hidden_game_idx in top_n_recommended_indices:
        hit = 1
        try:
            rank = top_n_recommended_indices.index(hidden_game_idx) + 1
        except ValueError: pass
    else:
        try:
            rank = sorted_recommended_indices.index(hidden_game_idx) + 1
        except ValueError: pass

    recommended_this_user = 1 if top_n_recommended_indices else 0
    return {'hit': hit, 'rank': rank, 'recommended_this_user': recommended_this_user}


def build_user_game_matrix(user_file, game_file, output_file):
    """
    Build a user-game matrix from user and game data files.
    """
    print(f"Loading user data from {user_file}...")
    start_time = time.time()
    user_data = []
    with open(user_file, 'r') as f:
        for line in f:
            user_data.append(safe_literal_eval(line))
    print(f"Loaded {len(user_data)} users' data.")

    print(f"Loading game details from {game_file}...")
    game_details = load_game_details(game_file)

    print("Extracting interactions...")
    user_interactions = []
    for user_dict in user_data:
        user_interactions.extend(extract_interactions(user_dict))
    print(f"Extracted {len(user_interactions)} interactions.")

    print("Building user-game matrix...")
    user_ids = sorted(set(interaction['user_id'] for interaction in user_interactions))
    game_ids = sorted(set(interaction['game_id'] for interaction in user_interactions))
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    game_id_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}

    matrix = np.zeros((len(user_ids), len(game_ids)))
    for interaction in user_interactions:
        user_idx = user_id_to_idx[interaction['user_id']]
        game_idx = game_id_to_idx[interaction['game_id']]
        matrix[user_idx, game_idx] = interaction['playtime']

    print(f"User-game matrix shape: {matrix.shape}")

    print("Converting matrix to Dask array...")
    matrix_da = da.from_array(matrix, chunks=(1000, 1000))

    print("Computing item similarities...")
    similar_games = compute_item_similarities(matrix_da, game_ids, output_file)

    print("Generating recommendations...")
    recommendations = generate_recommendations(matrix, user_ids, game_ids, similar_games, game_details, output_file)

    print(f"Total processing time for build_user_game_matrix (including all steps): {time.time() - start_time:.2f} seconds")
    return matrix, user_ids, game_ids, similar_games, recommendations

def load_recommendations_from_file(recommendations_file_path):
    """Loads recommendations from a JSON file."""
    print(f"LOG: Attempting to load recommendations from {recommendations_file_path}...")
    try:
        with open(recommendations_file_path, 'r') as f:
            recommendations = json.load(f)
        print(f"LOG: Successfully loaded recommendations for {len(recommendations)} users.")
        return recommendations
    except FileNotFoundError:
        print(f"LOG: ERROR - Recommendations file not found: {recommendations_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"LOG: ERROR - Could not decode JSON from recommendations file: {recommendations_file_path}")
        return None
    except Exception as e:
        print(f"LOG: ERROR - An unexpected error occurred while loading recommendations: {e}")
        return None


if __name__ == '__main__':
    print("LOG: Script execution started.")
    main_start_time = time.time()
    user_file = 'australian_users_items.json'
    game_file = 'cleaned_steam_games.json'  # Using the cleaned file
    output_file = 'user_game_matrix'
    recommendations_json_file = output_file + '.recommendations.json'


    try:
        existing_recommendations = load_recommendations_from_file(recommendations_json_file)
        if not existing_recommendations:
            print(f"LOG: Recommendations file {recommendations_json_file} is empty or invalid. Regenerating...")
            raise FileNotFoundError 
        print(f"LOG: Found existing and valid recommendations file: {recommendations_json_file}")
        all_recommendations = existing_recommendations
    except FileNotFoundError:
        print(f"LOG: Recommendations file not found or needs regeneration. Running full pipeline...")
        _, _, _, _, all_recommendations = build_user_game_matrix(user_file, game_file, output_file)
        if not all_recommendations:
            print("LOG: ERROR - Full pipeline ran but failed to generate recommendations. Exiting.")
            exit()

    print(f"LOG: Script setup finished in {time.time() - main_start_time:.2f} seconds.")

    while True:
        try:
            user_id_to_lookup = input("Enter a User ID to get recommendations (or type 'exit' to quit): ").strip()
            if user_id_to_lookup.lower() == 'exit':
                print("LOG: Exiting interactive mode.")
                break

            if not user_id_to_lookup:
                print("LOG: Please enter a User ID.")
                continue

            if all_recommendations:
                user_specific_recs = all_recommendations.get(user_id_to_lookup)
                if user_specific_recs and 'recommended_games' in user_specific_recs:
                    print(f"\n--- Recommendations for User ID: {user_id_to_lookup} ---")
                    if user_specific_recs['recommended_games']:
                        for i, game_rec in enumerate(user_specific_recs['recommended_games']):
                            title = game_rec.get('title', 'N/A')
                            game_id = game_rec.get('game_id', 'N/A')
                            score = game_rec.get('score', 0.0)
                            price = game_rec.get('price', 'N/A')
                            genres = ", ".join(game_rec.get('genres', [])) or 'N/A'
                            url = game_rec.get('url', 'N/A')
                            print(f"  {i+1}. Title: {title} (ID: {game_id})")
                            print(f"     Score: {score:.4f}")
                            print(f"     Price: {price}")
                            print(f"     Genres: {genres}")
                            print(f"     URL: {url}")
                            print("     ----------")
                    else:
                        print("  No game recommendations found for this user (list was empty).")
                    print("--- End of Recommendations ---\n")
                else:
                    print(f"LOG: No recommendations found for User ID: {user_id_to_lookup}. They might not be in the dataset or have no generated recommendations.")
            else:
                print("LOG: ERROR - Recommendations data is not loaded. Cannot provide recommendations.")
                break 

        except KeyboardInterrupt:
            print("\nLOG: Keyboard interrupt detected. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"LOG: An error occurred during interactive lookup: {e}")


    print(f"LOG: Total script execution time: {time.time() - main_start_time:.2f} seconds.")
