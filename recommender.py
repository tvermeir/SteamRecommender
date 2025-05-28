import numpy as np
import ast
import time
import json
import dask
import dask.bag as db
from dask.diagnostics import ProgressBar


# It's often good practice to use a Dask client, especially for diagnostics or more complex schedulers
# from dask.distributed import Client
# client = Client() # Initializes a local Dask cluster (multi-processing by default)
# print(f"Dask dashboard link: {client.dashboard_link}")

def process_user_line(line):
    try:
        line = line.strip()
        if not line:
            return None

        user_data = ast.literal_eval(line)
        user_id = user_data['user_id']
        items = user_data['items']
        game_playtimes = []
        for item in items:
            game_id = item['item_id']  # Assuming item_id is suitable as a direct key
            playtime = item['playtime_forever']
            log_playtime = np.log1p(playtime)
            game_playtimes.append((str(game_id), log_playtime))  # Ensure game_id is string for consistency

        return str(user_id), game_playtimes  # Ensure user_id is string
    except (ValueError, SyntaxError, KeyError) as e:
        # print(f"Error processing line: {e}") # Can be very verbose with Dask
        # print(f"Problematic line: {line[:100]}...")
        return None


def compute_cosine_similarity(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    return np.dot(X_norm, X_norm.T)


# Helper for dask.delayed in compute_item_similarities
def get_top_n_for_game(game_idx, game_id_str, single_game_similarities_row, all_game_ids_list, top_n_val):
    # Exclude self (game_idx)
    # Add 1 to top_n because we might remove self
    top_indices_with_self = np.argsort(single_game_similarities_row)[::-1][:top_n_val + 1]
    # Remove self from results
    top_indices = top_indices_with_self[top_indices_with_self != game_idx][:top_n_val]

    return game_id_str, {
        'similar_games': [all_game_ids_list[idx] for idx in top_indices],
        'similarities': [single_game_similarities_row[idx] for idx in top_indices]
    }


def compute_item_similarities(matrix, game_ids_list, output_file, top_n=10):
    print("Computing item-item similarities...")
    start_time = time.time()

    similarities_matrix = compute_cosine_similarity(matrix.T)
    print(f"Raw similarity matrix computed. Shape: {similarities_matrix.shape}")

    similar_games_delayed_tasks = []
    num_games = len(game_ids_list)
    print(f"Setting up Dask tasks for top N similar games for {num_games} games...")
    log_interval_games = 100  # Log every 100 games
    if num_games < log_interval_games * 2:  # if total games is small, log more frequently or every time
        log_interval_games = 1

    for i in range(num_games):
        print("hey")
        if (i + 1) % log_interval_games == 0 or (i + 1) == num_games:
            print(f"  Setting up delayed task for game {i + 1}/{num_games} (ID: {game_ids_list[i]})...")
        task = dask.delayed(get_top_n_for_game)(i, game_ids_list[i], similarities_matrix[i], game_ids_list, top_n)
        similar_games_delayed_tasks.append(task)

    print("Computing top N similar games with Dask...")
    with ProgressBar():
        results = dask.compute(*similar_games_delayed_tasks)

    similar_games_dict = dict(results)

    print("\nSaving similarity results...")
    np.save(output_file + '.item_similarities.npy', similarities_matrix)

    with open(output_file + '.top_similar_games.txt', 'w') as f:
        for game_id, data in similar_games_dict.items():
            f.write(f"Game {game_id}:\n")
            for similar_game, similarity_val in zip(data['similar_games'], data['similarities']):
                f.write(f"  - {similar_game} (similarity: {similarity_val:.4f})\n")
            f.write("\n")

    print(f"\nSimilarity computation complete in {time.time() - start_time:.2f} seconds")
    return similar_games_dict


def load_game_details_dask(game_file):
    print(f"Loading game details from {game_file} using Dask Bag...")

    def parse_game_line(line):
        try:
            line = line.strip()
            if not line: return None
            game_data = ast.literal_eval(line)
            game_id = game_data.get('id')
            if game_id is None: return None
            return str(game_id), game_data  # Ensure game_id is string
        except (ValueError, SyntaxError, KeyError, AttributeError) as e:
            # print(f"Error decoding game JSON: {e}, Line: {line[:100]}") # Verbose
            return None

    bag = db.read_text(game_file, blocksize='64MB').map(parse_game_line).filter(lambda x: x is not None)

    print("Computing game details from Dask Bag...")
    with ProgressBar():
        game_details_list = bag.compute()

    game_details_dict = dict(game_details_list)
    print(f"Loaded details for {len(game_details_dict)} games.")
    return game_details_dict


def build_user_game_matrix(user_file, game_file, output_file):
    print("Starting matrix construction...")
    start_time = time.time()

    game_details = load_game_details_dask(game_file)  # Using Dask version

    print("\nFirst pass: collecting users and games using Dask Bag...")

    user_lines_bag = db.read_text(user_file, blocksize='64MB')
    processed_user_bag = user_lines_bag.map(process_user_line).filter(lambda x: x is not None)

    print("Computing user playtimes from Dask Bag...")
    with ProgressBar():
        user_centric_playtimes = processed_user_bag.compute()

    print(f"\nProcessed {len(user_centric_playtimes)} users from file.")

    user_ids_ordered_list = []
    user_id_to_matrix_idx = {}
    all_game_ids_set = set()
    matrix_fill_data = []

    print("Building internal user/game mappings and matrix fill data...")
    num_processed_users = len(user_centric_playtimes)
    log_interval_mapping = 10000  # Log every 10000 users during this phase
    if num_processed_users < log_interval_mapping * 2:
        log_interval_mapping = 1000  # adjust if fewer users
        if num_processed_users < log_interval_mapping * 2:
            log_interval_mapping = 100

    for idx, (user_id_str, game_playtimes_list) in enumerate(user_centric_playtimes):
        if (idx + 1) % log_interval_mapping == 0 or (idx + 1) == num_processed_users:
            print(f"  Mapping user {idx + 1}/{num_processed_users}...")

        if user_id_str not in user_id_to_matrix_idx:
            user_id_to_matrix_idx[user_id_str] = len(user_ids_ordered_list)
            user_ids_ordered_list.append(user_id_str)

        current_user_idx = user_id_to_matrix_idx[user_id_str]

        for game_id_str, playtime_float in game_playtimes_list:
            all_game_ids_set.add(game_id_str)
            matrix_fill_data.append((current_user_idx, game_id_str, playtime_float))

    game_ids_ordered_list = sorted(list(all_game_ids_set))
    game_id_to_matrix_idx = {gid: i for i, gid in enumerate(game_ids_ordered_list)}

    print(
        f"\nFirst pass (Dask Bag + local processing) complete. Found {len(user_ids_ordered_list)} users and {len(game_ids_ordered_list)} unique games.")

    print("\nCreating and filling matrix...")
    matrix = np.zeros((len(user_ids_ordered_list), len(game_ids_ordered_list)), dtype=np.float32)

    fill_count = 0
    num_matrix_fill_entries = len(matrix_fill_data)
    log_interval_fill = 100000
    if num_matrix_fill_entries < log_interval_fill * 2:
        log_interval_fill = 10000

    for i, (user_idx, game_id_str, playtime_float) in enumerate(matrix_fill_data):
        game_idx = game_id_to_matrix_idx.get(game_id_str)
        if game_idx is not None:
            matrix[user_idx, game_idx] = playtime_float
            fill_count += 1  # Only count actual fills
        if (i + 1) % log_interval_fill == 0 or (i + 1) == num_matrix_fill_entries:
            print(
                f"  Processed {i + 1}/{num_matrix_fill_entries} potential matrix entries. Actual fills: {fill_count}...")

    print("\nSaving results...")
    np.save(output_file + '.matrix.npy', matrix)
    np.save(output_file + '.user_ids.npy', np.array(user_ids_ordered_list, dtype=object))
    np.save(output_file + '.game_ids.npy', np.array(game_ids_ordered_list, dtype=object))

    print("\nFinal statistics:")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Number of users: {len(user_ids_ordered_list)}")
    print(f"Number of games: {len(game_ids_ordered_list)}")

    similar_games = compute_item_similarities(matrix, game_ids_ordered_list, output_file)

    recommendations = generate_recommendations_dask(matrix, user_ids_ordered_list, game_ids_ordered_list,
                                                    similar_games, game_details, game_id_to_matrix_idx,
                                                    output_file)

    print(f"Total processing time for build_user_game_matrix: {time.time() - start_time:.2f} seconds")
    return matrix, user_ids_ordered_list, game_ids_ordered_list, similar_games, recommendations


def generate_recs_for_single_user(user_idx, user_id_str, user_playtimes_row,
                                  all_game_ids_list,
                                  game_id_to_matrix_idx_map,
                                  similar_games_dict, game_details_dict,
                                  top_n_recs, min_playtime_thresh):
    played_game_indices = np.where(user_playtimes_row > min_playtime_thresh)[0]

    if len(played_game_indices) == 0:
        return user_id_str, None

    game_scores = {}
    for played_game_idx in played_game_indices:
        played_game_id_str = all_game_ids_list[played_game_idx]
        playtime = float(user_playtimes_row[played_game_idx])

        if played_game_id_str not in similar_games_dict:
            continue
        similar_data = similar_games_dict[played_game_id_str]

        for similar_game_id_str, similarity_score in zip(similar_data['similar_games'], similar_data['similarities']):
            similar_game_matrix_idx = game_id_to_matrix_idx_map.get(similar_game_id_str)

            if similar_game_matrix_idx is None:
                continue

            if user_playtimes_row[similar_game_matrix_idx] > 0:
                continue

            score = float(similarity_score) * playtime
            game_scores[similar_game_id_str] = game_scores.get(similar_game_id_str, 0) + score

    if game_scores:
        top_games_list = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_recs]
        return user_id_str, {
            'recommendations': [
                {
                    'game_id': game_id_val,
                    'score': float(score_val),
                    'details': game_details_dict.get(game_id_val, {})
                }
                for game_id_val, score_val in top_games_list
            ]
        }
    return user_id_str, None


def generate_recommendations_dask(matrix, user_ids_list, game_ids_list,
                                  similar_games_map, game_details_map, game_id_to_idx_map,
                                  output_file, top_n=10, min_playtime_threshold=0.5):
    print("Generating personalized recommendations using Dask...")
    start_time = time.time()

    delayed_recommendations = []
    num_users = len(user_ids_list)
    print(f"Setting up Dask tasks for {num_users} users...")
    log_interval_users = 1000  # Log every 1000 users
    if num_users < log_interval_users * 2:  # if total users is small, log more frequently or every time
        log_interval_users = 100
        if num_users < log_interval_users * 2:
            log_interval_users = 10

    for user_idx, user_id_str in enumerate(user_ids_list):
        if (user_idx + 1) % log_interval_users == 0 or (user_idx + 1) == num_users:
            print(f"  Setting up delayed task for user {user_idx + 1}/{num_users} (ID: {user_id_str})...")

        user_playtimes_data = matrix[user_idx]
        task = dask.delayed(generate_recs_for_single_user)(
            user_idx, user_id_str, user_playtimes_data,
            game_ids_list, game_id_to_idx_map,
            similar_games_map, game_details_map,
            top_n, min_playtime_threshold
        )
        delayed_recommendations.append(task)

    print("Computing recommendations with Dask...")
    with ProgressBar():
        results = dask.compute(*delayed_recommendations)

    recommendations_dict = {}
    for user_id_key, rec_data in results:
        if rec_data:
            recommendations_dict[user_id_key] = rec_data

    print("\nSaving recommendations to JSON file...")
    with open(output_file + '.recommendations.json', 'w') as f:
        json.dump(recommendations_dict, f, indent=4)

    print(f"\nRecommendation generation complete in {time.time() - start_time:.2f} seconds")
    return recommendations_dict


def get_user_recommendations_from_file(user_id_to_find, recommendations_file_path):
    try:
        with open(recommendations_file_path, 'r') as f:
            all_recommendations = json.load(f)
        user_id_key = str(user_id_to_find)
        if user_id_key in all_recommendations:
            return all_recommendations[user_id_key]
        else:
            print(f"User ID '{user_id_key}' not found in the recommendations file.")
            return None
    except FileNotFoundError:
        print(f"Error: Recommendations file not found at '{recommendations_file_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{recommendations_file_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def evaluate_recommender(matrix, user_ids, game_ids, similar_games, output_file, top_n=10, min_playtime_threshold=0.5,
                         test_fraction=0.1):
    print("\n=== Starting Recommender System Evaluation ===")
    hit_rate, mean_rank, coverage_rate = 0.0, float('inf'), 0.0
    start_time_eval = time.time()
    print("\n=== Evaluation Results ===")
    print(f"Evaluation complete in {time.time() - start_time_eval:.2f} seconds")
    return hit_rate, mean_rank, coverage_rate


if __name__ == '__main__':
    user_file = 'australian_users_items.json'
    game_file = 'cleaned_steam_games.txt'
    output_file = 'user_game_matrix_dask'
    recommendations_file = output_file + '.recommendations.json'

    # To build the matrix and recommendations (run once, or if data changes):
    # print("Building matrix and generating recommendations for the first time...")
    # build_user_game_matrix(user_file, game_file, output_file)
    # print("\nBuild and recommendation process complete.")
    # print(f"Recommendation file generated at: {recommendations_file}")

    try:
        with open(recommendations_file, 'r') as f:
            pass
        print(f"\nFound recommendations file: {recommendations_file}")
    except FileNotFoundError:
        print(f"Recommendations file {recommendations_file} not found!")
        print("Attempting to build it now...")
        build_user_game_matrix(user_file, game_file, output_file)
        print("\nBuild and recommendation process complete.")

    while True:
        print("\n--- Game Recommendation Search ---")
        target_user_id_input = input("Enter your User ID to get recommendations (or type 'quit' to exit): ").strip()

        if target_user_id_input.lower() == 'quit':
            print("Exiting recommendation search.")
            break
        if not target_user_id_input:
            print("User ID cannot be empty. Please try again.")
            continue

        print(f"\nFetching recommendations for user: {target_user_id_input}...")
        user_recs = get_user_recommendations_from_file(target_user_id_input, recommendations_file)

        if user_recs and user_recs.get('recommendations'):
            print(f"Top game recommendations for {target_user_id_input}:")
            recommendations_list = user_recs['recommendations']
            for i, rec in enumerate(recommendations_list, 1):
                game_id = rec.get('game_id')
                score = rec.get('score')
                details = rec.get('details', {})
                game_title = details.get('title', details.get('app_name', 'N/A'))
                print(f"  {i}. Game ID: {game_id}, Title: {game_title}, Score: {score:.4f}")
        elif user_recs and not user_recs.get('recommendations'):
            print(
                f"User {target_user_id_input} was processed, but no recommendations could be generated (e.g., played too few games or no similar unplayed games).")
        else:
            pass