import numpy as np
import ast  # For safely evaluating Python literals
import time
from collections import defaultdict
import json


def process_user_line(line):
    """Process a single user's data line."""
    try:
        # Strip any whitespace and ensure the line is a complete dictionary
        line = line.strip()
        if not line:  # Skip empty lines
            return None

        # Use ast.literal_eval instead of json.loads to handle Python dictionary format
        user_data = ast.literal_eval(line)
        user_id = user_data['user_id']
        items = user_data['items']

        # Create list of (game_id, playtime) tuples
        game_playtimes = []
        for item in items:
            game_id = item['item_id']
            playtime = item['playtime_forever']
            # Apply log(1 + x) transformation
            log_playtime = np.log1p(playtime)
            game_playtimes.append((game_id, log_playtime))

        return user_id, game_playtimes
    except (ValueError, SyntaxError, KeyError) as e:
        print(f"Error processing line: {e}")
        print(f"Problematic line: {line[:100]}...")  # Print first 100 chars of problematic line
        return None


def compute_cosine_similarity(X):
    """
    Compute cosine similarity between all pairs of vectors in X.
    X should be a 2D array where each row is a vector.
    """
    # Calculate norms
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Handle zero vectors by setting their norm to 1 (they will have zero similarity with everything)
    norms[norms == 0] = 1

    # Normalize the vectors
    X_norm = X / norms

    # Compute cosine similarity
    return np.dot(X_norm, X_norm.T)


def compute_item_similarities(matrix, game_ids, output_file, top_n=10):
    """
    Compute cosine similarity between games and save the results.

    Args:
        matrix: User-game matrix (users x games)
        game_ids: List of game IDs corresponding to matrix columns
        output_file: Base filename for saving results
        top_n: Number of most similar games to save for each game
    """
    print("Computing item-item similarities...")
    start_time = time.time()

    # Compute cosine similarity between game vectors (columns)
    # Transpose matrix to get games as rows
    similarities = compute_cosine_similarity(matrix.T)

    # Create a dictionary to store top N similar games for each game
    similar_games = {}

    # For each game, find its top N most similar games
    for i in range(len(game_ids)):
        if i % 100 == 0:
            print(f"Processing game {i}/{len(game_ids)}...")
        # Get similarities for this game
        game_similarities = similarities[i]
        # Get indices of top N similar games (excluding self)
        # Add 1 to top_n because we'll remove self later
        top_indices = np.argsort(game_similarities)[::-1][:top_n + 1]
        # Remove self from results
        top_indices = top_indices[top_indices != i][:top_n]
        # Store results
        similar_games[game_ids[i]] = {
            'similar_games': [game_ids[idx] for idx in top_indices],
            'similarities': [game_similarities[idx] for idx in top_indices]
        }

    # Save results
    print("\nSaving similarity results...")
    np.save(output_file + '.item_similarities.npy', similarities)

    # Save top N similar games in a more readable format
    with open(output_file + '.top_similar_games.txt', 'w') as f:
        for game_id, data in similar_games.items():
            f.write(f"Game {game_id}:\n")
            for similar_game, similarity in zip(data['similar_games'], data['similarities']):
                f.write(f"  - {similar_game} (similarity: {similarity:.4f})\n")
            f.write("\n")

    print(f"\nSimilarity computation complete in {time.time() - start_time:.2f} seconds")
    return similar_games


def load_game_details(game_file):
    """
    Load detailed game information from the game data file.
    Returns a dictionary mapping game IDs to their details.
    """
    print(f"Loading game details from {game_file}...")
    game_details = {}
    try:
        # Read the entire JSON file as a list
        with open(game_file, 'r') as f:
            games_list = json.load(f)
            print(f"Loaded {len(games_list)} games from JSON.")
            for idx, game_data in enumerate(games_list):
                if idx == 0:
                    print("\nFirst game entry:")
                    print(game_data)
                game_id = str(game_data.get('id', ''))
                if not game_id:
                    print(f"Warning: Entry {idx+1} missing 'id' field")
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
        print(f"\nLoaded details for {len(game_details)} games")
        if game_details:
            first_game_id = next(iter(game_details))
            print("\nFirst game entry (from dict):")
            print(f"Game ID: {first_game_id}")
            for key, value in game_details[first_game_id].items():
                print(f"{key}: {value}")
        else:
            print("No game details were loaded!")
        return game_details
    except FileNotFoundError:
        print(f"Error: Game file {game_file} not found!")
        return {}
    except Exception as e:
        print(f"Error loading game details: {str(e)}")
        return {}


def generate_recommendations(matrix, user_ids, game_ids, similar_games, game_details, output_file, top_n=10,
                             min_playtime_threshold=0.5):
    """
    Generate personalized game recommendations for each user.

    Args:
        matrix: User-game matrix (users x games)
        user_ids: List of user IDs
        game_ids: List of game IDs
        similar_games: Dictionary of similar games for each game
        game_details: Dictionary of game details
        output_file: Base filename for saving results
        top_n: Number of recommendations to generate per user
        min_playtime_threshold: Minimum normalized playtime to consider a game as "high-playtime"
    """
    print("Generating personalized recommendations...")
    start_time = time.time()

    # Create a mapping from game_id to index
    game_id_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}

    # For each user, generate recommendations
    recommendations = {}

    for user_idx, user_id in enumerate(user_ids):
        if user_idx % 1000 == 0:
            print(f"Processing user {user_idx}/{len(user_ids)}...")

        # Get user's playtime vector
        user_playtimes = matrix[user_idx]

        # Find games the user has played significantly
        played_games = np.where(user_playtimes > min_playtime_threshold)[0]

        if len(played_games) == 0:
            continue

        # Calculate recommendation scores for each unplayed game
        game_scores = {}
        for played_game_idx in played_games:
            played_game_id = game_ids[played_game_idx]
            playtime = user_playtimes[played_game_idx]
            # Get similar games for this played game
            similar_data = similar_games[played_game_id]

            # Add weighted scores for each similar game
            for similar_game_id, similarity in zip(similar_data['similar_games'], similar_data['similarities']):
                similar_game_idx = game_id_to_idx[similar_game_id]
                # Skip if user has already played this game
                if user_playtimes[similar_game_idx] > 0:
                    continue
                # Score = similarity * playtime
                score = similarity * playtime
                if similar_game_id in game_scores:
                    game_scores[similar_game_id] += score
                else:
                    game_scores[similar_game_id] = score

        # Sort games by score and get top N
        if game_scores:
            top_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

            recommended_games_details = []
            for game_id_val, score_val in top_games:
                details = game_details.get(str(game_id_val), {}) # Ensure game_id is string for lookup
                game_info = {
                    'game_id': str(game_id_val),
                    'score': float(score_val),
                    'title': details.get('name', 'N/A'), # 'name' is used as title in load_game_details
                    'price': details.get('price', 0.0),
                    'publisher': details.get('publisher', 'N/A'),
                    'genres': details.get('genres', []),
                    'url': details.get('url', 'N/A') # Assuming 'url' key exists in details
                }
                recommended_games_details.append(game_info)

            recommendations[user_id] = {
                'recommended_games': recommended_games_details
            }

    # Save recommendations in a readable format
    print("\nSaving recommendations...")

    # Save recommendations to a JSON file
    recommendations_json_file = output_file + '.recommendations.json'
    try:
        with open(recommendations_json_file, 'w') as f_json:
            json.dump(recommendations, f_json, indent=4)
        print(f"Recommendations saved to {recommendations_json_file}")
    except IOError as e:
        print(f"Error saving recommendations to JSON file {recommendations_json_file}: {e}")
    except TypeError as e:
        print(f"Error serializing recommendations to JSON: {e}")

    print(f"\nRecommendation generation complete in {time.time() - start_time:.2f} seconds")
    return recommendations


def evaluate_recommender(matrix, user_ids, game_ids, similar_games, output_file, top_n=10, min_playtime_threshold=0.5,
                         test_fraction=0.1):
    """
    Evaluate the recommender system using leave-one-out validation.
    """
    print("\n=== Starting Recommender System Evaluation ===")
    print(f"Parameters:")
    print(f"- Top N recommendations: {top_n}")
    print(f"- Minimum playtime threshold: {min_playtime_threshold}")
    print(f"- Test fraction: {test_fraction}")
    start_time = time.time()

    # Create a mapping from game_id to index
    game_id_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}

    # Select random users for testing
    np.random.seed(42)  # For reproducibility
    test_user_indices = np.random.choice(len(user_ids), size=int(len(user_ids) * test_fraction), replace=False)
    print(f"\nSelected {len(test_user_indices)} users for testing")

    # Pre-compute played games for all test users
    print("Pre-computing played games...")
    played_games_mask = matrix[test_user_indices] > min_playtime_threshold
    valid_users = np.where(np.sum(played_games_mask, axis=1) >= 2)[0]
    test_user_indices = test_user_indices[valid_users]
    print(f"Found {len(test_user_indices)} valid users with sufficient games")

    # Pre-compute similarity matrix for all games
    print("Pre-computing similarity matrix...")
    similarity_matrix = np.zeros((len(game_ids), len(game_ids)))
    for i, game_id in enumerate(game_ids):
        similar_data = similar_games[game_id]
        for similar_id, similarity in zip(similar_data['similar_games'], similar_data['similarities']):
            j = game_id_to_idx[similar_id]
            similarity_matrix[i, j] = similarity

    # Metrics to track
    hit_rates = []
    ranks = []
    coverage = []
    successful_hits = 0
    total_attempts = 0
    skipped_users = 0

    # Process users in batches
    batch_size = 100
    for batch_start in range(0, len(test_user_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(test_user_indices))
        batch_indices = test_user_indices[batch_start:batch_end]
        print(f"\nProcessing users {batch_start + 1}-{batch_end} of {len(test_user_indices)}...")

        # Get playtime vectors for the batch
        batch_playtimes = matrix[batch_indices]
        for i, user_idx in enumerate(batch_indices):
            user_id = user_ids[user_idx]
            user_playtimes = batch_playtimes[i]

            # Find games the user has played significantly
            played_games_indices = np.where(user_playtimes > min_playtime_threshold)[0]

            if len(played_games_indices) < 2:
                skipped_users += 1
                continue

            # Randomly select one game to hide
            hidden_game_idx = np.random.choice(played_games_indices)
            # hidden_game_id = game_ids[hidden_game_idx] # Not directly used later for rank check by ID

            # Create a modified user playtime vector with the hidden game removed for score calculation
            temp_user_playtimes = user_playtimes.copy()
            temp_user_playtimes[hidden_game_idx] = 0 # Effectively hide this game for score calculation

            # Calculate recommendation scores
            current_played_games_for_scoring = np.where(temp_user_playtimes > min_playtime_threshold)[0]

            game_scores = defaultdict(float)
            for played_game_idx_for_scoring in current_played_games_for_scoring:
                # played_game_id_for_scoring = game_ids[played_game_idx_for_scoring] # Not needed
                playtime_for_scoring = temp_user_playtimes[played_game_idx_for_scoring]

                # Check if played_game_id_for_scoring is in similar_games
                pg_id_str = game_ids[played_game_idx_for_scoring] # Use the actual ID string for dict lookup
                if pg_id_str not in similar_games:
                    continue

                similar_data = similar_games[pg_id_str]

                for sim_game_id, sim_score in zip(similar_data['similar_games'], similar_data['similarities']):
                    sim_game_idx = game_id_to_idx[sim_game_id]
                    if temp_user_playtimes[sim_game_idx] == 0: # If not played (or is the hidden one)
                        game_scores[sim_game_idx] += sim_score * playtime_for_scoring

            # Get top N recommendations (indices)
            # Ensure items are sorted by score, then by game index for tie-breaking (optional but good practice)
            sorted_recommended_indices = sorted(game_scores.keys(), key=lambda k: game_scores[k], reverse=True)
            top_n_recommended_indices = sorted_recommended_indices[:top_n]

            # Check if hidden game is in recommendations
            if hidden_game_idx in top_n_recommended_indices:
                try:
                    rank = top_n_recommended_indices.index(hidden_game_idx) + 1
                    hit_rates.append(1)
                    ranks.append(rank)
                    successful_hits +=1
                except ValueError: # Should not happen if it's in the list
                    pass
            else:
                hit_rates.append(0)
                # If you want to record rank even if not in top_N, find its rank in sorted_recommended_indices
                try:
                    full_rank = sorted_recommended_indices.index(hidden_game_idx) + 1
                    ranks.append(full_rank)
                except ValueError: # Hidden game was not recommended at all
                    ranks.append(float('inf')) # Or some large number / specific indicator


            # Coverage: what fraction of items that could be recommended were ever recommended
            # This is a simpler interpretation of coverage: fraction of test items that appeared in any top-N list
            # For a more standard item coverage, you'd collect all unique recommended items across all test users.
            # Here, we just check if we could make a recommendation for this user.
            if top_n_recommended_indices: # If any recommendations were made
                 coverage.append(1)
            else:
                 coverage.append(0)
            total_attempts += 1

    # Calculate metrics
    hit_rate_metric = np.mean(hit_rates) if hit_rates else 0
    mean_rank_metric = np.mean([r for r in ranks if r != float('inf')]) if ranks and any(r != float('inf') for r in ranks) else float('inf')
    # Catalog coverage: percentage of items in the catalog that were recommended at least once.
    # This requires collecting all unique game_ids from all recommendations.
    # For simplicity, the current 'coverage' list is more like "recommendation opportunity coverage"
    coverage_rate_metric = np.mean(coverage) if coverage else 0


    # Save evaluation results
    print("\n=== Evaluation Results ===")
    evaluation_results_file = output_file + '.evaluation.txt'
    try:
        with open(evaluation_results_file, 'w') as f:
            f.write("Recommender System Evaluation Results:\n")
            f.write("=====================================\n\n")
            f.write("Parameters:\n")
            f.write(f"- Top N recommendations: {top_n}\n")
            f.write(f"- Minimum playtime threshold: {min_playtime_threshold}\n")
            f.write(f"- Test fraction: {test_fraction}\n\n")
            f.write("Test Statistics:\n")
            f.write(f"- Total initial test users: {len(np.random.choice(len(user_ids), size=int(len(user_ids) * test_fraction), replace=False))}\n")
            f.write(f"- Valid test users (>=2 played games): {len(test_user_indices)}\n")
            f.write(f"- Skipped users (insufficient games for hide-one): {skipped_users}\n")
            f.write(f"- Total recommendation attempts (for valid users): {total_attempts}\n")
            f.write(f"- Successful hits (hidden game in top N): {successful_hits}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"- Hit Rate (games found in top {top_n}): {hit_rate_metric:.4f}\n")
            f.write(f"- Mean Rank of Hidden Games (when found): {mean_rank_metric:.2f}\n")
            f.write(f"- Recommendation Opportunity Coverage: {coverage_rate_metric:.4f}\n") # Clarified metric name
            f.write(f"- Total evaluation time: {time.time() - start_time:.2f} seconds\n")
        print(f"Results saved to {evaluation_results_file}")
    except IOError as e:
        print(f"Error saving evaluation results to {evaluation_results_file}: {e}")


    print(f"Evaluation complete in {time.time() - start_time:.2f} seconds")
    return hit_rate_metric, mean_rank_metric, coverage_rate_metric


def build_user_game_matrix(user_file, game_file, output_file):
    """Build user-game matrix from user data."""
    print("Starting matrix construction...")
    start_time = time.time()

    # Load game details
    game_details = load_game_details(game_file)

    # First pass: collect all users and games
    print("First pass: collecting users and games...")
    user_ids_list = [] # Renamed to avoid conflict with game_ids set
    game_ids_set = set() # Renamed for clarity
    user_games_data = []  # List to store (user_idx, game_id_str, playtime_float) tuples

    line_count = 0
    try:
        with open(user_file, 'r') as f:
            for line in f:
                line_count += 1
                if line_count % 10000 == 0: # Increased log frequency
                    print(f"Processed {line_count} user lines...")

                result = process_user_line(line)
                if result is None:
                    continue

                user_id_str, games_playtimes_list = result

                # Ensure user_id_str is added to user_ids_list only once and get its index
                try:
                    current_user_idx = user_ids_list.index(user_id_str)
                except ValueError:
                    user_ids_list.append(user_id_str)
                    current_user_idx = len(user_ids_list) - 1

                for game_id_str, playtime_float in games_playtimes_list:
                    game_ids_set.add(str(game_id_str)) # Ensure game_id is string
                    user_games_data.append((current_user_idx, str(game_id_str), playtime_float))
    except FileNotFoundError:
        print(f"Error: User file {user_file} not found!")
        return None, None, None, None, None # Indicate failure
    except Exception as e:
        print(f"An error occurred during user file processing: {e}")
        return None, None, None, None, None # Indicate failure


    print(f"\nFirst pass complete. Found {len(user_ids_list)} users and {len(game_ids_set)} unique games.")

    # Convert game_ids_set to sorted list and create mapping
    print("\nCreating game ID mapping...")
    game_ids_ordered_list = sorted(list(game_ids_set))
    game_id_to_idx_map = {game_id: idx for idx, game_id in enumerate(game_ids_ordered_list)}

    # Create matrix
    print("\nCreating and filling matrix...")
    matrix = np.zeros((len(user_ids_list), len(game_ids_ordered_list)), dtype=np.float32)

    # Fill matrix
    fill_count = 0
    for i, (user_idx, game_id_str, playtime_float) in enumerate(user_games_data):
        if i % 1000000 == 0: # Increased log frequency
            print(f"Filled {fill_count} of {len(user_games_data)} entries into matrix...")

        game_idx = game_id_to_idx_map.get(game_id_str) # Use .get for safety, though all should be present
        if game_idx is not None:
            matrix[user_idx, game_idx] = playtime_float
            fill_count +=1

    print(f"Matrix filling complete. {fill_count} entries filled.")

    # Save matrix and mappings
    print("\nSaving results...")
    try:
        np.save(output_file + '.matrix.npy', matrix)
        np.save(output_file + '.user_ids.npy', np.array(user_ids_list, dtype=object)) # Save as object array for strings
        np.save(output_file + '.game_ids.npy', np.array(game_ids_ordered_list, dtype=object))
    except IOError as e:
        print(f"Error saving matrix or ID files: {e}")
        # Decide if to proceed or return failure

    # Print statistics
    print("\nFinal statistics:")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Number of users: {len(user_ids_list)}")
    print(f"Number of games: {len(game_ids_ordered_list)}")

    total_time_matrix = time.time() - start_time
    print(f"Matrix construction time: {total_time_matrix:.2f} seconds")

    # Compute and save item similarities
    similar_games = compute_item_similarities(matrix, game_ids_ordered_list, output_file)

    # Generate and save recommendations
    recommendations = generate_recommendations(matrix, user_ids_list, game_ids_ordered_list, similar_games, game_details, output_file)

    # Evaluate the recommender system
    # Ensure all inputs to evaluate_recommender are correct
    hit_rate, mean_rank, coverage = evaluate_recommender(matrix, user_ids_list, game_ids_ordered_list, similar_games, output_file)

    print(f"Total processing time for build_user_game_matrix: {time.time() - start_time:.2f} seconds")
    return matrix, user_ids_list, game_ids_ordered_list, similar_games, recommendations

if __name__ == '__main__':
    user_file = 'australian_users_items.json'
    game_file = 'cleaned_steam_games.json'  # Using the cleaned file instead
    output_file = 'user_game_matrix'
    build_user_game_matrix(user_file, game_file, output_file)

