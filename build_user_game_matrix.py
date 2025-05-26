import numpy as np
import ast  # For safely evaluating Python literals
import time
from collections import defaultdict

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
        top_indices = np.argsort(game_similarities)[::-1][:top_n+1]
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
        with open(game_file, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")
                try:
                    # Print first line for debugging
                    if line_count == 1:
                        print("\nFirst line of file:")
                        print(line[:200] + "..." if len(line) > 200 else line)
                    
                    game_data = ast.literal_eval(line.strip())
                    if 'id' not in game_data:
                        print(f"Warning: Line {line_count} missing 'id' field")
                        continue
                        
                    game_id = int(game_data['id'])
                    details = {
                        'name': game_data.get('name', 'Unknown'),
                        'publisher': game_data.get('publisher', 'Unknown'),
                        'genres': game_data.get('genres', []),
                        'price': float(game_data.get('price', 0.0)),
                        'release_date': game_data.get('release_date', 'Unknown'),
                        'developer': game_data.get('developer', 'Unknown'),
                        'tags': game_data.get('tags', []),
                        'specs': game_data.get('specs', [])
                    }
                    game_details[game_id] = details
                except (ValueError, SyntaxError, KeyError) as e:
                    print(f"Error processing line {line_count}: {str(e)}")
                    print(f"Problematic line: {line[:100]}...")
                    continue
                
        print(f"\nLoaded details for {len(game_details)} games")
        
        # Print first entry
        if game_details:
            first_game_id = next(iter(game_details))
            print("\nFirst game entry:")
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

def generate_recommendations(matrix, user_ids, game_ids, similar_games, game_details, output_file, top_n=10, min_playtime_threshold=0.5):
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
            recommendations[user_id] = {
                'recommendations': [game_id for game_id, _ in top_games],
                'scores': [score for _, score in top_games]
            }
    
    # Save recommendations in a readable format
    print("\nSaving recommendations...")
    with open(output_file + '.recommendations.txt', 'w') as f:
        for user_id, data in recommendations.items():
            f.write(f"User {user_id}:\n")
            f.write("=" * 50 + "\n")
            
            # Write user's played games
            user_playtimes = matrix[user_ids.index(user_id)]
            played_games = [(game_ids[i], user_playtimes[i]) for i in np.where(user_playtimes > min_playtime_threshold)[0]]
            played_games.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Currently Playing:\n")
            for game_id, playtime in played_games[:5]:  # Show top 5 played games
                details = game_details.get(game_id, {})
                f.write(f"  - {details.get('name', f'Game {game_id}')}\n")
                f.write(f"    Playtime: {playtime:.2f}\n")
                f.write(f"    Genres: {', '.join(details.get('genres', ['Unknown']))}\n")
            f.write("\n")
            
            # Write recommendations
            f.write("Recommended Games:\n")
            for game_id, score in zip(data['recommendations'], data['scores']):
                details = game_details.get(game_id, {})
                f.write(f"  - {details.get('name', f'Game {game_id}')}\n")
                f.write(f"    Score: {score:.4f}\n")
                f.write(f"    Publisher: {details.get('publisher', 'Unknown')}\n")
                f.write(f"    Genres: {', '.join(details.get('genres', ['Unknown']))}\n")
                f.write(f"    Price: ${details.get('price', 0.0):.2f}\n")
                f.write(f"    Release Date: {details.get('release_date', 'Unknown')}\n")
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"\nRecommendation generation complete in {time.time() - start_time:.2f} seconds")
    return recommendations

def evaluate_recommender(matrix, user_ids, game_ids, similar_games, output_file, top_n=10, min_playtime_threshold=0.5, test_fraction=0.1):
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
        
        print(f"\nProcessing users {batch_start+1}-{batch_end} of {len(test_user_indices)}...")
        
        # Get playtime vectors for the batch
        batch_playtimes = matrix[batch_indices]
        
        for i, user_idx in enumerate(batch_indices):
            user_id = user_ids[user_idx]
            user_playtimes = batch_playtimes[i]
            
            # Find games the user has played significantly
            played_games = np.where(user_playtimes > min_playtime_threshold)[0]
            
            if len(played_games) < 2:
                skipped_users += 1
                continue
            
            # Randomly select one game to hide
            hidden_game_idx = np.random.choice(played_games)
            hidden_game_id = game_ids[hidden_game_idx]
            hidden_game_playtime = user_playtimes[hidden_game_idx]
            
            # Create a copy of the matrix with the hidden game removed
            test_matrix = matrix.copy()
            test_matrix[user_idx, hidden_game_idx] = 0
            
            # Calculate recommendation scores using vectorized operations
            played_mask = user_playtimes > min_playtime_threshold
            played_mask[hidden_game_idx] = False  # Exclude hidden game
            
            # Get similarities for played games
            played_similarities = similarity_matrix[played_mask]
            
            # Calculate scores for all games
            scores = np.zeros(len(game_ids))
            for played_idx in np.where(played_mask)[0]:
                playtime = user_playtimes[played_idx]
                scores += similarity_matrix[played_idx] * playtime
            
            # Zero out scores for played games
            scores[played_mask] = 0
            
            # Get top N recommendations
            top_indices = np.argsort(scores)[::-1][:top_n]
            recommended_games = [game_ids[idx] for idx in top_indices]
            
            # Check if hidden game is in recommendations
            try:
                rank = recommended_games.index(hidden_game_id) + 1
                hit = 1 if rank <= top_n else 0
                hit_rates.append(hit)
                ranks.append(rank)
                coverage.append(1)
                
                if hit:
                    successful_hits += 1
            except ValueError:
                coverage.append(0)
            
            total_attempts += 1
    
    # Calculate metrics
    hit_rate = np.mean(hit_rates) if hit_rates else 0
    mean_rank = np.mean(ranks) if ranks else float('inf')
    coverage_rate = np.mean(coverage) if coverage else 0
    
    # Save evaluation results
    print("\n=== Evaluation Results ===")
    with open(output_file + '.evaluation.txt', 'w') as f:
        f.write("Recommender System Evaluation Results:\n")
        f.write("=====================================\n\n")
        f.write("Parameters:\n")
        f.write(f"- Top N recommendations: {top_n}\n")
        f.write(f"- Minimum playtime threshold: {min_playtime_threshold}\n")
        f.write(f"- Test fraction: {test_fraction}\n\n")
        
        f.write("Test Statistics:\n")
        f.write(f"- Total test users: {len(test_user_indices)}\n")
        f.write(f"- Skipped users (insufficient games): {skipped_users}\n")
        f.write(f"- Total recommendation attempts: {total_attempts}\n")
        f.write(f"- Successful hits: {successful_hits}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"- Hit Rate (games found in top {top_n}): {hit_rate:.4f}\n")
        f.write(f"- Mean Rank of Hidden Games: {mean_rank:.2f}\n")
        f.write(f"- Coverage Rate: {coverage_rate:.4f}\n")
        f.write(f"- Total evaluation time: {time.time() - start_time:.2f} seconds\n")
    
    print(f"Results saved to {output_file}.evaluation.txt")
    print(f"Evaluation complete in {time.time() - start_time:.2f} seconds")
    return hit_rate, mean_rank, coverage_rate

def build_user_game_matrix(user_file, game_file, output_file):
    """Build user-game matrix from user data."""
    print("Starting matrix construction...")
    start_time = time.time()
    # Load game details
    game_details = load_game_details(game_file)
    # First pass: collect all users and games
    print("First pass: collecting users and games...")
    user_ids = []
    game_ids = set()
    user_games = []  # List to store (user_idx, game_id, playtime) tuples
    line_count = 0
    with open(user_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 1000 == 0:
                print(f"Processed {line_count} lines...")
            result = process_user_line(line)
            if result is None:
                continue
            user_id, games = result
            user_ids.append(user_id)
            for game_id, playtime in games:
                game_ids.add(game_id)
                user_games.append((len(user_ids) - 1, game_id, playtime))

    print(f"\nFirst pass complete. Found {len(user_ids)} users and {len(game_ids)} unique games.")
    
    # Convert game_ids to sorted list and create mapping
    print("\nCreating game ID mapping...")
    game_ids = sorted(list(game_ids))
    game_id_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}
    
    # Create matrix
    print("\nCreating and filling matrix...")
    matrix = np.zeros((len(user_ids), len(game_ids)), dtype=np.float32)
    
    # Fill matrix
    for i, (user_idx, game_id, playtime) in enumerate(user_games):
        if i % 100000 == 0:
            print(f"Filled {i} entries...")
        game_idx = game_id_to_idx[game_id]
        matrix[user_idx, game_idx] = playtime
    
    # Save matrix and mappings
    print("\nSaving results...")
    np.save(output_file + '.matrix.npy', matrix)
    np.save(output_file + '.user_ids.npy', np.array(user_ids))
    np.save(output_file + '.game_ids.npy', np.array(game_ids))
    
    # Print statistics
    print("\nFinal statistics:")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Number of users: {len(user_ids)}")
    print(f"Number of games: {len(game_ids)}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    # Compute and save item similarities
    similar_games = compute_item_similarities(matrix, game_ids, output_file)
    C
    # Generate and save recommendations
    recommendations = generate_recommendations(matrix, user_ids, game_ids, similar_games, game_details, output_file)
    
    # Evaluate the recommender system
    hit_rate, mean_rank, coverage = evaluate_recommender(matrix, user_ids, game_ids, similar_games, output_file)
    
    return matrix, user_ids, game_ids, similar_games, recommendations

if __name__ == '__main__':
    user_file = 'australian_users_items.json'
    game_file = 'cleaned_steam_games.txt'  # Using the cleaned file instead
    output_file = 'user_game_matrix'
    build_user_game_matrix(user_file, game_file, output_file) 