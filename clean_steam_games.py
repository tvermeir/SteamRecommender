import ast
import re
import json

def clean_string(s):
    if not isinstance(s, str):
        return str(s)
    
    s = ''.join(c for c in s if c.isprintable())
   
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_price(price):
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        if price.lower() in ['free to play', 'free']:
            return 0.0
    
        match = re.search(r'\d+\.?\d*', price)
        if match:
            return float(match.group())
    return 0.0

def clean_game_data(input_file, output_file):
    processed_games = 0
    total_games = 0
    cleaned_games = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_games += 1
            try:
               
                game = ast.literal_eval(line.strip())

               
                cleaned_game = {
                    'id': game.get('id', ''),
                    'name': clean_string(game.get('app_name', '')),
                    'title': clean_string(game.get('title', game.get('app_name', ''))),
                    'publisher': clean_string(game.get('publisher', 'Unknown')),
                    'developer': clean_string(game.get('developer', 'Unknown')),
                    'genres': [clean_string(g) for g in game.get('genres', [])],
                    'tags': [clean_string(t) for t in game.get('tags', [])],
                    'price': clean_price(game.get('price', 0.0)),
                    'discount_price': clean_price(game.get('discount_price', 0.0)),
                    'release_date': game.get('release_date', ''),
                    'early_access': game.get('early_access', False),
                    'specs': [clean_string(s) for s in game.get('specs', [])],
                    'url': game.get('url', ''),
                    'reviews_url': game.get('reviews_url', ''),
                    'sentiment': game.get('sentiment', '')
                }

                
                if cleaned_game['id'] and cleaned_game['name']:
                    cleaned_games.append(cleaned_game)
                    processed_games += 1

            except (ValueError, SyntaxError) as e:
                print(f"Error processing game: {e}")
                continue

    with open(output_file, 'w', encoding='utf-8') as outf:
        json.dump(cleaned_games, outf, ensure_ascii=False, indent=2)
    print(f"Processed {processed_games} out of {total_games} games")

if __name__ == '__main__':
    input_file = 'steam_games.json'
    output_file = 'cleaned_steam_games.json'
    clean_game_data(input_file, output_file)

