import praw
import os
import pandas as pd

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="MmsK7PahE1pwyzbLF2wBKQ",
    client_secret="zAFr9cMvdk1TacfbKiuLL2HH5I0X6A",
    user_agent="mmmirvari"
)

# Function to fetch Reddit headlines
def fetch_reddit_headlines(subreddit_name, limit, save_path):
    print(f"Fetching {limit} headlines from r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    headlines = []
    for post in subreddit.hot(limit=limit):
        headlines.append({'title': post.title, 'score': post.score})

    df = pd.DataFrame(headlines)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(f"{save_path}/{subreddit_name}_headlines.csv", index=False)
    print(f"Headlines saved at {save_path}/{subreddit_name}_headlines.csv")

# Example Usage
if __name__ == "__main__":
    fetch_reddit_headlines("stockmarket", 100, "data/sentiment/raw")
