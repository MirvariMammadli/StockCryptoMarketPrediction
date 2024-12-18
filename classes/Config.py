class Config:
    def __init__(self, reddit_config):
        self.reddit_client_id = reddit_config["reddit_client_id"]
        self.reddit_client_secret = reddit_config["reddit_client_secret"]
        self.reddit_user_agent = reddit_config["reddit_user_agent"]
