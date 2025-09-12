from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "username": "john",
    "password": "john123",
    "authenticated": False,
    "tweets": {},
    "comments": {},
    "retweets": {},
    "following_list": ["alice", "bob"],
    "tweet_counter": 0,
}

class TwitterAPI:
    def __init__(self):
        self.username: str
        self.password: str
        self.authenticated: bool
        self.tweets: Dict[int, Dict[str, Union[int, str, List[str]]]]
        self.comments: Dict[int, List[Dict[str, str]]]
        self.retweets: Dict[str, List[int]]
        self.following_list: List[str]
        # tweet_counter is used to assign unique IDs to tweets, it might not be the same as the length of the tweets list for different scenarios
        self.tweet_counter: int
        self._api_description = "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        """
        Load a scenario into the TwitterAPI instance.
        Args:
            scenario (dict): A dictionary containing Twitter data.
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.username = scenario.get("username", DEFAULT_STATE_COPY["username"])
        self.password = scenario.get("password", DEFAULT_STATE_COPY["password"])
        self.authenticated = scenario.get(
            "authenticated", DEFAULT_STATE_COPY["authenticated"]
        )
        self.tweets = scenario.get("tweets", DEFAULT_STATE_COPY["tweets"])
        self.tweets = {int(k): v for k, v in self.tweets.items()} # Convert tweet keys from string to int from loaded scenario
        self.comments = scenario.get("comments", DEFAULT_STATE_COPY["comments"])
        self.retweets = scenario.get("retweets", DEFAULT_STATE_COPY["retweets"])
        self.following_list = scenario.get(
            "following_list", DEFAULT_STATE_COPY["following_list"]
        )
        self.tweet_counter = scenario.get(
            "tweet_counter", DEFAULT_STATE_COPY["tweet_counter"]
        )

    def save_scenario(self) -> Dict[str, Union[dict, str]]:
        """
        Save the current TwitterAPI state to a dictionary.

        Returns:
            scenario (Dict): A dictionary containing the current state of the TwitterAPI.
        """
        try:
            scenario = {
                "username": self.username,
                "password": self.password,
                "authenticated": self.authenticated,
                "tweets": self.tweets,
                "comments": self.comments,
                "retweets": self.retweets,
                "following_list": self.following_list,
                "tweet_counter": self.tweet_counter,
            }
            return {"scenario": scenario, "message": "Scenario saved successfully."}
        except Exception as e:
            return {"error": f"Failed to save scenario: {str(e)}"}

    def authenticate_twitter(self, username: str, password: str) -> Dict[str, bool]:
        """
        Authenticate a user with username and password.

        Args:
            username (str): Username of the user.
            password (str): Password of the user.
        Returns:
            authentication_status (bool): True if authenticated, False otherwise.
        """
        if username == self.username and password == self.password:
            self.authenticated = True
            return {"authentication_status": True}
        return {"authentication_status": False}

    def posting_get_login_status(self) -> Dict[str, Union[bool, str]]:
        """
        Get the login status of the current user.

        Returns:
            login_status (bool): True if the current user is logged in, False otherwise.
        """
        return {"login_status": bool(self.authenticated)}

    def post_tweet(
        self, content: str, tags: List[str] = [], mentions: List[str] = []
    ) -> Dict[str, Union[int, str, List[str]]]:
        """
        Post a tweet for the authenticated user.

        Args:
            content (str): Content of the tweet.
            tags (List[str]): [Optional] List of tags for the tweet. Tag name should start with #. This is only relevant if the user wants to add tags to the tweet.
            mentions (List[str]): [Optional] List of users mentioned in the tweet. Mention name should start with @. This is only relevant if the user wants to add mentions to the tweet.
        Returns:
            id (int): ID of the posted tweet.
            username (str): Username of the poster.
            content (str): Content of the tweet.
            tags (List[str]): List of tags associated with the tweet.
            mentions (List[str]): List of users mentioned in the tweet.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please authenticate before posting."}

        tweet = {
            "id": self.tweet_counter,
            "username": self.username,
            "content": content,
            "tags": tags,
            "mentions": mentions,
        }
        self.tweets[self.tweet_counter] = tweet
        self.tweet_counter += 1
        return tweet

    def retweet(self, tweet_id: int) -> Dict[str, str]:
        """
        Retweet a tweet for the authenticated user.

        Args:
            tweet_id (int): ID of the tweet to retweet.
        Returns:
            retweet_status (str): Status of the retweet action.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please authenticate before retweeting."}
                
        if tweet_id not in self.tweets:
            return {"error": f"Tweet with ID {tweet_id} not found."}

        if self.username not in self.retweets:
            self.retweets[self.username] = []

        if tweet_id in self.retweets[self.username]:
            return {"retweet_status": "Already retweeted"}

        self.retweets[self.username].append(tweet_id)
        return {"retweet_status": "Successfully retweeted"}

    def comment(self, tweet_id: int, comment_content: str) -> Dict[str, str]:
        """
        Comment on a tweet for the authenticated user.

        Args:
            tweet_id (int): ID of the tweet to comment on.
            comment_content (str): Content of the comment.
        Returns:
            comment_status (str): Status of the comment action.
        """
        if not self.authenticated:
            raise {"error": "User not authenticated. Please authenticate before commenting."}


        if tweet_id not in self.tweets:
            return {"error": f"Tweet with ID {tweet_id} not found."}

        if tweet_id not in self.comments:
            self.comments[tweet_id] = []

        self.comments[tweet_id].append(
            {"username": self.username, "content": comment_content}
        )
        return {"comment_status": "Comment added successfully"}

    def mention(self, tweet_id: int, mentioned_usernames: List[str]) -> Dict[str, str]:
        """
        Mention specified users in a tweet.

        Args:
            tweet_id (int): ID of the tweet where users are mentioned.
            mentioned_usernames (List[str]): List of usernames to be mentioned.
        Returns:
            mention_status (str): Status of the mention action.
        """
        if tweet_id not in self.tweets:
            return {"error": f"Tweet with ID {tweet_id} not found."}

        tweet = self.tweets[tweet_id]
        tweet["mentions"].extend(mentioned_usernames)

        return {"mention_status": "Users mentioned successfully"}

    def follow_user(self, username_to_follow: str) -> Dict[str, bool]:
        """
        Follow a user for the authenticated user.

        Args:
            username_to_follow (str): Username of the user to follow.
        Returns:
            follow_status (bool): True if followed, False if already following.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please authenticate before following."}

        if username_to_follow in self.following_list:
            return {"follow_status": False}

        self.following_list.append(username_to_follow)
        return {"follow_status": True}

    def list_all_following(self) -> List[str]:
        """
        List all users that the authenticated user is following.

        Returns:
            following_list (List[str]): List of all users that the authenticated user is following.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please authenticate before listing following."}

        return self.following_list

    def unfollow_user(self, username_to_unfollow: str) -> Dict[str, bool]:
        """
        Unfollow a user for the authenticated user.

        Args:
            username_to_unfollow (str): Username of the user to unfollow.
        Returns:
            unfollow_status (bool): True if unfollowed, False if not following.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please authenticate before unfollowing."}

        if username_to_unfollow not in self.following_list:
            return {"unfollow_status": False}

        self.following_list.remove(username_to_unfollow)
        return {"unfollow_status": True}

    def get_tweet(self, tweet_id: int) -> Dict[str, Union[int, str, List[str]]]:
        """
        Retrieve a specific tweet.

        Args:
            tweet_id (int): ID of the tweet to retrieve.
        Returns:
            id (int): ID of the retrieved tweet.
            username (str): Username of the tweet's author.
            content (str): Content of the tweet.
            tags (List[str]): List of tags associated with the tweet.
            mentions (List[str]): List of users mentioned in the tweet.
        """
        if tweet_id not in self.tweets:
            return {"error": f"Tweet with ID {tweet_id} not found."}

        return self.tweets[tweet_id]

    def get_user_tweets(self, username: str) -> List[Dict[str, Union[int, str, List[str]]]]:
        """
        Retrieve all tweets from a specific user.

        Args:
            username (str): Username of the user whose tweets to retrieve.
        Returns:
            user_tweets (List[Dict]): List of dictionaries, each containing tweet information.
                - id (int): ID of the retrieved tweet.
                - username (str): Username of the tweet's author.
                - content (str): Content of the tweet.
                - tags (List[str]): List of tags associated with the tweet.
                - mentions (List[str]): List of users mentioned in the tweet.
        """
        return [tweet for tweet in self.tweets.values() if tweet["username"] == username]

    def search_tweets(self, keyword: str) -> List[Dict[str, Union[int, str, List[str]]]]:
        """
        Search for tweets containing a specific keyword.

        Args:
            keyword (str): Keyword to search for in the content of the tweets.
        Returns:
            matching_tweets (List[Dict]): List of dictionaries, each containing tweet information.
                - id (int): ID of the retrieved tweet.
                - username (str): Username of the tweet's author.
                - content (str): Content of the tweet.
                - tags (List[str]): List of tags associated with the tweet.
                - mentions (List[str]): List of users mentioned in the tweet.
        """
        return [
            tweet
            for tweet in self.tweets.values()
            if keyword.lower() in tweet["content"].lower()
            or keyword.lower() in [tag.lower() for tag in tweet["tags"]]
        ]

    def get_tweet_comments(self, tweet_id: int) -> List[Dict[str, str]]:
        """
        Retrieve all comments for a specific tweet.

        Args:
            tweet_id (int): ID of the tweet to retrieve comments for.
        Returns:
            comments (List[Dict]): List of dictionaries, each containing comment information.
                - username (str): Username of the commenter.
                - content (str): Content of the comment.
        """
        if tweet_id not in self.tweets:
            return {"error": f"Tweet with ID {tweet_id} not found."}
        return self.comments.get(tweet_id, [])

    def get_user_stats(self, username: str) -> Dict[str, int]:
        """
        Get statistics for a specific user.

        Args:
            username (str): Username of the user to get statistics for.
        Returns:
            tweet_count (int): Number of tweets posted by the user.
            following_count (int): Number of users the specified user is following.
            retweet_count (int): Number of retweets made by the user.
        """
        tweet_count = len(
            [tweet for tweet in self.tweets.values() if tweet["username"] == username]
        )
        following_count = len(self.following_list) if username == self.username else 0
        retweet_count = len(self.retweets.get(username, []))

        return {
            "tweet_count": tweet_count,
            "following_count": following_count,
            "retweet_count": retweet_count,
        }
