from typing import List, Dict, Union
from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.posting_api import TwitterAPI

mcp = FastMCP("Posting")

twitter_api = TwitterAPI()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Load a scenario into the TwitterAPI instance.

    Args:
        scenario (dict): A dictionary containing Twitter data.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        twitter_api._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def save_scenario():
    """
    Save current scenario from the TwitterAPI instance.
    """
    try:
        result = twitter_api.save_scenario()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("scenario", {})
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def authenticate_twitter(username: str, password: str):
    """
    Authenticate a user with username and password.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.

    Returns:
        authentication_status (bool): True if authenticated, False otherwise.
    """
    try:
        result = twitter_api.authenticate_twitter(username, password)
        status = result.get('authentication_status', False)
        return f"Authentication status: {'Success' if status else 'Failed'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def posting_get_login_status():
    """
    Get the login status of the current user.

    Returns:
        login_status (bool): True if the current user is logged in, False otherwise.
    """
    try:
        result = twitter_api.posting_get_login_status()
        status = result.get('login_status', False)
        return f"Login status: {'Logged in' if status else 'Not logged in'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def post_tweet(content: str, tags: List[str] = [], mentions: List[str] = []):
    """
    Post a tweet for the authenticated user.

    Args:
        content (str): Content of the tweet.
        tags (List[str]): [Optional] List of tags for the tweet. Tag name should start with #.
        mentions (List[str]): [Optional] List of users mentioned in the tweet. Mention name should start with @.

    Returns:
        id (int): ID of the posted tweet.
        username (str): Username of the poster.
        content (str): Content of the tweet.
        tags (List[str]): List of tags associated with the tweet.
        mentions (List[str]): List of users mentioned in the tweet.
    """
    try:
        result = twitter_api.post_tweet(content, tags, mentions)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Tweet posted successfully!\nID: {result.get('id')}\nUsername: {result.get('username')}\nContent: {result.get('content')}\nTags: {result.get('tags', [])}\nMentions: {result.get('mentions', [])}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def retweet(tweet_id: int):
    """
    Retweet a tweet for the authenticated user.

    Args:
        tweet_id (int): ID of the tweet to retweet.

    Returns:
        retweet_status (str): Status of the retweet action.
    """
    try:
        result = twitter_api.retweet(tweet_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Retweet status: {result.get('retweet_status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def comment(tweet_id: int, comment_content: str):
    """
    Comment on a tweet for the authenticated user.

    Args:
        tweet_id (int): ID of the tweet to comment on.
        comment_content (str): Content of the comment.

    Returns:
        comment_status (str): Status of the comment action.
    """
    try:
        result = twitter_api.comment(tweet_id, comment_content)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Comment status: {result.get('comment_status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def mention(tweet_id: int, mentioned_usernames: List[str]):
    """
    Mention specified users in a tweet.

    Args:
        tweet_id (int): ID of the tweet where users are mentioned.
        mentioned_usernames (List[str]): List of usernames to be mentioned.

    Returns:
        mention_status (str): Status of the mention action.
    """
    try:
        result = twitter_api.mention(tweet_id, mentioned_usernames)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Mention status: {result.get('mention_status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def follow_user(username_to_follow: str):
    """
    Follow a user for the authenticated user.

    Args:
        username_to_follow (str): Username of the user to follow.

    Returns:
        follow_status (bool): True if followed, False if already following.
    """
    try:
        result = twitter_api.follow_user(username_to_follow)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status = result.get('follow_status', False)
        if status:
            return f"Successfully followed user: {username_to_follow}"
        else:
            return f"Already following user: {username_to_follow}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def list_all_following():
    """
    List all users that the authenticated user is following.

    Returns:
        following_list (List[str]): List of all users that the authenticated user is following.
    """
    try:
        result = twitter_api.list_all_following()
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        
        if isinstance(result, list):
            if result:
                return f"Following list: {', '.join(result)}"
            else:
                return "Not following anyone."
        else:
            return f"Following list: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def unfollow_user(username_to_unfollow: str):
    """
    Unfollow a user for the authenticated user.

    Args:
        username_to_unfollow (str): Username of the user to unfollow.

    Returns:
        unfollow_status (bool): True if unfollowed, False if not following.
    """
    try:
        result = twitter_api.unfollow_user(username_to_unfollow)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status = result.get('unfollow_status', False)
        if status:
            return f"Successfully unfollowed user: {username_to_unfollow}"
        else:
            return f"Not following user: {username_to_unfollow}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_tweet(tweet_id: int):
    """
    Retrieve a specific tweet.

    Args:
        tweet_id (int): ID of the tweet to retrieve.

    Returns:
        Tweet information including ID, username, content, tags and mentions.
    """
    try:
        result = twitter_api.get_tweet(tweet_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Tweet details:\nID: {result.get('id')}\nUsername: {result.get('username')}\nContent: {result.get('content')}\nTags: {result.get('tags', [])}\nMentions: {result.get('mentions', [])}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_user_tweets(username: str):
    """
    Retrieve all tweets from a specific user.

    Args:
        username (str): Username of the user whose tweets to retrieve.

    Returns:
        user_tweets (List[Dict]): List of dictionaries, each containing tweet information.
    """
    try:
        result = twitter_api.get_user_tweets(username)
        if not result:
            return f"User {username} has not posted any tweets."
        
        tweets_info = []
        for tweet in result:
            tweets_info.append(f"ID: {tweet.get('id')}, Content: {tweet.get('content')}")
        
        return f"User {username}'s tweets ({len(result)} tweets):\n" + "\n".join(tweets_info)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def search_tweets(keyword: str):
    """
    Search for tweets containing a specific keyword.

    Args:
        keyword (str): Keyword to search for in the content of the tweets.

    Returns:
        matching_tweets (List[Dict]): List of dictionaries, each containing tweet information.
    """
    try:
        result = twitter_api.search_tweets(keyword)
        if not result:
            return f"No tweets found containing keyword '{keyword}'."
        
        tweets_info = []
        for tweet in result:
            tweets_info.append(f"ID: {tweet.get('id')}, User: {tweet.get('username')}, Content: {tweet.get('content')}")
        
        return f"Tweets containing keyword '{keyword}' ({len(result)} tweets):\n" + "\n".join(tweets_info)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_tweet_comments(tweet_id: int):
    """
    Retrieve all comments for a specific tweet.

    Args:
        tweet_id (int): ID of the tweet to retrieve comments for.

    Returns:
        comments (List[Dict]): List of dictionaries, each containing comment information.
    """
    try:
        result = twitter_api.get_tweet_comments(tweet_id)
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        
        if not result:
            return f"Tweet {tweet_id} has no comments."
        
        comments_info = []
        for comment in result:
            comments_info.append(f"User: {comment.get('username')}, Content: {comment.get('content')}")
        
        return f"Comments for tweet {tweet_id} ({len(result)} comments):\n" + "\n".join(comments_info)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_user_stats(username: str):
    """
    Get statistics for a specific user.

    Args:
        username (str): Username of the user to get statistics for.

    Returns:
        Statistics including tweet count, following count and retweet count.
    """
    try:
        result = twitter_api.get_user_stats(username)
        return f"User {username} statistics:\nTweet count: {result.get('tweet_count', 0)}\nFollowing count: {result.get('following_count', 0)}\nRetweet count: {result.get('retweet_count', 0)}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Twitter Posting Server...")
    mcp.run(transport='stdio')