import urllib.parse

APP_ID = "YOUR_APP_ID"
APP_SECRET = "YOUR_APP_SECRET"
REDIRECT_URI = "https://your-redirect-url.com/oauth/callback"
CANTO_OAUTH_DOMAIN = "https://umd.canto.com"

def get_authorize_url():
    base_url = f"{CANTO_OAUTH_DOMAIN}/oauth2/authorize"
    params = {
        "response_type": "code",
        "app_id": APP_ID,
        "redirect_uri": REDIRECT_URI,
        "state": "myRandomState123",  # or generate a random string
    }
    query = urllib.parse.urlencode(params)
    return f"{base_url}?{query}"

if __name__ == "__main__":
    print("Go to this URL to authorize your app:")
    print(get_authorize_url())
