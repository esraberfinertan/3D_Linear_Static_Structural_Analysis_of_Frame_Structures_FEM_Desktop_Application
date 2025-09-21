import os
import json
import threading
import webbrowser
from flask import Flask, request, redirect, session, jsonify
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import time


class GoogleAuthServer:
    def __init__(self, client_secrets_file=None):
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.user_info = None
        self.auth_complete = False
        self.server_thread = None
        
        # Google OAuth configuration
        self.client_secrets_file = client_secrets_file or self.create_default_client_secrets()
        self.scopes = ['openid', 'email', 'profile']
        self.redirect_uri = 'http://localhost:5000/callback'
        
        self.setup_routes()
    
    def create_default_client_secrets(self):
        """Create a default client secrets configuration"""
        # Note: In a real application, you should create a proper Google OAuth app
        # and download the client_secrets.json file from Google Cloud Console
        
        secrets = {
            "web": {
                "client_id": "YOUR ID",
                "client_secret": "YOUR SECRET",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": ["http://localhost:5000/callback"]
            }
        }
        
        # Save to file
        with open("client_secrets.json", "w") as f:
            json.dump(secrets, f)
        
        return "client_secrets.json"
    
    def setup_routes(self):
        @self.app.route('/auth')
        def auth():
            try:
                flow = Flow.from_client_secrets_file(
                    self.client_secrets_file,
                    scopes=self.scopes,
                    redirect_uri=self.redirect_uri
                )
                
                authorization_url, state = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true'
                )
                
                session['state'] = state
                return redirect(authorization_url)
            except Exception as e:
                return f"Error initiating OAuth: {str(e)}", 500
        
        @self.app.route('/callback')
        def callback():
            try:
                # For demo purposes, return mock user data if real OAuth fails
                if not os.path.exists(self.client_secrets_file) or self.is_demo_config():
                    # Mock successful authentication for demo
                    self.user_info = {
                        'id': 'demo_google_id_123',
                        'name': 'Demo User',
                        'email': 'demo@example.com'
                    }
                    self.auth_complete = True
                    return '''
                    <html>
                        <body>
                            <h2>Demo Authentication Successful!</h2>
                            <p>You have been authenticated as Demo User (demo@example.com)</p>
                            <p>You can now close this window and return to the application.</p>
                            <script>window.close();</script>
                        </body>
                    </html>
                    '''
                
                # Real OAuth flow
                flow = Flow.from_client_secrets_file(
                    self.client_secrets_file,
                    scopes=self.scopes,
                    redirect_uri=self.redirect_uri,
                    state=session['state']
                )
                
                flow.fetch_token(authorization_response=request.url)
                
                credentials = flow.credentials
                idinfo = id_token.verify_oauth2_token(
                    credentials.id_token,
                    Request(),
                    flow.client_config['client_id']
                )
                
                self.user_info = {
                    'id': idinfo['sub'],
                    'name': idinfo['name'],
                    'email': idinfo['email']
                }
                self.auth_complete = True
                
                return '''
                <html>
                    <body>
                        <h2>Authentication Successful!</h2>
                        <p>You can now close this window and return to the application.</p>
                        <script>window.close();</script>
                    </body>
                </html>
                '''
            except Exception as e:
                return f"Authentication failed: {str(e)}", 400
        
        @self.app.route('/status')
        def status():
            return jsonify({
                'complete': self.auth_complete,
                'user_info': self.user_info
            })
        
        @self.app.route('/shutdown')
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                return 'Not running with the Werkzeug Server'
            func()
            return 'Server shutting down...'
    
    def is_demo_config(self):
        """Check if using demo configuration"""
        try:
            with open(self.client_secrets_file, 'r') as f:
                config = json.load(f)
                return config['web']['client_id'] == "YOUR ONE"
        except:
            return True
    
    def start_auth_flow(self):
        """Start the authentication flow"""
        self.user_info = None
        self.auth_complete = False
        
        # Start Flask server in a separate thread
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(port=5000, debug=False, use_reloader=False)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(1)
        
        # Open browser to start auth flow
        webbrowser.open('http://localhost:5000/auth')
    
    def wait_for_auth(self, timeout=60):
        """Wait for authentication to complete"""
        start_time = time.time()
        while not self.auth_complete and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        return self.user_info if self.auth_complete else None
    
    def stop_server(self):
        """Stop the Flask server"""
        try:
            import requests
            requests.get('http://localhost:5000/shutdown', timeout=1)
        except:
            pass  # Server might already be down

def authenticate_with_google():
    """Main function to authenticate with Google"""
    auth_server = GoogleAuthServer()
    
    try:
        print("Starting Google authentication...")
        auth_server.start_auth_flow()
        print("Browser opened. Please complete authentication...")
        
        user_info = auth_server.wait_for_auth()
        if user_info:
            print(f"Authentication successful! Welcome {user_info['name']}")
            return user_info
        else:
            print("Authentication failed or timed out.")
            return None
    finally:
        auth_server.stop_server()

if __name__ == "__main__":
    # Test the Google authentication
    user_info = authenticate_with_google()
    if user_info:
        print("User Info:", user_info) 