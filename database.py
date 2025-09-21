import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, List, Tuple

class DatabaseManager:
    def __init__(self, db_path: str = "frame_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT,
                google_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES Users (id)
            )
        ''')
        
        # Nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                node_index INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES Projects (id)
            )
        ''')
        
        # Members table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                start_node INTEGER NOT NULL,
                end_node INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES Projects (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, name: str, email: str, password: str = None, google_id: str = None) -> Optional[int]:
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            hashed_password = self.hash_password(password) if password else None
            cursor.execute('''
                INSERT INTO Users (name, email, password, google_id)
                VALUES (?, ?, ?, ?)
            ''', (name, email, hashed_password, google_id))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None  # User already exists
        finally:
            conn.close()
    
    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user with email and password"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        hashed_password = self.hash_password(password)
        cursor.execute('''
            SELECT id, name, email FROM Users 
            WHERE email = ? AND password = ?
        ''', (email, hashed_password))
        
        user = cursor.fetchone()
        if user:
            # Update last login
            cursor.execute('''
                UPDATE Users SET last_login = ? WHERE id = ?
            ''', (datetime.now(), user[0]))
            conn.commit()
            
            result = {
                'id': user[0],
                'name': user[1],
                'email': user[2]
            }
        else:
            result = None
        
        conn.close()
        return result
    
    def get_user_by_google_id(self, google_id: str) -> Optional[dict]:
        """Get user by Google ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, email FROM Users WHERE google_id = ?
        ''', (google_id,))
        
        user = cursor.fetchone()
        if user:
            # Update last login
            cursor.execute('''
                UPDATE Users SET last_login = ? WHERE id = ?
            ''', (datetime.now(), user[0]))
            conn.commit()
            
            result = {
                'id': user[0],
                'name': user[1],
                'email': user[2]
            }
        else:
            result = None
        
        conn.close()
        return result
    
    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, email, google_id FROM Users WHERE email = ?
        ''', (email,))
        
        user = cursor.fetchone()
        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'google_id': user[3]
            }
        return None
    
    def create_project(self, user_id: int, name: str, description: str = "") -> Optional[int]:
        """Create a new project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO Projects (user_id, name, description)
            VALUES (?, ?, ?)
        ''', (user_id, name, description))
        
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return project_id
    
    def get_user_projects(self, user_id: int) -> List[dict]:
        """Get all projects for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, created_at, updated_at 
            FROM Projects WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        projects = []
        for row in cursor.fetchall():
            projects.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'created_at': row[3],
                'updated_at': row[4]
            })
        
        conn.close()
        return projects
    
    def save_project_data(self, project_id: int, nodes: List[Tuple], members: List[Tuple]):
        """Save nodes and members for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM Nodes WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM Members WHERE project_id = ?', (project_id,))
        
        # Insert nodes
        for i, (x, y, z) in enumerate(nodes):
            cursor.execute('''
                INSERT INTO Nodes (project_id, x, y, z, node_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, x, y, z, i + 1))
        
        # Insert members
        for start_node, end_node in members:
            cursor.execute('''
                INSERT INTO Members (project_id, start_node, end_node)
                VALUES (?, ?, ?)
            ''', (project_id, start_node, end_node))
        
        # Update project timestamp
        cursor.execute('''
            UPDATE Projects SET updated_at = ? WHERE id = ?
        ''', (datetime.now(), project_id))
        
        conn.commit()
        conn.close()
    
    def load_project_data(self, project_id: int) -> Tuple[List[Tuple], List[Tuple]]:
        """Load nodes and members for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Load nodes
        cursor.execute('''
            SELECT x, y, z FROM Nodes 
            WHERE project_id = ? 
            ORDER BY node_index
        ''', (project_id,))
        nodes = [tuple(row) for row in cursor.fetchall()]
        
        # Load members
        cursor.execute('''
            SELECT start_node, end_node FROM Members 
            WHERE project_id = ?
        ''', (project_id,))
        members = [tuple(row) for row in cursor.fetchall()]
        
        conn.close()
        return nodes, members
    
    def delete_project(self, project_id: int, user_id: int) -> bool:
        """Delete a project (only if owned by user)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Verify ownership
        cursor.execute('''
            SELECT id FROM Projects WHERE id = ? AND user_id = ?
        ''', (project_id, user_id))
        
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Delete project data
        cursor.execute('DELETE FROM Nodes WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM Members WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM Projects WHERE id = ?', (project_id,))
        
        conn.commit()
        conn.close()
        return True 