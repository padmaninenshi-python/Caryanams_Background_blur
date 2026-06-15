"""
Standalone Remove Background + Number Plate Tool
Run: python app.py
Visit: http://localhost:5000
"""

import os
from flask import Flask
from extensions import db

def create_app():
    app = Flask(__name__)
    app.secret_key = 'removebg-standalone-secret'

    # SQLite DB
    db_path = os.path.join(app.root_path, 'instance', 'removebg.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # Allow canvas crossOrigin reads for mirror reflection (same-origin images)
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return response

    from routes import bp
    app.register_blueprint(bp)

    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n✅  Remove BG Tool running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
