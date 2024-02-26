from app import create_app
from dotenv import load_dotenv
import os
load_dotenv()

if __name__ == "__main__":
    app = create_app()
    template_folder_value = app.template_folder
    app.run()
